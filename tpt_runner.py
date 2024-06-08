import sys
from tqdm import tqdm
import pathlib
import argparse
import wandb
from copy import deepcopy
from enum import Enum

import torch
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from imagenetv2_pytorch import ImageNetV2Dataset
import numpy as np

from omegaconf import OmegaConf

sys.path.append('../')
from factory import PretrainedTPTHFOpenCLIPFactory
from evaluator.evaluator import ZeroShotImageNetEvaluator
from evaluator.imagenet_config import simple_prompts, ensemble_prompts, imagenet_classes
from evaluator.imagenet_variant_config import imagenet_a_classes, imagenet_r_classes

from misc.config import get_config
from misc.logger import get_logger
from misc.tpt_transforms import AugMixAugmenter

def get_args_parser():
    parser = argparse.ArgumentParser('Tuning hyper parameters used in LoRA for TTT', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, help='path to a config file')
    parser.add_argument('--reconst', choices=['pixel', 'feature'], help='a kind of reconstruction')
    parser.add_argument('--opts', help="Modify config options by adding 'KEY=VALUE' list. ", default=None, nargs='+')
    parser.add_argument('--wandb', action='store_true')
    return parser

def main():

    args = get_args_parser()
    args = args.parse_args()
    cfg = get_config(args)

    device = 'cuda'

    if cfg.output:
        pathlib.Path(cfg.output).mkdir(parents=True, exist_ok=True)

    logger = get_logger()

    if not cfg.train.lr:
        cfg.train.lr = cfg.train.base_lr * cfg.data.batch_size * world_size / 256 # 1e-3 * 64 / 256 = 0.00025

    if cfg.wandb:
        import wandb
        run = wandb.init(project=cfg.wandb_project,
                         entity="ykojima",
                         dir=cfg.output,
                         config=OmegaConf.to_container(cfg, resolve=True))
        cfg = OmegaConf.create(dict(wandb.config))
        cfg.output = pathlib.Path(run.dir) / "../check/"
        cfg.output.mkdir(parents=True, exist_ok=True)
    else:
        wandb = None 

    logger.info(cfg)

    # [NOTE]: factory used in this script is only for Hugging Face.
    factory = PretrainedTPTHFOpenCLIPFactory(cfg.model, mae=cfg.reconst)
    clip, tokenizer, transform = factory.create()
    clip = clip.to(device)
    status = clip.state_dict()

    logger.info('parameters')
    for name, param in clip.named_parameters():
        param.requires_grad = False
        if ('prompt_learner.ctx' in name):
            param.requires_grad = True
        logger.info(f'{name}: {param.requires_grad}')

    # [NOTE]: initial evaluation
    dataset = ImageNetV2Dataset(transform=transform('valid')) 
    evaluator = ZeroShotImageNetEvaluator(tokenizer, dataset, ensemble_prompts, imagenet_classes, device)
    eval_stats = evaluator(clip)
    print(eval_stats)


    datasets = {}
    for ds in cfg.data.dataset['tta']:
        datasets[ds] = cfg.data.dataset.meta[ds]
    run_tpt(clip, status, datasets, cfg.ttt, device)


def select_confident_samples(logits, top):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
    return logits[idx], idx

def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True) # logits = outputs.log_softmax(dim=1) [N, 1000]
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0]) # avg_logits = logits.mean(0) [1, 1000]
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)


def run_tpt(clip, status, datasets, config, device):
    '''This function is based on https://github.com/azshue/TPT/blob/main/tpt_classification.py#L54-L84
    ''' 

    # [NOTE]: fixed parameter for the reproduction
    batch_size = 64
    lr = 0.005
    num_workers = 4

    # [TODO]: shoud be changed depending on the dataset
    classnames = imagenet_classes 
    arch = 'ViT-B/16'

    # [TODO]: How do I reset the clip in TPT?
    # clip.load_state_dict(status)

    trainable_param = clip.prompt_learner.parameters()
    optimizer = torch.optim.AdamW(trainable_param, lr)
    optim_state = deepcopy(optimizer.state_dict())

    # setup automatic mixed-precision (Amp) loss scaling
    scaler = torch.cuda.amp.GradScaler(init_scale=1000)

    print('=> Using native Torch AMP. Training in mixed precision.')

    cudnn.benchmark = True

    # norm stats from clip.load()
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])

    base_transform = transforms.Compose([
                transforms.Resize(224, interpolation=BICUBIC),
                transforms.CenterCrop(224)])
    preprocess = transforms.Compose([
                transforms.ToTensor(),
                normalize])
    tta_transform = AugMixAugmenter(base_transform, preprocess, n_views=batch_size-1, 
                                     augmix=False)

    batchsize = 1


    datasets_stats = {}
    for name, dataset in datasets.items():

        print(f"evaluating: {name}")

        if dataset['classes'] == 'imagenet':
            classes = imagenet_classes
        elif dataset['classes'] == 'imagenet_a':
            classes = imagenet_a_classes
        elif dataset['classes'] == 'imagenet_r':
            classes = imagenet_r_classes
        else:
            raise TypeError

        clip.reset_classnames(classes, arch)
        data_root = pathlib.Path(dataset['path'])
        tta_dataset = torchvision.datasets.ImageFolder(root=data_root, transform=tta_transform)
        # [NOTE]: batch size is 1 since this is single sample TTA.
        tta_data_loader = torch.utils.data.DataLoader(
                    tta_dataset,
                    batch_size=batchsize, shuffle=True,
                    num_workers=num_workers, pin_memory=True)

        top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
        top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)

        progress = ProgressMeter(
            len(tta_data_loader),
            [top1, top5],
            prefix='Test: ')

        # reset model and switch to evaluate mode
        clip.eval()
        with torch.no_grad():
            clip.reset()
 
        for i, (images, target) in tqdm(enumerate(tta_data_loader)):
            for k in range(len(images)):
                images[k] = images[k].to(device)
            target = target.to(device)
            image = images[0]
            images = torch.cat(images, dim=0)
    
            # reset the tunable prompt to its initial state
            if config.epochs > 0:
                with torch.no_grad():
                    clip.reset()

            # [NOTE]: I don't know why optimizer is loaded here.
            optimizer.load_state_dict(optim_state)
            test_time_tuning(clip, images, optimizer, scaler, config)

            # [NOTE]: inference 
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    output = clip(image)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
                
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))

            if (i+1) % 200 == 0:
                progress.display(i)

        progress.display_summary()


def test_time_tuning(clip, inputs, optimizer, scaler, config):

    # [NOTE]: fixed parameters for the reproduction
    selection_p = 0.1

    selected_idx = None
    for j in range(config.epochs):
        with torch.cuda.amp.autocast():
            clip = clip.to('cuda')
            output = clip(inputs) 

            if selected_idx is not None:
                output = output[selected_idx]
            else:
                output, selected_idx = select_confident_samples(output, selection_p)

            loss = avg_entropy(output)
            # print('loss: ', loss)
 
        optimizer.zero_grad()
        # compute gradient and do SGD step
        scaler.scale(loss).backward()
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.step(optimizer)
        scaler.update()
    return

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k
       This function comes from https://github.com/azshue/TPT/blob/main/utils/tools.py#L88-L102
    """

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res 

class Summary(Enum):
    NONE = 0 
    AVERAGE = 1 
    SUM = 2 
    COUNT = 3 

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt 
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0 
        self.avg = 0 
        self.sum = 0 
        self.count = 0 

    def update(self, val, n=1):
        self.val = val 
        self.sum += val * n 
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)
 
class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
    
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1)) 
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']' 

if __name__ == "__main__":
    main()
