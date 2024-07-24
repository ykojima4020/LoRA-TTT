import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import  pathlib
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from factory import PretrainedHFOpenCLIPFactory
from tta import MEMLoss, MAELoss, MAEMEMLoss, accuracy
from tta.tta_with_gradcam import LoRATTARunnerWithGradCAM
from evaluator.imagenet_config import simple_prompts, ensemble_prompts, imagenet_classes
from evaluator.imagenet_variant_config import imagenet_a_classes, imagenet_r_classes
from misc.config import load_config
from omegaconf import OmegaConf
from misc.tpt_transforms import AugMixAugmenter
from misc.config import get_config

import sys
sys.path.append('./external/CLIP_Explainability/code/')
from image_utils import show_cam_on_image

import copy

def get_args_parser():
    parser = argparse.ArgumentParser('Run GradCAM', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, help='path to a config file')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--opts', help="Modify config options by adding 'KEY=VALUE' list. ", default=None, nargs='+')
    return parser
    
 
def main():

    args = get_args_parser()
    args = args.parse_args()
    config = get_config(args)

    if config.wandb:
        import wandb
        run = wandb.init(project=config.wandb_project,
                         entity="ykojima",
                         dir=config.output,
                         config=OmegaConf.to_container(config, resolve=True))
        config = OmegaConf.create(dict(wandb.config))
        config.output = pathlib.Path(run.dir) / "../check/"
        config.output.mkdir(parents=True, exist_ok=True)
    else:
        wandb = None 
 
    # configurations
    # config = './config/embed_run.yaml'
    device = 'cuda'
    # config = load_config(config)
    # OmegaConf.set_struct(config, True)

    factory = PretrainedHFOpenCLIPFactory(config.model, mae=config.reconst)
    model, tokenizer, transform = factory.create()
    model = model.to(device)
    # without fine-tuning, deep copy is really important because state_dict is refered and can be changed during TTA.
    if config.finetune:
        status = torch.load(config.checkpoint, map_location=device)
        print(f'{config.checkpoint} is loaded.')
    else:
        status = {'model': copy.deepcopy(model.mae.state_dict())}

    # [NOTE]: Data augmentation
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])
    base_transform = transforms.Compose([
                    transforms.Resize(224, interpolation=BICUBIC),
                    transforms.CenterCrop(224)])
    preprocess = transforms.Compose([
                    transforms.ToTensor(),
                    normalize])
    batch_size = 64
    tta_transform = AugMixAugmenter(base_transform, preprocess, n_views=batch_size-1,
                                   augmix=False)

    # dataset
    tta_datasets = {}
    for ds in config.data.dataset['tta']:
        tta_datasets[ds] = config.data.dataset.meta[ds]

    tta_config = {}
    for m in config.tta['params']:
        tta_config[m] = config.tta[m]
    tta_config = OmegaConf.create(tta_config)


    stats = {}
    for name, dataset in tta_datasets.items():
        data_root = pathlib.Path(dataset['path'])

        if dataset['prompt'] == 'simple':
            prompts = simple_prompts
        elif dataset['prompt'] == 'ensemble':
            prompts = ensemble_prompts
        else:
            prompts = eval(f"{dataset['prompt']}_prompts")

        if 'imagenet' in dataset['classes']:
            if dataset['classes'] == 'imagenet':
                classes = imagenet_classes
            elif dataset['classes'] == 'imagenet_a':
                classes = imagenet_a_classes
            elif dataset['classes'] == 'imagenet_r':
                classes = imagenet_r_classes
            else:
                raise TypeError
        else:
            classes = eval("{}_classes".format(dataset['classes']))
            # [NOTE]: remove under bars in classes according to TPT or C-TPT.
            classes = [cls.replace('_', ' ') for cls in classes]

        # [TODO]: Choose TTA algorithm here.
        if ('peft' in tta_config.keys()) and ('tpt' in tta_config.keys()):
            raise NotImplementedError
        elif not ('peft' in tta_config.keys()) and ('tpt' in tta_config.keys()):
            # [NOTE]: MEM for updating Text Prompts
            tta_runner = TPTTTARunner(tta_config['tpt'])
        elif ('peft' in tta_config.keys()) and not ('tpt' in tta_config.keys()):
            loss = tta_config['peft']['loss']
            if ('mem' in loss) and not ('mae' in loss):
                # [NOTE]: MEM for updating LoRA
                loss = MEMLoss()
            elif not ('mem' in loss) and ('mae' in loss):
                # [NOTE]: MAE for updating LoRA
                loss = MAELoss()
            elif ('mem' in loss) and ('mae' in loss):
                # [NOTE]: MAE + MEM for updating LoRA
                loss = MAEMEMLoss(tta_config['peft']['mae']['weight'],
                                  tta_config['peft']['mem']['weight'])
            if config.finetune:
                save_dir = f'./gradcam/{name}/{loss.__class__.__name__}_with_finetune/'
            else:
                save_dir = f'./gradcam/{name}/{loss.__class__.__name__}/'
            tta_runner = LoRATTARunnerWithGradCAM(tta_config['peft'], loss, save_dir=save_dir)
        else:
            raise TypeError


        if 'imagenet' in dataset['classes']:
            tta_data = ImageFolderWithPaths(root=data_root, transform=tta_transform)
        elif dataset['classes'] == 'aircraft':
            tta_data = Aircraft(data_root, 'test', None, tta_transform)
        else:
            tta_data = BaseJsonDataset(data_root, dataset['label'], 'test', None, tta_transform)


        top1_before_tta, top1_after_tta = tta_runner(factory, status['model'], tta_data,
                                                    prompts, classes)

        stats.update({name: {'top1_before_tta': top1_before_tta,
                             'top1_after_tta': top1_after_tta}})

        if config.wandb:
            wandb.log(stats)
 
        print(f'top1 before TTA: {top1_before_tta}')
        print(f'top1 after TTA: {top1_after_tta}')


class TruthClassifier():
    def __init__(self):
        pass

    def __call__(self, model, image, text_embeddings, target):
        model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                image_features = model.clip.image_encode(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                output = image_features @ text_embeddings
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        res = True if acc1 == 100.0 else False
        return res

from torchvision.datasets import ImageFolder

class ImageFolderWithPaths(ImageFolder):
    def __getitem__(self, index):
        # 標準の ImageFolder データセットの機能を使用してデータとラベルを取得
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # データパスを取得
        path = self.imgs[index][0]
        # データ、ラベル、パスを返す
        return original_tuple + (path,)

if __name__ == "__main__":
    main()

