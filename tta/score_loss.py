'''
This script aims to do test time training proposed by Test-Time Training with Masked Autoencoders.
This implementation tries to be the same as that.
'''

import argparse
import torch
import torchvision
import wandb
from tqdm import tqdm
import pathlib

import sys
sys.path.append('../')
from ttt import TestTimeTrainer
from evaluator.evaluator import ZeroShotImageNetEvaluator
from factory import PretrainedHFOpenCLIPFactory
from evaluator import imagenet_config
from misc.config import load_config
from omegaconf import OmegaConf

from misc.utils import AvgMeter

import numpy as np

columns = ['lr', 'weight_decay', 'batch_size', 'optimizer', 'severity', 'corruption',
           'epoch', 'score', 'loss', 'top1', 'top5']

corruptions_name = ['brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog',
                    'gaussian_noise', 'glass_blur', 'impulse_noise', 'jpeg_compression',
                    'motion_blur', 'pixelate', 'shot_noise', 'snow', 'zoom_blur']

def get_args_parser():
    parser = argparse.ArgumentParser('Test Time Training', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, help='path to a config file')
    parser.add_argument('--type', default='open', choices=['normal', 'open', 'hf_open'], help='a kind of archtectures')
    parser.add_argument('--data_root', default='/home/ykojima/dataset/imagenetv2-c/', help='a path to ttt data')
    parser.add_argument('--reconst', choices=['pixel', 'feature'], help='a kind of reconstruction')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to a pth file')
    parser.add_argument('--output', type=str, default='./output.csv', help='path to a output csv file')
    return parser

def main(args, corruption, severity):
    # fixed parameters
    eps = 1e-8 
    device = 'cuda'
    num_workers = 4
    pin_memory = True

    config = load_config(args.cfg)
    OmegaConf.set_struct(config, True)    

    if args.wandb:
        run = wandb.init(project='mae_clip_ttt',
                         entity="ykojima",
                         config=OmegaConf.to_container(config, resolve=True))
        config = OmegaConf.create(dict(wandb.config))
    print(config)

    # [NOTE]: this table is for logging
    table = wandb.Table(columns=columns)

    if args.type == 'normal':
        factory = RILSMAECLIPFactory(config.model)
    elif args.type == 'open':
        factory = PretrainedOpenCLIPDecoderEncoderFineTuneFactory(config.model, mae=args.reconst)
    elif args.type == 'hf_open':
        factory = PretrainedHFOpenCLIPFactory(config.model, mae=args.reconst)
    else:
        raise TypeError

    model, tokenizer, transform = factory.create()
    model = model.to(device)
    status = torch.load(args.checkpoint, map_location="cuda")

    # [NOTE]: freze parameters not related to TTT
    for name, param in model.named_parameters():
        if ('decoder' in name):
            param.requires_grad = False
        if ('text_model.encoder' in name):
            param.requires_grad = False
        print(name, param.requires_grad)

    # [NOTE]: update only image encoder
    if config.ttt.optimizer == 'adam':
        optimizer = torch.optim.AdamW(model.image_encoder.parameters(),
                eps=eps, lr=config.ttt.lr, betas=(0.9, 0.95), weight_decay=config.ttt.weight_decay)
    elif config.ttt.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.image_encoder.parameters(), lr=config.ttt.lr, weight_decay=config.ttt.weight_decay) 
    else:
        raise TypeError

    data_root = pathlib.Path(args.data_root) / corruption / str(severity)
    dataset = torchvision.datasets.ImageFolder(root=data_root, transform=transform('valid'))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=config.ttt.batch_size, num_workers=num_workers, shuffle=False)

    model.load_state_dict(status['model'])
    text_embeddings = zeroshot_weights(model.clip, tokenizer, imagenet_config.imagenet_classes, imagenet_config.imagenet_templates, device)
    logit_scale = model.clip.logit_scale

    # [NOTE]: initial calculation
    initial_score, initial_loss = get_mean_score_and_loss(model, data_loader, text_embeddings, device) 
    evaluator = ZeroShotImageNetEvaluator(tokenizer, device, dataset)

    eval_stats = evaluator(model.clip, update=False)        
    top1 = eval_stats['eval']['imagenet']['top1']
    top5 = eval_stats['eval']['imagenet']['top5']

    if args.wandb:
        wandb.log({'epoch': -1, 'clip_score': initial_score, 'mae_loss': initial_loss, 'top1': top1, 'top5': top5})

    table.add_data(config.ttt.lr, config.ttt.weight_decay, config.ttt.batch_size, config.ttt.optimizer, severity, corruption, -1, initial_score, initial_loss, top1, top5)
    tttrainer = TestTimeTrainer(data_loader, optimizer, device)

    for epoch in range(config.ttt.epochs):
        # training
        tttrainer(model.mae)
        score, loss = get_mean_score_and_loss(model, data_loader, text_embeddings, device)
        eval_stats = evaluator(model.clip, update=False)        
        top1 = eval_stats['eval']['imagenet']['top1']
        top5 = eval_stats['eval']['imagenet']['top5']
        table.add_data(config.ttt.lr, config.ttt.weight_decay, config.ttt.batch_size, config.ttt.optimizer, severity, corruption, epoch, score, loss, top1, top5)

        if args.wandb:
            wandb.log({'epoch': epoch, 'clip_score': score, 'mae_loss': loss, 'top1': top1, 'top5': top5})

    print(table.get_dataframe())
    table.get_dataframe().to_csv(args.output, index=False)

    if args.wandb:
        wandb.log({'result': table})

def get_mean_score_and_loss(model, data_loader, text_embeddings, device):
    # [NOTE]: initial calculation
    logit_scale = model.clip.logit_scale
    mae_loss_meter = AvgMeter()
    clip_score_meter = AvgMeter()
    for images, targets in data_loader:
        images = images.to(device)
        targets = targets.to(device)
        with torch.no_grad():
            mae_loss, reconstruction, mask = model.mae(images)
            count = images.size(0)
            mae_loss_meter.update(mae_loss.item(), count)

            clip_score = get_score(model, text_embeddings, images, targets, logit_scale=logit_scale.exp())
            clip_score_meter.update(clip_score.item(), count)

    return clip_score_meter.avg, mae_loss_meter.avg
 
def get_score(model, zeroshot_weights, images, targets, logit_scale=100):
    with torch.no_grad():
        image_features = model.clip.image_encode(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        scores = (logit_scale * image_features @ zeroshot_weights).softmax(dim=-1)

    # [NOTE]: this is tricky. validaty is already comfirmed.
    targets_expand = torch.unsqueeze(targets, 0)
    n = scores.shape[0]
    results = scores[np.arange(n), targets_expand].squeeze(0)
    return torch.mean(results)
 

def zeroshot_weights(model, tokenizer, classnames, templates, device):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            # 80 patterns per class
            texts = [template.format(classname) for template in templates] #format with class
            max_length = 15
            tokens = tokenizer(texts, padding=True, truncation=True, max_length=max_length)
            batch = {key: values.to(device) for key, values in tokens.items()}
            class_embeddings = model.text_encode(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True) # the norm shape is torch.Size([80, 1])
            class_embedding = class_embeddings.mean(dim=0) # the mean shape is torch.Size([256])
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    for corruption in corruptions_name:
        main(args, corruption, 5)

