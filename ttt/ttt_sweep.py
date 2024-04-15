import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import sys
from tqdm import tqdm
import pandas as pd
import numpy as np
import wandb
import pathlib

sys.path.append('../')
from ttt import TestTimeTrainer
from evaluator.evaluator import ZeroShotImageNetEvaluator
from factory import RILSMAECLIPFactory, PretrainedOpenCLIPFactory, PretrainedOpenCLIPDecoderEncoderFineTuneFactory, \
                    PretrainedHFOpenCLIPFactory
from misc.config import load_config
from omegaconf import OmegaConf

corruptions_name = ['brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog',
                    'frost', 'gaussian_noise', 'glass_blur', 'impulse_noise', 'jpeg_compression',
                    'motion_blur', 'pixelate', 'shot_noise', 'snow', 'zoom_blur']

columns = ['layer', 'lr', 'weight_decay', 'batch_size', 'epochs', 'optimizer', 'severity', 'corruption',
           'top1', 'top5', 'diff_top1', 'diff_top5', 'nor_top1', 'nor_top5']

def get_args_parser():
    parser = argparse.ArgumentParser('Evaluation on ImageNet-C', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, help='path to a config file')
    parser.add_argument('--type', default='open', choices=['normal', 'open', 'hf_open'], help='a kind of archtectures')
    parser.add_argument('--reconst', default='feature', choices=['pixel', 'feature'], help='a kind of reconstruction')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to a pth file')
    parser.add_argument('--output', type=str, default='./output.csv', help='path to a output csv file')
    return parser

def sweep():
    sweep_config = load_config('ttt_sweep.yaml')
    sweep_id = wandb.sweep(OmegaConf.to_container(sweep_config, resolve=True), project="mae_clip_ttt")
    wandb.agent(sweep_id, main)

def main():
    args = get_args_parser()
    args = args.parse_args()

    # fixed parameters
    eps = 1e-8 
    device = 'cuda'

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

    status = torch.load(args.checkpoint, map_location="cuda")

    diff_top1s = []
    diff_top5s = []
    nor_top1s = []
    nor_top5s = []

    ds = config.data.dataset['ttt'][0]
    ds_meta = config.data.dataset.meta[ds]
    for severity in ds_meta['severities']:
        for corruption in ds_meta['corruptions']:
            # [NOTE]: there's no corruption dataset named frost
            if corruption == 'frost':
                continue

            data_root = pathlib.Path(ds_meta['path']) / corruption / str(severity)
            top1_before_ttt, top5_before_ttt, top1_after_ttt, top5_after_ttt = run_ttt_enhancement(factory, status, config.ttt, data_root)

            diff_top1 = top1_after_ttt - top1_before_ttt
            diff_top5 = top5_after_ttt - top5_before_ttt
            nor_top1 = top1_after_ttt / top1_before_ttt
            nor_top5 = top5_after_ttt / top5_before_ttt

            diff_top1s.append(diff_top1)
            diff_top5s.append(diff_top5)
            nor_top1s.append(nor_top1)
            nor_top5s.append(nor_top5)

            table.add_data(config.ttt.layer, config.ttt.lr, config.ttt.weight_decay, config.ttt.batch_size, config.ttt.epochs, config.ttt.optimizer, severity, corruption,
                           top1_after_ttt, top5_after_ttt, diff_top1, diff_top5, nor_top1, nor_top5)
            print(table.get_dataframe())
            table.get_dataframe().to_csv(args.output, index=False)

    stats = {'diff_top1': np.mean(diff_top1s),
             'diff_top5': np.mean(diff_top5s),
             'nor_top1': np.mean(nor_top1s),
             'nor_top5': np.mean(nor_top5s)}
    print(stats)

    if args.wandb:
        wandb.log(stats)
        wandb.log({'result': table})

    torch.cuda.empty_cache()

def run_ttt_enhancement(factory, status, config, data_root,
                        num_workers=4, pin_memory=True, device='cuda'):

    # [TODO]: evaluator and trainer should be provided as arguments.
    model, tokenizer, transform = factory.create()
    model = model.to(device)

    # [NOTE]: freze parameters not related to TTTPretrainedHFOpenCLIPFactory
    for name, param in model.named_parameters():
        if ('decoder' in name):
            param.requires_grad = False
        if ('text_model.encoder' in name):
            param.requires_grad = False

    # [NOTE]: update only image encoder
    if config.optimizer == 'adam':
        optimizer = torch.optim.AdamW(model.image_encoder.parameters(),
                eps=eps, lr=config.lr, betas=(0.9, 0.95), weight_decay=config.weight_decay)
    elif config.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.image_encoder.parameters(), lr=config.lr, weight_decay=config.weight_decay) 
    else:
        raise TypeError

    # [NOTE]: initialization
    dataset = torchvision.datasets.ImageFolder(root=data_root, transform=transform('valid'))
    evaluator = ZeroShotImageNetEvaluator(tokenizer, device, dataset)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, num_workers=num_workers, pin_memory=pin_memory)
    tttrainer = TestTimeTrainer(train_loader, optimizer, device)

    # [NOTE]: STEP1: Evaluation of initial model before TTT.
    before_ttt = evaluator(model.clip)
    before_ttt_top1 = before_ttt['eval']['imagenet']['top1']
    before_ttt_top5 = before_ttt['eval']['imagenet']['top5']

    # [NOTE]: after culculation original zero-shot performance, load the finetuned weights.
    model.load_state_dict(status['model'])

    # [NOTE]: STEP2: TTT
    for epoch in range(0, config.epochs):
        tttrainer(model.mae)

    # [NOTE]: STEP3: Evaluation of model after TTT.
    after_ttt = evaluator(model.clip, update=False)
    after_ttt_top1 = after_ttt['eval']['imagenet']['top1']
    after_ttt_top5 = after_ttt['eval']['imagenet']['top5']

    return before_ttt_top1, before_ttt_top5, after_ttt_top1, after_ttt_top5

if __name__ == "__main__":
    main()
    # sweep()
