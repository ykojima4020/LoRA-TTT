import argparse
import torch
import sys
from tqdm import tqdm
import pandas as pd
import numpy as np
import wandb
import pathlib

sys.path.append('../')
from tta import TestTimeAdapter 
from factory import RILSMAECLIPFactory, PretrainedOpenCLIPDecoderEncoderFineTuneFactory, \
                    PretrainedHFOpenCLIPFactory
from misc.config import load_config
from misc.transforms import get_open_clip_vitb16_transforms, get_tta_transforms
from omegaconf import OmegaConf

columns = ['lr', 'weight_decay', 'batch_size', 'epochs', 'optimizer', 'severity', 'corruption',
           'top1', 'top5', 'diff_top1', 'diff_top5', 'nor_top1', 'nor_top5']

def get_args_parser():
    parser = argparse.ArgumentParser('Run TTA', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, help='path to a config file')
    parser.add_argument('--type', default='open', choices=['normal', 'open', 'hf_open'], help='a kind of archtectures')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to a pth file')
    parser.add_argument('--output', type=str, default='./output.csv', help='path to a output csv file')
    return parser

def main():
    args = get_args_parser()
    args = args.parse_args()

    # fixed parameters
    eps = 1e-8 
    device = 'cuda'

    config = load_config(args.cfg)
    OmegaConf.set_struct(config, True)    

    if args.wandb:
        run = wandb.init(project=config.wandb_project,
                         entity="ykojima",
                         config=OmegaConf.to_container(config, resolve=True))
        config = OmegaConf.create(dict(wandb.config))
    print(config)

    # [NOTE]: this table is for logging
    table = wandb.Table(columns=columns)

    if args.type == 'normal':
        factory = RILSMAECLIPFactory(config.model)
    elif args.type == 'open':
        factory = PretrainedOpenCLIPDecoderEncoderFineTuneFactory(config.model, mae=config.reconst)
    elif args.type == 'hf_open':
        factory = PretrainedHFOpenCLIPFactory(config.model, mae=config.reconst)
    else:
        raise TypeError

    tta_runner = TestTimeAdapter(single=config.ttt.single) 
    if config.ttt.augmentation == 'simple':
        tta_transform = get_open_clip_vitb16_transforms
    elif config.ttt.augmentation == 'basic':
        tta_transform = get_tta_transforms
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
            top1_before_ttt, top5_before_ttt, top1_after_ttt, top5_after_ttt = tta_runner(factory, status['model'], config.ttt, data_root, tta_transform)

            diff_top1 = top1_after_ttt - top1_before_ttt
            diff_top5 = top5_after_ttt - top5_before_ttt
            try: 
                nor_top1 = top1_after_ttt / top1_before_ttt
            except ZeroDivisionError:
                print('Error: Cannot divide by zero.')
                nor_top1 = np.nan
            try: 
                nor_top5 = top5_after_ttt / top5_before_ttt
            except ZeroDivisionError:
                print('Error: Cannot divide by zero.')
                nor_top5 = np.nan

            diff_top1s.append(diff_top1)
            diff_top5s.append(diff_top5)
            nor_top1s.append(nor_top1)
            nor_top5s.append(nor_top5)

            table.add_data(config.ttt.lr, config.ttt.weight_decay, config.ttt.batch_size, config.ttt.epochs, config.ttt.optimizer, severity, corruption,
                           top1_after_ttt, top5_after_ttt, diff_top1, diff_top5, nor_top1, nor_top5)
            print(table.get_dataframe())
            table.get_dataframe().to_csv(args.output, index=False)

    stats = {'diff_top1': np.nanmean(diff_top1s),
             'diff_top5': np.nanmean(diff_top5s),
             'nor_top1': np.nanmean(nor_top1s),
             'nor_top5': np.nanmean(nor_top5s)}
    print(stats)

    if args.wandb:
        wandb.log(stats)
        wandb.log({'result': table})

    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
