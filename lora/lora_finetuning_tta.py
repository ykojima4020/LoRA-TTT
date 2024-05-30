import os
import sys
from tqdm import tqdm
import pathlib
import argparse
import wandb

import torch
import torch.distributed as dist
import torchvision
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from imagenetv2_pytorch import ImageNetV2Dataset
import numpy as np

from omegaconf import OmegaConf, read_write

sys.path.append('../')
from factory import PretrainedHFOpenCLIPFactory
from data.dataloader_builder import CLIPDataLoaderBuilder, GCC3MDataLoaderBuilder
from trainer.trainer import SimpleTrainer
from trainer.validater import SimpleValidater
from evaluator.evaluator import ZeroShotImageNetEvaluator
from tta import TestTimeAdapter

from misc.transforms import get_open_clip_vitb16_transforms, get_tta_transforms, get_tta_transforms_color
from misc.config import get_config, load_config
from misc.lr_scheduler import build_scheduler
from misc.logger import get_logger
from misc.optimizer import build_optimizer

import random

def get_args_parser():
    parser = argparse.ArgumentParser('Tuning hyper parameters used in LoRA for TTT', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, help='path to a config file')
    parser.add_argument('--reconst', choices=['pixel', 'feature'], help='a kind of reconstruction')
    parser.add_argument('--opts', help="Modify config options by adding 'KEY=VALUE' list. ", default=None, nargs='+')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    return parser

def process(rank, world_size, cfg):

    setup(rank, world_size)

    logger = get_logger()
    logger.info(f"Running train on rank {rank}.")

    if not cfg.train.lr:
        cfg.train.lr = cfg.train.base_lr * cfg.data.batch_size * world_size / 256 # 1e-3 * 64 / 256 = 0.00025

    if cfg.wandb and dist.get_rank() == 0:
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
    # [NOTE]: waiting wandb init
    dist.barrier()

    if dist.get_rank() == 0:
        logger.info(cfg)

    # [NOTE]: factory used in this script is only for Hugging Face.
    factory = PretrainedHFOpenCLIPFactory(cfg.model, mae=cfg.reconst)
    model, tokenizer, transform = factory.create()
    model = model.to(rank)

    for name, param in model.named_parameters():
        if ('decoder' in name):
            param.requires_grad = True

    logger.info('finetuning parameters')
    if dist.get_rank() == 0:
        for name, param in model.named_parameters():
            logger.info(f'{name}: {param.requires_grad}')

    ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    model_without_ddp = ddp_model.module

    dataloader_builder = CLIPDataLoaderBuilder(cfg.data, tokenizer, transform)
    gcc3m_dataloader_builder = GCC3MDataLoaderBuilder(cfg.data, tokenizer, transform)

    train_loader, train_sampler = gcc3m_dataloader_builder('train', rank, world_size, test=cfg.test)
    val_loader, _ = dataloader_builder(cfg.data.dataset.val_image_path,
                                      cfg.data.dataset.val_json, 'val', rank, world_size, test=cfg.test)

    optimizer = build_optimizer(cfg.train, model)
    lr_scheduler = build_scheduler(cfg.train, optimizer, len(train_loader))

    trainer = SimpleTrainer(train_loader, optimizer, lr_scheduler, cfg.train.clip_grad, rank)
    validater = SimpleValidater(val_loader, optimizer, rank)
    if dist.get_rank() == 0:
        dataset = ImageNetV2Dataset(transform=transform('valid')) 
        evaluator = ZeroShotImageNetEvaluator(tokenizer, rank, dataset)

    best_loss = float('inf')
    best_ttt_enhancement = float('-inf')
    best_acc_5 = 0
    best_acc_1 = 0

    logger.info('Start training')
    for epoch in range(cfg.train.start_epoch, cfg.train.epochs):
        dist.barrier()
        stats = {'epoch': epoch}
        ttt_table = None
        logger.info(f"Epoch: {epoch + 1}")
        ddp_model.train()
        train_stats = trainer(ddp_model, epoch)
        stats = stats | train_stats

        ddp_model.eval()
        with torch.no_grad():
            valid_stats, image_table = validater(ddp_model)
            stats = stats | valid_stats

        # [NOTE]: evaluation, ttt, and saving
        if dist.get_rank() == 0:

            eval_stats = evaluator(ddp_model.module.clip)
            stats = stats | eval_stats

            if stats['valid']['mae_loss'] < best_loss:
                logger.info("Best Model!")
                best_loss = stats['valid']['mae_loss']
                checkpoint = os.path.join(cfg.output, 'checkpoint.pth')
                metrics = {'val_loss': stats['valid']['mae_loss'],
                           'acc_1': stats['eval']['imagenet']['top1'],
                           'acc_5': stats['eval']['imagenet']['top5']}
                save_state = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'metrics': metrics,
                    'epoch': epoch,
                    'config': cfg
                 }
                torch.save(save_state, checkpoint)

            if ((epoch+1) % cfg.ttt.run_freq == 0):
                # [NOTE]: load best model
                # [TODO]: if there's no best model
                if checkpoint:
                    status = torch.load(checkpoint, map_location="cuda")
                else:
                    raise Exception('no checkpoint')

                ds = cfg.data.dataset['ttt'][0]
                ds_meta = cfg.data.dataset.meta[ds]
                ttt_stats, ttt_table = run_ttt(factory, status['model'], cfg.ttt, ds_meta)
                stats = stats | ttt_stats
                if stats['ttt']['diff_top1'] > best_ttt_enhancement:
                    best_ttt_enhancement = stats['ttt']['diff_top1']
                stats['best_ttt_enhancement'] = best_ttt_enhancement

        if cfg.wandb and dist.get_rank() == 0:
            logger.info(stats)
            wandb.log(stats)
            wandb.log({'image': image_table})
        dist.barrier()
        logger.info(stats)

    cleanup()

def run_ttt(factory, status, config, dataset):

    diff_top1s = []
    diff_top5s = []
    nor_top1s = []
    nor_top5s = []

    table = wandb.Table(columns=['corruption', 'severity', 'top1_before_ttt', 'top1_after_ttt', 'top5_before_ttt', 'top5_after_ttt'])

    tta_runner = TestTimeAdapter(single=config.single)
    if config.augmentation == 'simple':
        tta_transform = get_open_clip_vitb16_transforms
    elif config.augmentation == 'basic':
        tta_transform = get_tta_transforms
    elif config.augmentation == 'color':
        tta_transform = get_tta_transforms_color
    else:
        raise TypeError

    sev_stats = {}
    for severity in dataset['severities']:
        corr_stats = {}
        for corruption in dataset['corruptions']:
            # [NOTE]: there's no corruption dataset named frost
            if corruption == 'frost':
                continue
            data_root = pathlib.Path(dataset['path']) / corruption / str(severity)
            top1_before_ttt, top5_before_ttt, top1_after_ttt, top5_after_ttt = tta_runner(factory, status, config, data_root, tta_transform)

            diff_top1 = top1_after_ttt - top1_before_ttt
            diff_top5 = top5_after_ttt - top5_before_ttt
            nor_top1 = top1_after_ttt / top1_before_ttt
            nor_top5 = top5_after_ttt / top5_before_ttt

            diff_top1s.append(diff_top1)
            diff_top5s.append(diff_top5)
            nor_top1s.append(nor_top1)
            nor_top5s.append(nor_top5)

            table.add_data(corruption, severity, top1_before_ttt, top1_after_ttt, top5_before_ttt, top5_after_ttt)
            corr_stats.update({corruption: {'top1_before_ttt': top1_before_ttt,
                                            'top1_after_ttt': top1_after_ttt,
                                            'top5_before_ttt': top5_before_ttt,
                                            'top5_after_ttt': top5_after_ttt}})
        sev_stats.update({severity: corr_stats})

    stats = {'ttt': {'diff_top1': float(np.mean(diff_top1s)),
                     'diff_top5': float(np.mean(diff_top5s)),
                     'nor_top1': float(np.mean(nor_top1s)),
                     'nor_top5': float(np.mean(nor_top5s)),
                     'all': sev_stats}}
    return stats, table

def get_random_port():
    return random.randint(1024, 65535)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(get_random_port())

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main():
    args = get_args_parser()
    args = args.parse_args()
    cfg = get_config(args)

    if cfg.output:
        pathlib.Path(cfg.output).mkdir(parents=True, exist_ok=True)

    mp.spawn(process,
        args=(cfg.world_size, cfg,),
        nprocs=cfg.world_size,
        join=True)

if __name__ == "__main__":
    main()
