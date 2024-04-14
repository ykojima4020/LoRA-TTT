import os
import sys
from tqdm import tqdm
import pathlib
import argparse

import torch
import torch.distributed as dist
import torchvision
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import open_clip
from imagenetv2_pytorch import ImageNetV2Dataset
import numpy as np

from omegaconf import OmegaConf, read_write

sys.path.append('../')
from factory import PretrainedHFOpenCLIPFactory
from data.dataloader_builder import CLIPDataLoaderBuilder, GCC3MDataLoaderBuilder
from trainer.trainer import SimpleTrainer
from trainer.validater import SimpleValidater
from evaluator.evaluator import ZeroShotImageNetEvaluator
from ttt import run_ttt_enhancement 

from misc.config import get_config
from misc.lr_scheduler import build_scheduler
from misc.checkpoint import auto_resume_helper, load_checkpoint, save_checkpoint
from misc.logger import get_logger
from misc.optimizer import build_optimizer

def get_args_parser():
    parser = argparse.ArgumentParser('Tuning hyper parameters used in LoRA for TTT', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, help='path to a config file')
    parser.add_argument('--reconst', default='feature', choices=['pixel', 'feature'], help='a kind of reconstruction')
    parser.add_argument('--opts', help="Modify config options by adding 'KEY=VALUE' list. ", default=None, nargs='+')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--wandb', action='store_true')

    parser.add_argument('--batch_size', type=int, help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--device', type=str)

    parser.add_argument('--output', default='./tmp',
                        help='path where to save, empty for no saving')
    return parser

def main(rank, world_size, cfg):

    setup(rank, world_size)

    logger = get_logger()
    logger.info(f"Running train on rank {rank}.")

    if dist.get_rank() == 0:
        print(cfg)

    if not cfg.train.lr:
        cfg.train.lr = cfg.train.base_lr * cfg.data.batch_size * world_size / 256 # 1e-3 * 64 / 256 = 0.00025

    if cfg.wandb and dist.get_rank() == 0:
        import wandb
        run = wandb.init(
            project='mae_clip_lora_finetuning', entity="ykojima",
            dir=cfg.output,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume=cfg.checkpoint.auto_resume)
    else:
        wandb = None 
    # [NOTE]: waiting wandb init
    dist.barrier()

    # [NOTE]: factory used in this script is only for Hugging Face.
    factory = PretrainedHFOpenCLIPFactory(cfg.model, mae=cfg.reconst)
    model, tokenizer, transform = factory.create()
    model = model.to(rank)

    for name, param in model.named_parameters():
        if ('decoder' in name):
            param.requires_grad = True

    print('finetuning parameters')
    if dist.get_rank() == 0:
        for name, param in model.named_parameters():
            print(name, param.requires_grad)

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

    if cfg.checkpoint.auto_resume:
        # [NOTE]: Retrieve the most recent .pth file from the designated directory upon auto_resume.
        #         If the file is founded, cfg.checkpoint.resume is overwritten. 
        resume_file = auto_resume_helper(cfg.output)
        if resume_file:
            if cfg.checkpoint.resume:
                logger.warning(f'auto-resume changing resume file from {cfg.checkpoint.resume} to {resume_file}')
            with read_write(cfg):
                cfg.checkpoint.resume = resume_file
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {cfg.output}, ignoring auto resume')

    best_loss = float('inf')
    best_acc_5 = 0
    best_acc_1 = 0

    if cfg.checkpoint.resume:
        max_metrics = load_checkpoint(cfg, model_without_ddp, optimizer, lr_scheduler)
        print(max_metrics)
        best_loss = max_metrics['val_loss']
        best_acc_1 = max_metrics['acc_1']
        best_acc_5 = max_metrics['acc_5']
    print(best_loss, best_acc_5, best_acc_1)

    logger.info('Start training')
    for epoch in range(cfg.train.start_epoch, cfg.train.epochs):
        dist.barrier()
        # [TODO]: when using webdataset, sampler can not be used. How should I do?
        # train_sampler.set_epoch(epoch)
        stats = {'epoch': epoch}
        logger.info(f"Epoch: {epoch + 1}")
        ddp_model.train()
        train_stats = trainer(ddp_model, epoch)
        stats = stats | train_stats

        ddp_model.eval()
        with torch.no_grad():
            valid_stats, table = validater(ddp_model)
            stats = stats | valid_stats

        # [NOTE]: evaluation, ttt, and saving
        if dist.get_rank() == 0:
            status = {'model': model_without_ddp.state_dict()}

            if (epoch % cfg.ttt.run_freq == 0):
                ttt_stats = run_ttt(factory, status, cfg.ttt)
                stats = stats | ttt_stats

            eval_stats = evaluator(ddp_model.module.clip)
            stats = stats | eval_stats
            # [NOTE]: in this case, Zero-Shot top1 accuracy is used as the metric for saving the models.
            # [TODO]: This metric should be flexible.
            if stats['valid']['mae_loss'] < best_loss:
                # best_acc_1 = stats['eval']['imagenet']['top1']
                save_checkpoint(cfg, epoch, model_without_ddp, {
                    'val_loss': stats['valid']['mae_loss'],
                    'acc_1': stats['eval']['imagenet']['top1'],
                    'acc_5': stats['eval']['imagenet']['top5']
                }, optimizer, lr_scheduler)
                logger.info("Saved Best Model!")

        if cfg.wandb and dist.get_rank() == 0:
            wandb.log(stats)
            wandb.log({'image': table})
        dist.barrier()
        print(stats)

    cleanup()

def run_ttt(factory, status, config):

    diff_top1s = []
    diff_top5s = []
    nor_top1s = []
    nor_top5s = []

    corruptions_name = ['brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog',
                        'frost', 'gaussian_noise', 'glass_blur', 'impulse_noise', 'jpeg_compression',
                        'motion_blur', 'pixelate', 'shot_noise', 'snow', 'zoom_blur']

    # [NOTE]: choose kind of important corruptions
    corruptions_name = ['contrast', 'elastic_transform', 'gaussian_noise', 'motion_blur']

    severity = 5
    for corruption in corruptions_name:
        # [NOTE]: there's no corruption dataset named frost
        if corruption == 'frost':
            continue
        top1_before_ttt, top5_before_ttt, top1_after_ttt, top5_after_ttt = run_ttt_enhancement(factory, status, config, corruption, severity)

        diff_top1 = top1_after_ttt - top1_before_ttt
        diff_top5 = top5_after_ttt - top5_before_ttt

        nor_top1 = top1_after_ttt / top1_before_ttt
        nor_top5 = top5_after_ttt / top5_before_ttt
        
        diff_top1s.append(diff_top1)
        diff_top5s.append(diff_top5)
        nor_top1s.append(nor_top1)
        nor_top5s.append(nor_top5)

    stats = {'ttt': {'diff_top1': np.mean(diff_top1s),
                     'diff_top5': np.mean(diff_top5s),
                     'nor_top1': np.mean(nor_top1s),
                     'nor_top5': np.mean(nor_top5s)}}

    return stats 


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    cfg = get_config(args)
    if cfg.output:
        pathlib.Path(args.output).mkdir(parents=True, exist_ok=True)

    mp.spawn(main,
        args=(cfg.world_size, cfg,),
        nprocs=cfg.world_size,
        join=True)
