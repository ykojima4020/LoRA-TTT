import os
import sys
from tqdm import tqdm
import pathlib
import argparse

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import open_clip
from imagenetv2_pytorch import ImageNetV2Dataset

from omegaconf import OmegaConf, read_write

from factory import RILSMAECLIPFactory, PretrainedOpenCLIPFactory, PretrainedOpenCLIPDecoderFineTuneFactory, PretrainedOpenCLIPDecoderEncoderFineTuneFactory
from factory import PretrainedHFOpenCLIPFactory
from data.dataloader_builder import CLIPDataLoaderBuilder, GCC3MDataLoaderBuilder
from trainer.trainer import SimpleTrainer
from trainer.validater import SimpleValidater
from evaluator.evaluator import ZeroShotImageNetEvaluator

from misc.config import get_config
from misc.lr_scheduler import build_scheduler
from misc.checkpoint import auto_resume_helper, load_checkpoint, save_checkpoint
from misc.logger import get_logger
from misc.optimizer import build_optimizer

def get_args_parser():
    parser = argparse.ArgumentParser('CLIP pre-training', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, help='path to a config file')
    parser.add_argument('--type', default='open', choices=['normal', 'open', 'hf_open'], help='a kind of archtectures')
    parser.add_argument('--reconst', default='feature', choices=['pixel', 'feature'], help='a kind of reconstruction')
    parser.add_argument('--opts', help="Modify config options by adding 'KEY=VALUE' list. ", default=None, nargs='+')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--wandb', action='store_true')

    parser.add_argument('--batch_size', type=int, help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', type=int)
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
            project='mae_clip', entity="ykojima",
            dir=cfg.output,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume=cfg.checkpoint.auto_resume)
    else:
        wandb = None 
    # [NOTE]: waiting wandb init
    dist.barrier()

    # [NOTE]: organize how factory and model are created.
    if cfg.type == 'normal':
        factory = RILSMAECLIPFactory(cfg.model)
    elif cfg.type == 'open':
        # factory = PretrainedOpenCLIPFactory(cfg.model)
        # factory = PretrainedOpenCLIPDecoderFineTuneFactory(cfg.model, mae=cfg.reconst)
        factory = PretrainedOpenCLIPDecoderEncoderFineTuneFactory(cfg.model, mae=cfg.reconst)
    elif args.type == 'hf_open':
        factory = PretrainedHFOpenCLIPFactory(cfg.model, mae=cfg.reconst, peft='lora')
    else:
        raise TypeError

    model, tokenizer, transform = factory.create()
    model = model.to(rank)

    for name, param in model.named_parameters():
        if ('decoder' in name):
            param.requires_grad = True

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

        # [NOTE]: evaluation and saving
        if dist.get_rank() == 0:
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

    cleanup()

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
