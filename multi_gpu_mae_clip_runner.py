import os
import sys
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

from omegaconf import OmegaConf

sys.path.append('../')
from factory import PretrainedHFOpenCLIPFactory
from data.dataloader_builder import CLIPDataLoaderBuilder, GCC3MDataLoaderBuilder
from trainer.trainer import SimpleTrainer, Loss
from trainer.validater import SimpleValidater
from evaluator.evaluator import ZeroShotEvaluator
from evaluator.imagenet_config import simple_prompts, ensemble_prompts, imagenet_classes
from evaluator.imagenet_variant_config import imagenet_a_classes, imagenet_r_classes

from misc.config import get_config
from misc.lr_scheduler import build_scheduler
from misc.logger import get_logger
from misc.optimizer import build_optimizer
from tta_pipeline import run_tta

import random

from misc.seed_util import initialize_seed

use_fixed_seed = False
seed_value = 42

initialize_seed(use_fixed_seed, seed_value)

def get_args_parser():
    parser = argparse.ArgumentParser('Tuning hyper parameters used in LoRA for TTT', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, help='path to a config file')
    parser.add_argument('--opts', help="Modify config options by adding 'KEY=VALUE' list. ", default=None, nargs='+')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--checkpoint', type=str, help='path to a pth file')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--analyser', action='store_true')
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
    if 'tp' in cfg.tta['params']:
        factory = PretrainedHFOpenCLIPFactory(cfg.model, tpt=True)
        logger.info('TPT enable.')
    else:
        factory = PretrainedHFOpenCLIPFactory(cfg.model, tpt=cfg.fs.coop)
        logger.info('TPT disable.')

    model, tokenizer, transform = factory.create()
    model = model.to(rank)

    tta_datasets = {}
    for ds in cfg.data.dataset['tta']:
        tta_datasets[ds] = cfg.data.dataset.meta[ds]

    # [NOTE]: extract valid TTA method
    tta_config = {}
    for m in cfg.tta['params']:
        tta_config[m] = cfg.tta[m]
    tta_config['run_freq'] = cfg.tta.run_freq
    cfg.tta = OmegaConf.create(tta_config)

    if cfg.checkpoint:
        status = torch.load(cfg.checkpoint, map_location=cfg.device)
        model.mae.load_state_dict(status['model'])
        logger.info(f'{cfg.checkpoint} loaded.')
    else:
        status = {'model': model.mae.state_dict()}
        logger.info('initial weight')

    if not cfg.finetune:
        stats = run_tta(factory, status['model'], tta_datasets, cfg.tta, analyser=cfg.analyser)
        if cfg.wandb:
            wandb.log(stats)
        logger.info(stats)
        return


    run_validate = False
    run_evaluate = False
    # [NOTE]: Start MAE fine-tuning
    logger.info('Run fine-tuning.')
    for name, param in model.image_encoder.named_parameters():
        if 'lora' in name:
            param.requires_grad = True

    for name, param in model.mae.decoder.named_parameters():
        param.requires_grad = True

    logger.info('fine-tuning parameters.')
    if dist.get_rank() == 0:
        for name, param in model.named_parameters():
            logger.info(f'{name}: {param.requires_grad}')
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Trainable parameters: {trainable_params}')


    ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    model_without_ddp = ddp_model.module

    gcc3m_dataloader_builder = GCC3MDataLoaderBuilder(cfg.data, tokenizer, transform)
    dataloader_builder = CLIPDataLoaderBuilder(cfg.data, tokenizer, transform)

    train_loader, train_sampler = gcc3m_dataloader_builder('train', rank, world_size, test=cfg.test)

    val_loader, _ = dataloader_builder(cfg.data.dataset.val_image_path,
                                      cfg.data.dataset.val_json, 'val', rank, world_size, test=cfg.test)

    optimizer = build_optimizer(cfg.train, model)
    lr_scheduler = build_scheduler(cfg.train, optimizer, len(train_loader))

    trainer = SimpleTrainer(train_loader, optimizer, lr_scheduler, cfg.train.clip_grad, rank)
    loss = Loss(clip_weight=cfg.train.loss.clip_weight, mae_weight=cfg.train.loss.mae_weight)

    validater = SimpleValidater(val_loader, optimizer, rank)
    if dist.get_rank() == 0:
        # [NOTE]: Metrics is ImageNetV2 here.
        dataset = ImageNetV2Dataset(transform=transform('valid')) 
        evaluator = ZeroShotEvaluator(tokenizer, dataset, simple_prompts, imagenet_classes, rank)

    for epoch in range(cfg.train.start_epoch, cfg.train.epochs):
        dist.barrier()
        stats = {'epoch': epoch}
        tta_table = None
        logger.info(f"Epoch: {epoch + 1}")
        ddp_model.train()
        train_stats = trainer(ddp_model, loss, epoch)
        stats = stats | train_stats

        if run_validate:
            ddp_model.eval()
            with torch.no_grad():
                valid_stats, image_table = validater(ddp_model)
                stats = stats | valid_stats

        # [NOTE]: evaluation, tta, and saving
        if dist.get_rank() == 0:

            if run_evaluate:
                eval_stats = evaluator(ddp_model.module.clip)
                stats = stats | eval_stats

            logger.info("Save Model!")
            checkpoint = os.path.join(cfg.output, f'checkpoint_{epoch:03}.pth')
            save_state = {
                'model': model_without_ddp.mae.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'config': cfg
            }
            torch.save(save_state, checkpoint)

            if ((epoch+1) % cfg.tta.run_freq == 0):
                # [NOTE]: load best model
                # [TODO]: if there's no best model
                if checkpoint:
                    status = torch.load(checkpoint, map_location="cuda")
                else:
                    raise Exception('no checkpoint')

                tta_stats = run_tta(factory, status['model'], tta_datasets, cfg.tta)
                stats = stats | tta_stats

        if cfg.wandb and dist.get_rank() == 0:
            logger.info(stats)
            wandb.log(stats)
        dist.barrier()
        logger.info(stats)

    cleanup()

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
