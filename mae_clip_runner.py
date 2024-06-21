import os
import sys
from tqdm import tqdm
import pathlib
import argparse
import wandb

import torch
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from torch.nn.parallel import DistributedDataParallel as DDP
from imagenetv2_pytorch import ImageNetV2Dataset
import numpy as np

from omegaconf import OmegaConf, read_write

sys.path.append('../')
from factory import PretrainedHFOpenCLIPFactory, PretrainedTPTHFOpenCLIPFactory
from data.dataloader_builder import CLIPDataLoaderBuilder, GCC3MDataLoaderBuilder
from trainer.trainer import SimpleTrainer
from trainer.validater import SimpleValidater
from evaluator.evaluator import ZeroShotImageNetEvaluator
from evaluator.imagenet_config import simple_prompts, ensemble_prompts, imagenet_classes
from evaluator.imagenet_variant_config import imagenet_a_classes, imagenet_r_classes
from tta import TPTMAETTARunner, TPTTTARunner, MEMLoRATTARunner, MEMLoRATPTTTARunner

from misc.transforms import get_open_clip_vitb16_transforms, get_tta_transforms, get_tta_transforms_color
from misc.tpt_transforms import AugMixAugmenter
from misc.config import get_config, load_config
from misc.lr_scheduler import build_scheduler
from misc.logger import get_logger
from misc.optimizer import build_optimizer

import random

logger = get_logger()

def get_args_parser():
    parser = argparse.ArgumentParser('Tuning hyper parameters used in LoRA for TTT', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, help='path to a config file')
    parser.add_argument('--reconst', choices=['pixel', 'feature'], help='a kind of reconstruction')
    parser.add_argument('--opts', help="Modify config options by adding 'KEY=VALUE' list. ", default=None, nargs='+')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--tta_only', action='store_true')
    parser.add_argument('--checkpoint', type=str, help='path to a pth file')
    parser.add_argument('--wandb', action='store_true')
    return parser

def main():
    args = get_args_parser()
    args = args.parse_args()
    cfg = get_config(args)

    device = 'cuda'

    if cfg.output:
        pathlib.Path(cfg.output).mkdir(parents=True, exist_ok=True)

    logger.info(f"Running train on device {device}.")

    if not cfg.train.lr:
        cfg.train.lr = cfg.train.base_lr * cfg.data.batch_size / 256 # 1e-3 * 64 / 256 = 0.00025

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
    model, tokenizer, transform = factory.create()
    model = model.to(device)

    tta_datasets = {}
    for ds in cfg.data.dataset['tta']:
        tta_datasets[ds] = cfg.data.dataset.meta[ds]

    # [NOTE]: extract valid TTA method
    tta_config = {}
    for m in cfg.tta['enable']:
        tta_config[m] = cfg.tta[m]
    cfg.tta = OmegaConf.create(tta_config)
 
    if cfg.tta_only:
        if args.checkpoint:
            print(args.checkpoint)
            status = torch.load(args.checkpoint, map_location=device)
        else:
            raise Exception('no checkpoint')
        stats = run_tta(factory, status['model'], tta_datasets, cfg.tta)
        if cfg.wandb:
            wandb.log(stats)
        logger.info(stats)
        return


    # [NOTE]: Preparation for fine-tuning
    dataloader_builder = CLIPDataLoaderBuilder(cfg.data, tokenizer, transform)
    gcc3m_dataloader_builder = GCC3MDataLoaderBuilder(cfg.data, tokenizer, transform)

    world_size = 1
    train_loader, train_sampler = gcc3m_dataloader_builder('train', device, world_size, test=cfg.test)
    val_loader, _ = dataloader_builder(cfg.data.dataset.val_image_path,
                                      cfg.data.dataset.val_json, 'val', device, world_size, test=cfg.test)

    optimizer = build_optimizer(cfg.train, model)
    lr_scheduler = build_scheduler(cfg.train, optimizer, len(train_loader))

    trainer = SimpleTrainer(train_loader, optimizer, lr_scheduler, cfg.train.clip_grad, device)
    validater = SimpleValidater(val_loader, optimizer, device)
    # [NOTE]: Metrics is ImageNetV2 here.
    dataset = ImageNetV2Dataset(transform=transform('valid')) 
    evaluator = ZeroShotImageNetEvaluator(tokenizer, dataset, ensemble_prompts, imagenet_classes, device)

    best_loss = float('inf')

    # [NOTE]: trainable parameters
    for name, param in model.named_parameters():
        if ('decoder' in name):
            param.requires_grad = True
        # [TODO]: LoRA parameters should be trainalbe here.

    logger.info('finetuning parameters')
    for name, param in model.named_parameters():
        logger.info(f'{name}: {param.requires_grad}')

    logger.info('Start fine-tuning')
    for epoch in range(cfg.train.start_epoch, cfg.train.epochs):
        stats = {'epoch': epoch}
        tta_table = None
        logger.info(f"Epoch: {epoch + 1}")
        model.train()
        train_stats = trainer(model, epoch)
        stats = stats | train_stats

        model.eval()
        with torch.no_grad():
            valid_stats, image_table = validater(model)
            stats = stats | valid_stats

        # [NOTE]: evaluation, tta, and saving
        eval_stats = evaluator(model.clip)
        stats = stats | eval_stats

        # [TODO]: this metrix should be specified by configuration.
        if stats['valid']['mae_loss'] < best_loss:
            logger.info("Best Model!")
            best_loss = stats['valid']['mae_loss']
            checkpoint = os.path.join(cfg.output, 'checkpoint.pth')
            metrics = {'val_loss': stats['valid']['mae_loss'],
                       'acc_1': stats['eval']['imagenet']['top1'],
                       'acc_5': stats['eval']['imagenet']['top5']}
            save_state = {
                'model': model.mae.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'metrics': metrics,
                'epoch': epoch,
                'config': cfg
             }
            torch.save(save_state, checkpoint)

        if ((epoch+1) % cfg.tta.run_freq == 0):
            # [NOTE]: load best model
            # [TODO]: if there's no best model
            if checkpoint:
                status = torch.load(checkpoint, map_location=device)
            else:
                raise Exception('no checkpoint')
            tta_stats, tta_table = run_tta(factory, status['model'], tta_datasets, cfg.tta)
            stats = stats | tta_stats

        if cfg.wandb:
            wandb.log(stats)
            wandb.log({'image': image_table})
        logger.info(stats)


def run_tta(factory, status, datasets, config):

    diff_top1s = []
    diff_top5s = []
    nor_top1s = []
    nor_top5s = []

    # [NOTE]: fixed parameters
    device = 'cuda'

    # [NOTE]: Data augmentation based on TPT
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])
    base_transform = transforms.Compose([
                transforms.Resize(224, interpolation=BICUBIC),
                transforms.CenterCrop(224)])
    preprocess = transforms.Compose([
                transforms.ToTensor(),
                normalize])
    batch_size = min([p.batch_size for n, p in config.items()])
    logger.info(f"TTA Batch size: {batch_size}")
    tta_transform = AugMixAugmenter(base_transform, preprocess, n_views=batch_size-1,
                                    augmix=False)

    datasets_stats = {}
    for name, dataset in datasets.items():
        data_root = pathlib.Path(dataset['path'])
        tta_data = torchvision.datasets.ImageFolder(root=data_root, transform=tta_transform)

        if dataset['prompt'] == 'simple':
            prompts = simple_prompts
        elif dataset['prompt'] == 'ensemble':
            prompts = ensemble_prompts
        else:
            raise TypeError

        if dataset['classes'] == 'imagenet':
            classes = imagenet_classes
        elif dataset['classes'] == 'imagenet_a':
            classes = imagenet_a_classes
        elif dataset['classes'] == 'imagenet_r':
            classes = imagenet_r_classes
        else:
            raise TypeError

        # [TODO]: Choose TTA algorithm here.
        if ('mae' in config.keys()) and ('tpt' in config.keys()):
            tta_runner = MEMLoRATPTTTARunner()
        elif not ('mae' in config.keys()) and ('tpt' in config.keys()):
            tta_runner = TPTTTARunner()
        elif ('mae' in config.keys()) and not ('tpt' in config.keys()):
            tta_runner = MEMLoRATTARunner()
        else:
            raise TypeError

        # [TODO]: first of all, calculate initial peformance before fine-tuning.
        model, tokenizer, transform = factory.create()
        model = model.to(device)
        tta_test_dataset = torchvision.datasets.ImageFolder(root=data_root, transform=transform('val'))
        evaluator = ZeroShotImageNetEvaluator(tokenizer, tta_test_dataset, prompts, classes, device)
        before_tta = evaluator(model.clip)
        top1_before_tta = before_tta['eval']['imagenet']['top1']
        top5_before_tta = before_tta['eval']['imagenet']['top5']
        del model

        top1_after_tta, top5_after_tta = tta_runner(factory, status, tta_data,
                                                    prompts, classes, config)

        diff_top1 = top1_after_tta - top1_before_tta
        diff_top5 = top5_after_tta - top5_before_tta

        try:
            nor_top1 = top1_after_tta / top1_before_tta
        except ZeroDivisionError:
            print('Error: Cannot divide by zero.')
            nor_top1 = np.nan
        try:
            nor_top5 = top5_after_tta / top5_before_tta
        except ZeroDivisionError:
            print('Error: Cannot divide by zero.')
            nor_top5 = np.nan

        diff_top1s.append(diff_top1)
        diff_top5s.append(diff_top5)
        nor_top1s.append(nor_top1)
        nor_top5s.append(nor_top5)

        datasets_stats.update({name: {'top1_before_tta': top1_before_tta,
                                      'top1_after_tta': top1_after_tta,
                                      'top5_before_tta': top5_before_tta,
                                      'top5_after_tta': top5_after_tta}})

    stats = {'tta': {'diff_top1': float(np.mean(diff_top1s)),
                     'diff_top5': float(np.mean(diff_top5s)),
                     'nor_top1': float(np.mean(nor_top1s)),
                     'nor_top5': float(np.mean(nor_top5s)),
                     'all': datasets_stats}}
    return stats


if __name__ == "__main__":
    main()
