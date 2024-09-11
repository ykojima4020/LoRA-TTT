import sys
import pathlib

import torch
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

import numpy as np

sys.path.append('../')
from evaluator.evaluator import ZeroShotEvaluator
from evaluator.imagenet_config import simple_prompts, ensemble_prompts, imagenet_classes
from evaluator.imagenet_variant_config import imagenet_a_classes, imagenet_r_classes
from tta import TTARunner, MAELoss, MEMLoss, MAEMEMLoss

from misc.tpt_transforms import AugMixAugmenter
from misc.logger import get_logger

from external.TPT.data.fewshot_datasets import BaseJsonDataset, Aircraft
from external.TPT.data.cls_to_names import *

logger = get_logger()

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
    batch_size = min([p.batch_size for n, p in config.items() if hasattr(p, 'batch_size')])
    logger.info(f"TTA Batch size: {batch_size}")
    tta_transform = AugMixAugmenter(base_transform, preprocess, n_views=batch_size-1,
                                    augmix=False)

    datasets_stats = {}
    for name, dataset in datasets.items():
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


        if all(param in ['tp', 'peft', 'ie'] for param in config.keys()):
            raise NotImplementedError
        elif any(param in ['tp'] for param in config.keys()):
            loss = config['tp']['loss']
            if ('mem' in loss) and not ('mae' in loss):
                # [NOTE]: MEM for updating LoRA
                loss = MEMLoss(tpt=True)
            else:
                raise NotImplementedError
            # [NOTE]: Choose Loss for Text Prompt here.
            tta_runner = TTARunner(config['tp'], loss, tp=True)

        elif any(param in ['peft'] for param in config.keys()):
            # [NOTE]: Choose Loss for PEFT here.
            loss = config['peft']['loss']
            if ('mem' in loss) and not ('mae' in loss):
                # [NOTE]: MEM for updating LoRA
                loss = MEMLoss()
            elif not ('mem' in loss) and ('mae' in loss):
                # [NOTE]: MAE for updating LoRA
                loss = MAELoss()
            elif ('mem' in loss) and ('mae' in loss):
                # [NOTE]: MAE + MEM for updating LoRA
                loss = MAEMEMLoss(config['peft']['mae']['weight'],
                                  config['peft']['mem']['weight'])
            else:
                raise NotImplementedError
            tta_runner = TTARunner(config['peft'], loss, tp=False, lora=True)

        elif any(param in ['ie'] for param in config.keys()):
            # [NOTE]: Choose Loss for PEFT here.
            loss = config['ie']['loss']
            if ('mem' in loss) and not ('mae' in loss):
                # [NOTE]: MEM for updating LoRA
                loss = MEMLoss()
            elif not ('mem' in loss) and ('mae' in loss):
                # [NOTE]: MAE for updating LoRA
                loss = MAELoss()
            elif ('mem' in loss) and ('mae' in loss):
                # [NOTE]: MAE + MEM for updating LoRA
                loss = MAEMEMLoss(config['ie']['mae']['weight'],
                                  config['ie']['mem']['weight'])
            else:
                raise NotImplementedError
            tta_runner = TTARunner(config['ie'], loss, tp=False, lora=False)
        else:
            raise TypeError

        logger.info(f'{type(tta_runner)} created.')

        # [TODO]: first of all, calculate initial peformance before fine-tuning.
        model, tokenizer, transform = factory.create()
        model = model.to(device)

        if 'imagenet' in dataset['classes']:
            tta_test_dataset = torchvision.datasets.ImageFolder(root=data_root, transform=transform('val'))
        elif dataset['classes'] == 'aircraft':
            tta_test_dataset = Aircraft(data_root, 'test', None, transform('val'))
        else:
            tta_test_dataset = BaseJsonDataset(data_root, dataset['label'], 'test', None, transform('val'))

        evaluator = ZeroShotEvaluator(tokenizer, tta_test_dataset, prompts, classes, device)
        before_tta = evaluator(model.clip)
        top1_before_tta = before_tta['eval']['imagenet']['top1']
        top5_before_tta = before_tta['eval']['imagenet']['top5']
        del model

        if 'imagenet' in dataset['classes']:
            tta_data = torchvision.datasets.ImageFolder(root=data_root, transform=tta_transform)
        elif dataset['classes'] == 'aircraft':
            tta_data = Aircraft(data_root, 'test', None, tta_transform)
        else:
            tta_data = BaseJsonDataset(data_root, dataset['label'], 'test', None, tta_transform)

        top1_after_tta, top5_after_tta = tta_runner(factory, status, tta_data,
                                                    prompts, classes)

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

