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
from tta import TTARunner, TTARunnerAnalyser, ParallelTTARunner, ImageEncoderTTA, TextPromptTTA, MTA
from tta import MAELoss, WeightedMAELoss, MAEConsistencyLoss, MEMLoss, MAEMEMLoss, MAEMEMLossV2

from misc.tpt_transforms import AugMixAugmenter
from misc.logger import get_logger

from external.TPT.data.fewshot_datasets import BaseJsonDataset, Aircraft
from external.TPT.data.cls_to_names import *

logger = get_logger()

def run_tta(factory, status, datasets, config, analyser=False):

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

    tta_runner = build_tta_runner(factory, status, config, analyser=analyser)

    datasets_stats = {}
    for name, dataset in datasets.items():

        # dataset preparation
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

        top1_before_tta, top5_before_tta =  evaluate_before_tta(factory, data_root, dataset, classes, prompts)

        if 'imagenet' in dataset['classes']:
            tta_data = torchvision.datasets.ImageFolder(root=data_root, transform=tta_transform)
        elif dataset['classes'] == 'aircraft':
            tta_data = Aircraft(data_root, 'test', None, tta_transform)
        else:
            tta_data = BaseJsonDataset(data_root, dataset['label'], 'test', None, tta_transform)

        top1_after_tta, top5_after_tta = tta_runner(tta_data, classes, prompts, dname=name)

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


def evaluate_before_tta(factory, data_root, dataset, classes, prompts, device='cuda'):

    # [TODO]: first of all, calculate initial peformance before TTA.
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
    return top1_before_tta, top5_before_tta



def build_tta_runner(factory, status, config, device='cuda', analyser=False):

    params = ['tp', 'peft', 'ie', 'tp_peft']
    n = count_elements_in_list(list(config.keys()), params)
    if n == 1:
        logger.info('Single TTA.')
        tta_runner = build_single_tta_runner(factory, status, config, analyser=analyser)
    elif n == 2:
        logger.info('Doble TTA.')
        tta_runner = build_double_tta_runner(factory, status, config)
    else:
        logger.info('multiple TTA is not implemented.')
        raise NotImplementedError
    logger.info(f'{type(tta_runner)} created.')
    return tta_runner

def build_single_tta_runner(factory, status, config, device='cuda', analyser=False):
    model, tokenizer, _ = factory.create()
    model = model.to(device)

    if 'tp' in config.keys():
        if not config['tp']['loss'] == ['mem']:
            logger.info('MEM is only used for TPT.')
            raise NotImplementedError
        # [NOTE]: Choose Loss for Text Prompt here.
        loss = MEMLoss(tpt=True, selection_p=config['tp']['selection_p'])
        if config['tp']['mta']:
            handler = MTA(model, tokenizer, status, loss, config['tp'])
        else:
            handler = TextPromptTTA(model, tokenizer, status, loss, config['tp'])

    elif 'peft' in config.keys():
        # [NOTE]: Choose Loss for PEFT here.
        loss = loss_selector(config['peft'])
        handler = ImageEncoderTTA(model, tokenizer, status, loss, config['peft'], lora=True)

    elif 'ie' in config.keys():
        # [NOTE]: Choose Loss for PEFT here.
        loss = loss_selector(config['ie'])
        handler = ImageEncoderTTA(model, tokenizer, status, loss, config['ie'], lora=False)
    elif 'tt_peft' in config.keys():
        raise NotImplementedError
    else:
        raise TypeError

    logger.info(f'{type(handler)} created.')
    if analyser:
        tta_runner = TTARunnerAnalyser(handler)
    else:
        tta_runner = TTARunner(handler)
    return tta_runner

def build_double_tta_runner(factory, status, config, device='cuda'):
    '''
    This fucntion is only used for combination of TPT and ImageEncoder Tuning.
    '''

    model, tokenizer, _ = factory.create()
    model = model.to(device)

    if 'tp' in config.keys():
        if not config['tp']['loss'] == ['mem']:
            logger.info('MEM is only used for TPT.')
            raise NotImplementedError
        # [NOTE]: Choose Loss for Text Prompt here.
        loss = MEMLoss(tpt=True, selection_p=config['tp']['selection_p'])
        tp_tta_handler = TextPromptTTA(model, tokenizer, status, loss, config['tp'])
    else:
        raise TypeError

    if 'peft' in config.keys():
        # [NOTE]: Choose Loss for PEFT here.
        loss = loss_selector(config['peft'])
        ie_tta_handler = ImageEncoderTTA(model, tokenizer, status, loss, config['peft'], lora=True)
    else:
        raise TypeError

    # [NOTE]: should be TPT first,
    tta_runner = ParallelTTARunner(tp_tta_handler, ie_tta_handler)
    return tta_runner



def loss_selector(config):
    loss = config['loss']
    loss = sorted(loss)
    if loss == ['mem']:
        # [NOTE]: MEM for updating LoRA
        loss = MEMLoss(selection_p=config['selection_p'])
    elif loss == ['mae']:
        # [NOTE]: MAE for updating LoRA
        loss = MAELoss(selection_p=config['selection_p'])
    elif loss == ['weighted_mae']:
        loss = WeightedMAELoss(selection_p=config['selection_p'])
    elif loss == ['mae_consis']:
        loss = MAEConsistencyLoss(selection_p=config['selection_p'])
    elif loss == ['mae', 'mem']:
        # [NOTE]: MAE + MEM for updating LoRA
        loss = MAEMEMLossV2(config['mae']['weight'],
                          config['mem']['weight'],
                          selection_p=config['selection_p'])
    else:
        raise NotImplementedError
    return loss


def count_elements_in_list(main_list, sublist):
    return sum(main_list.count(element) for element in sublist)

