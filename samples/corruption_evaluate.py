import torch
import torchvision.transforms as transforms
from imagenetv2_pytorch import ImageNetV2Dataset

import pandas as pd
import argparse

import sys
sys.path.append('../')
from factory import PretrainedHFOpenCLIPFactory
from misc.config import load_config
from omegaconf import OmegaConf

from misc.transforms import Corruption, get_corruption_transform 
from evaluator.evaluator import ZeroShotEvaluator
from evaluator.imagenet_config import simple_prompts, ensemble_prompts, imagenet_classes

corruptions_name = ['brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog',
               'frost', 'gaussian_noise', 'glass_blur', 'impulse_noise', 'jpeg_compression',
               'motion_blur', 'pixelate', 'shot_noise', 'snow', 'zoom_blur']

def get_args_parser():
    parser = argparse.ArgumentParser('Evaluation on ImageNet-C', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, help='path to a config file')
    return parser

def main():
    args = get_args_parser()
    args = args.parse_args()

    cfg = load_config(args.cfg)
    OmegaConf.set_struct(cfg, True)

    device = 'cuda'
    factory = PretrainedHFOpenCLIPFactory(cfg.model)
    model, tokenizer, transform = factory.create()
    model.to(device)
    model.eval()

    severities = []
    corruptions = []
    acc_1 = []
    acc_5 = []

    for severity in range(1,6):
        for corruption in corruptions_name:
            transform = get_corruption_transform(Corruption(severity=severity, corruption_name=corruption))
            dataset = ImageNetV2Dataset(transform=transform)
            evaluator = ZeroShotEvaluator(tokenizer, dataset, ensemble_prompts, imagenet_classes, device)
            result = evaluator(model.clip)
            top1 = result['eval']['imagenet']['top1']
            top5 = result['eval']['imagenet']['top5']

            print(severity, corruption, top1, top5)
            severities.append(severity)
            corruptions.append(corruption)
            acc_1.append(top1)
            acc_5.append(top5)

    df = pd.DataFrame({'severity': severities,
                       'corruption': corruptions,
                       'top1': acc_1,
                       'top5': acc_5})
    print(df) 
    df.to_csv(args.output, index=False)   

if __name__ == "__main__":
    main()
