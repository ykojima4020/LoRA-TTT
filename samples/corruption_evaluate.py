import torch
import torchvision.transforms as transforms
from imagenetv2_pytorch import ImageNetV2Dataset

import numpy as np
from tqdm import tqdm
import pandas as pd
import argparse

import sys
sys.path.append('../external/robustness/ImageNet-C/imagenet_c')
from imagenet_c import corrupt
sys.path.append('../')

from transformers import DistilBertTokenizer
from factory import MAECLIPFactory
from evaluator import imagenet_config

from misc.config import load_config
from omegaconf import OmegaConf

from misc.transforms import Corruption, get_corruption_transform 

corruptions_name = ['brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog',
               'frost', 'gaussian_noise', 'glass_blur', 'impulse_noise', 'jpeg_compression',
               'motion_blur', 'pixelate', 'shot_noise', 'snow', 'zoom_blur']

def zeroshot_classifier(classnames, templates, tokenizer, model):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            # 80 patterns per class
            texts = [template.format(classname) for template in templates] #format with class
            max_length = 15
            texts = tokenizer(texts, padding=True, truncation=True, max_length=max_length)
            batch = {key: torch.tensor(values).to("cuda") for key, values in texts.items()}
            class_embeddings = model.text_encode(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True) # the norm shape is torch.Size([80, 1])
            class_embedding = class_embeddings.mean(dim=0) # the mean shape is torch.Size([256])
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights

def zeroshot_classifier_open_clip(classnames, templates, tokenizer, model):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            # 80 patterns per class
            texts = [template.format(classname) for template in templates] #format with class
            texts = tokenizer(texts).cuda()
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True) # the norm shape is torch.Size([80, 1])
            class_embedding = class_embeddings.mean(dim=0) # the mean shape is torch.Size([256])
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

def evaluate(model, loader, zeroshot_weights):
    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for i, (images, target) in enumerate(tqdm(loader)):
            images = images.cuda()
            target = target.cuda()
        
            # predict
            image_features = model.image_encode(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = image_features @ zeroshot_weights
            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)
    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100
    return top1, top5

def evaluate_open_clip(model, loader, zeroshot_weights):
    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for i, (images, target) in enumerate(tqdm(loader)):
            images = images.cuda()
            target = target.cuda()
            
            # predict
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = image_features @ zeroshot_weights
            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)
    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100
    return top1, top5

   
def get_args_parser():
    parser = argparse.ArgumentParser('Evaluation on ImageNet-C', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, help='path to a config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to a pth file')
    parser.add_argument('--output', type=str, default='./output.csv', help='path to a output csv file')
    parser.add_argument('--open_clip', action='store_true', help='Use a model pretrained by OpenCLIP')
    return parser

def main():
    args = get_args_parser()
    args = args.parse_args()

    cfg = load_config(args.cfg)
    OmegaConf.set_struct(cfg, True)    

    if args.open_clip:
        import open_clip
        tokenizer = open_clip.get_tokenizer('ViT-B-16')
        model, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='datacomp_l_s1b_b8k')
        model.to('cuda')
        model.eval()

    else:
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    
        factory = MAECLIPFactory(cfg.model)
        model = factory.create().to("cuda")
        status = torch.load(args.checkpoint, map_location="cuda")
    
        model.load_state_dict(status['model'])
        model = model.clip
        model.eval()
 
    imagenet_classes = imagenet_config.imagenet_classes
    imagenet_templates = imagenet_config.imagenet_templates

    if args.open_clip:
        zeroshot_weights = zeroshot_classifier_open_clip(imagenet_classes, imagenet_templates, tokenizer, model) # the size is ([256, 1000])
    else:
        zeroshot_weights = zeroshot_classifier(imagenet_classes, imagenet_templates, tokenizer, model) # the size is ([256, 1000])

    severities = []
    corruptions = []
    acc_1 = []
    acc_5 = []

    for severity in range(1,6):
        for corruption in corruptions_name:
            transform = get_corruption_transform(Corruption(severity=severity, corruption_name=corruption))
            dataset = ImageNetV2Dataset(transform=transform)
            loader = torch.utils.data.DataLoader(dataset, batch_size=32, pin_memory=True, num_workers=2)
            if args.open_clip:
                top1, top5 = evaluate_open_clip(model, loader, zeroshot_weights)
            else:
                top1, top5 = evaluate(model, loader, zeroshot_weights)
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

main()

