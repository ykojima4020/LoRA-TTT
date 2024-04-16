# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: venv
#     language: python
#     name: venv
# ---

# %%
import torch
import torchvision

import sys
sys.path.append('../')

from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

from factory import RILSMAECLIPFactory, PretrainedOpenCLIPFactory, PretrainedOpenCLIPDecoderFineTuneFactory
from factory import PretrainedHFOpenCLIPFactory
from evaluator.evaluator import ZeroShotImageNetEvaluator

from imagenetv2_pytorch import ImageNetV2Dataset

# %%
import importlib
import factory
importlib.reload(factory)

# %%
corruptions_name = ['brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog',
                    'frost', 'gaussian_noise', 'glass_blur', 'impulse_noise', 'jpeg_compression',
                    'motion_blur', 'pixelate', 'shot_noise', 'snow', 'zoom_blur']

# %%
from misc.config import load_config
from omegaconf import OmegaConf

config = '../config/default_local.yaml'
cfg = load_config(config)
OmegaConf.set_struct(cfg, True)

device = 'cuda'


# %%
def robustness(model, tokenizer, transform, device, severity=5):

    corruptions = []
    top1s = []
    top5s = []

    dataset = torchvision.datasets.ImageFolder(root=f'/home/ykojima/dataset/imagenetv2-c/original/0', transform=transform('valid'))
    evaluator = ZeroShotImageNetEvaluator(tokenizer, device, dataset)
    eval_res = evaluator(model.clip, update=False)
    corruptions.append('original')
    top1s.append(eval_res['eval']['imagenet']['top1'])
    top5s.append(eval_res['eval']['imagenet']['top5'])

    for x in corruptions_name:
        print(x)
        if x == 'frost':
            continue
        dataset = torchvision.datasets.ImageFolder(root=f'/home/ykojima/dataset/imagenetv2-c/{x}/{severity}', transform=transform('valid'))
        evaluator = ZeroShotImageNetEvaluator(tokenizer, device, dataset)
        eval_res = evaluator(model.clip, update=False)
        corruptions.append(x)
        top1s.append(eval_res['eval']['imagenet']['top1'])
        top5s.append(eval_res['eval']['imagenet']['top5'])
        print(eval_res)

    return pd.DataFrame({'corruption': corruptions, 'top1': top1s, 'top5': top5s})


# %%
# df.to_csv('./feature_reconst_zeroshot_robustness.csv')
df = pd.read_csv('./robustness/feature_reconst_zeroshot_robustness.csv', index_col=0)
df

# %%
# open_clip_df.to_csv('./open_clip_zeroshot_robustness.csv')
open_clip_df = pd.read_csv('./robustness/open_clip_zeroshot_robustness.csv', index_col=0)
open_clip_df

# %%
severity = 5
index = list(range(15))
width = 0.3
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar([x-0.3 for x in index], df['top5'], width=width, label='Feature Reconstruction trained CC3M\n(finetune Decoder and 4 layers of Encoder)')
# ax.bar([x for x in index], pixel_reconst_df['top5'], width=width, label='Pixel Reconstruction trained CC3MS\n(finetune Decoder and 4 layers of Encoder)')
ax.bar([x+0.3 for x in index], open_clip_df['top5'], width=width, color='orange', label='Open CLIP')
ax.set_ylabel('Top5 Accuracy')
ax.set_xticks(index)
ax.set_xticklabels(df['corruption'].tolist(), rotation=90)
plt.legend()
plt.title(f'severity={severity}')

# %% [markdown]
# ### LoRA Robustness

# %%
from peft import LoraConfig, get_peft_model

# %%
# Hugging face open-clip robustness without fine-tuning

factory = PretrainedHFOpenCLIPFactory(cfg.model)
model, tokenizer, transform = factory.create()
model = model.to(device)

model.eval()

hf_open_clip_zeroshot_robustness_results = robustness(model, tokenizer, transform, device, severity=5)

# %%
# hf_open_clip_zeroshot_robustness_results.to_csv('./robustness/hf_open_clip_zeroshot_robustness.csv')
lora_robustness_results = pd.read_csv('./robustness/hf_open_clip_lora_zeroshot_robustness.csv', index_col=0)

# %%
lora_robustness_results

# %%
severity = 5
index = list(range(15))
width = 0.3
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar([x-0.3 for x in index], open_clip_df['top5'], width=width, color='orange', label='Original Open CLIP (datacomp_l_s1b-b8k)')
ax.bar([x for x in index], df['top5'], width=width, label='Feature Reconstruction trained CC3M\n(finetune Decoder and 4 layers of Encoder)')
ax.bar([x+0.3 for x in index], lora_robustness_results['top5'], width=width, label='Pixel Reconstruction CC3M \n(LoRA: r=2, w=kvqo)')
# ax.bar([x for x in index], pixel_reconst_df['top5'], width=width, label='Pixel Reconstruction trained CC3MS\n(finetune Decoder and 4 layers of Encoder)')
ax.set_ylabel('Top5 Accuracy')
ax.set_xticks(index)
ax.set_xticklabels(lora_robustness_results['corruption'].tolist(), rotation=90)
plt.legend()
plt.title(f'severity={severity}')

# %%
severity = 5
index = list(range(15))
width = 0.4
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar([x-0.2 for x in index], open_clip_df['top5'], width=width, color='orange', label='Original Open CLIP (datacomp_l_s1b-b8k)')
ax.bar([x+0.2 for x in index], lora_robustness_results['top5'], width=width, label='Pixel Reconstruction CC3M \n(LoRA: r=2, w=kvqo)')
ax.set_ylabel('Top5 Accuracy')
ax.set_xticks(index)
ax.set_xticklabels(lora_robustness_results['corruption'].tolist(), rotation=90)
plt.legend()
plt.title(f'severity={severity}')

# %%
#### Feature Reconstruction

# %%
model_path = '/home/ykojima/Desktop/clip/mae_clip/output/20240402_decoder_layer8_encoder_lora_finetune_feature_recon_hf_open_clip_cc3m_wd_0001_blr_1e5/checkpoint.pth'

factory = PretrainedHFOpenCLIPFactory(cfg.model, mae='feature')
model, tokenizer, transform = factory.create()

model = model.to(device)

status = torch.load(model_path, map_location="cuda")
model.load_state_dict(status['model'])

model.eval()

# %%
# parameters fixed
for name, param in model.named_parameters():
    param.requires_grad = False
    print(name, param.requires_grad)

# %%
df = robustness(model, tokenizer, transform, device, severity=5)

# %%
severity = 5
index = list(range(15))
width = 0.4
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar([x-0.2 for x in index], open_clip_df['top5'], width=width, color='orange', label='Original Open CLIP (datacomp_l_s1b-b8k)')
ax.bar([x+0.2 for x in index], lora_feature_reconst_robustness_results['top5'], width=width, label='Feature Reconstruction CC3M \n(LoRA: r=2, w=kvqo)')
ax.set_ylabel('Top5 Accuracy')
ax.set_xticks(index)
ax.set_xticklabels(lora_feature_reconst_robustness_results['corruption'].tolist(), rotation=90)
plt.legend()
plt.title(f'severity={severity}')

# %%
severity = 5
index = list(range(15))
width = 0.3
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar([x-0.3 for x in index], open_clip_df['top5'], width=width, color='orange', label='Original Open CLIP (datacomp_l_s1b-b8k)')
ax.bar([x for x in index], lora_robustness_results['top5'], width=width, label='Pixel Reconstruction CC3M \n(LoRA: r=2, w=kvqo)')
ax.bar([x+0.3 for x in index], lora_feature_reconst_robustness_results['top5'], width=width, label='Feature Reconstruction CC3M \n(LoRA: r=2, w=kvqo)')
ax.set_ylabel('Top5 Accuracy')
ax.set_xticks(index)
ax.set_xticklabels(lora_feature_reconst_robustness_results['corruption'].tolist(), rotation=90)
plt.legend()
plt.title(f'severity={severity}')

# %%
