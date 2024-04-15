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
from factory import PretrainedHFOpenCLIPDecoderEncoderFineTuneFactory
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

config = '../config/ttt_mae.yaml'
cfg = load_config(config)
OmegaConf.set_struct(cfg, True)

device = 'cuda'

# %%
# df.to_csv('./feature_reconst_zeroshot_robustness.csv')
df = pd.read_csv('./feature_reconst_zeroshot_robustness.csv', index_col=0)
df

# %%
# open_clip_df.to_csv('./open_clip_zeroshot_robustness.csv')
open_clip_df = pd.read_csv('./open_clip_zeroshot_robustness.csv', index_col=0)
open_clip_df

# %%

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
model_path = '/home/ykojima/Desktop/clip/mae_clip/output/20240401_decoder_layer8_encoder_lora_finetune_pix_recon_hf_open_clip_cc3m_wd_0001_blr_1e5/checkpoint.pth'

factory = PretrainedHFOpenCLIPDecoderEncoderFineTuneFactory(cfg.model, mae='pixel')
model, tokenizer, transform = factory.create()

# LoRA settings
config = LoraConfig(r=2,
                    target_modules=["k_proj", "v_proj", "q_proj", "out_proj"],
                    lora_dropout=0.01,
                    bias="none"
                    )
model = get_peft_model(model, config)
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
dataset = ImageNetV2Dataset(transform=transform('valid'))

evaluator = ZeroShotImageNetEvaluator(tokenizer, device, dataset)
eval_res = evaluator(model.clip, update=False)

# %%
severity = 5

corruptions = []
top1s = []
top5s = []

dataset = torchvision.datasets.ImageFolder(root=f'/home/ykojima/dataset/imagenetv2-c/original', transform=transform('valid'))
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

lora_robustness_results = pd.DataFrame({'corruption': corruptions, 'top1': top1s, 'top5': top5s})

# %%
lora_robustness_results.to_csv('./hf_open_clip_lora_zeroshot_robustness.csv')  

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
