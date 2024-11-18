# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: mae_clip
#     language: python
#     name: mae_clip
# ---

# +
import wandb
import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('../')
from wandb_adapter import runs_params_formatter, basic_formatter
# -

loss_mapping = {"['mae']": 'MAE',
                "['mem']": 'MEM'}

# +
# select sweep_id
sweep_ids = ['<entitiy>/<project>/<sweep id>']
sweep_ids = ['ykojima/mae_clip_mem_lora_without_finetune/zeireekk']

runs = []
for sid in sweep_ids:
    api = wandb.Api()
    sweep = api.sweep(sid)
    runs.extend(list(sweep.runs))
print(len(runs))
# -

metric = ['tta.all.imagenet_a.top1_after_tta',
          'tta.all.imagenetv2.top1_after_tta',
          'tta.all.imagenet.top1_after_tta',
          'tta.all.imagenet_r.top1_after_tta',
          'tta.all.imagenet_sketch.top1_after_tta']
runs_summary = runs_params_formatter(runs)
df = basic_formatter(runs_summary, metric)

# +
loss = df['tta.peft.loss'][:1][0]
loss_label = loss_mapping[str(loss)]
modules_label = 'kvqo'

categories = {'tta.all.imagenet_a.top1_after_tta': 'ImageNet-A',
              'tta.all.imagenet_r.top1_after_tta': 'ImageNet-R',
              'tta.all.imagenet_sketch.top1_after_tta': 'ImageNet-Sketch',
              'tta.all.imagenet.top1_after_tta': 'ImageNet-Val',
              'tta.all.imagenetv2.top1_after_tta': 'ImageNet-V2'}

fig, ax = plt.subplots()
for k,v in categories.items():
    tmp = df.dropna(subset=[k])
    ax.plot(tmp['model.peft.r'], tmp[k], 'o-', label=v)
ax.set_ylabel('Top1')
ax.set_ylim(45, 95)
ax.set_xlabel('LoRA rank')
ax.set_title(f'TTA Loss={loss_label} without fine-tuning, LoRA scale=2, modules={modules_label}')
plt.legend(loc='upper right', fontsize=8)

# +
# select sweep_id
sweep_ids = ['<entitiy>/<project>/<sweep id>']
sweep_ids = ['ykojima/mae_clip_mem_lora_without_finetune/dyay6ue6']

runs = []
for sid in sweep_ids:
    api = wandb.Api()
    sweep = api.sweep(sid)
    runs.extend(list(sweep.runs))
runs_summary = runs_params_formatter(runs)
df = basic_formatter(runs_summary, metric)

# +
loss = df['tta.peft.loss'][:1][0]
loss_label = loss_mapping[str(loss)]
modules_label = 'kvqo'

categories = {'tta.all.imagenet_a.top1_after_tta': 'ImageNet-A',
              'tta.all.imagenet_r.top1_after_tta': 'ImageNet-R',
              'tta.all.imagenet_sketch.top1_after_tta': 'ImageNet-Sketch',
              'tta.all.imagenet.top1_after_tta': 'ImageNet-Val',
              'tta.all.imagenetv2.top1_after_tta': 'ImageNet-V2'}

fig, ax = plt.subplots()
for k,v in categories.items():
    tmp = df.dropna(subset=[k])
    ax.plot(tmp['model.peft.r'], tmp[k], 'o-', label=v)
ax.set_ylabel('Top1')
ax.set_ylim(45, 95)
ax.set_xlabel('LoRA rank')
ax.set_title(f'TTA Loss={loss_label} without fine-tuning, LoRA scale=2, modules={modules_label}')
plt.legend(loc='upper right', fontsize=8)
# -

