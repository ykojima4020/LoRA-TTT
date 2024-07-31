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
from wandb_adapter import runs_params_formatter, average_duplicate_rows

# +
# select sweep_id
sweep_ids = ['<entitiy>/<project>/<sweep id>']

runs = []
for sid in sweep_ids:
    api = wandb.Api()
    sweep = api.sweep(sid)
    runs.extend(list(sweep.runs))
print(len(runs))
# -

sweep_summary = runs_params_formatter(runs)

# +
# ablation_parameters
metric = 'tta.all.imagenet_a.top1_after_tta'

p = ['name', 'model.peft.r', 'model.peft.target_modules', 'reconst', metric]
df = sweep_summary[p]
print(f'number of total runs: {len(df)}')

# name to order
order_mapping = {'k_proj+v_proj+q_proj+out_proj': 'kvqo', 'v_proj+q_proj': 'vq', 'q_proj': 'q'}
df['model.peft.target_modules'] = df['model.peft.target_modules'].map(order_mapping)
order_mapping = {'kvqo': 1, 'vq': 2, 'q': 3}
df['model.peft.target_modules'] = df['model.peft.target_modules'].map(order_mapping)

df[metric] = df[metric].astype(float)

# df = average_duplicate_rows(df, ['model.peft.r', 'model.peft.target_modules', 'reconst'], 'best_ttt_enhancement')
df = df.sort_values(by=['model.peft.r', 'model.peft.target_modules'])
print(f'number of unique runs: {len(df)}')

# +
reconst = 'pixel' # pixel or feature
metrix_label = 'ImageNet-A Top1 Accuracy'

target = df[(df['reconst'] == reconst)]

# rank vs ttt_enhancement
fig, ax = plt.subplots()

for k,v in order_mapping.items():
    tmp = target[target['model.peft.target_modules'] == v]
    ax.plot(tmp['model.peft.r'], tmp[metric], 'o-', label=f'LoRA modules = {k}')
ax.axhline(64.06, linestyle='-.', color='red', alpha=0.5, label='ZERO (SOTA)')
ax.set_ylabel(metrix_label)
ax.set_xlabel('LoRA rank')
ax.set_ylim(45,70)
ax.set_title(reconst)
plt.legend()
# -


