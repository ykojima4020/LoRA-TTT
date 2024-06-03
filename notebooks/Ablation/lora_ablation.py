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

import wandb
import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# +
# select sweep_id
sweep_id = '<entitiy>/<project>/<sweep id>'
sweep_id = 'ykojima/mae_clip_lora_finetuning/up3tbiw4'
# sweep_id = 'ykojima/mae_clip_lora_finetuning/863r1fiq'
# sweep_id = 'ykojima/mae_clip_lora_finetuning/4829xrou'

api = wandb.Api()
sweep = api.sweep(sweep_id)
# -

print(sweep.state)
print(sweep.expected_run_count)

# +
# sweep.display()
# -

runs = sweep.runs
runs = list(runs)
print(len(runs))

# [NOTE]: add additional runs to the sweep
run_id = 'ykojima/mae_clip_lora_finetuning/i390gl8e'
r = api.run(run_id)
runs.append(r)


def flatten_dict(d, parent_key='', sep='_'):
    """
    辞書をフラットにします。
    
    Args:
        d (dict): フラット化する辞書
        parent_key (str): 再帰的に呼び出された際の親キー
        sep (str): 親キーと子キーの間に挿入される区切り文字
    
    Returns:
        dict: フラット化された辞書
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# the following is reference code of how summary data is downloaded
#
# ```
# import pandas as pd 
# import wandb
# api = wandb.Api()
#
# # Project is specified by <entity/project-name>
# runs = api.runs("ykojima/mae_clip_lora_finetuning")
#
# summary_list, config_list, name_list = [], [], []
# for run in runs: 
#     # .summary contains the output keys/values for metrics like accuracy.
#     #  We call ._json_dict to omit large files 
#     summary_list.append(run.summary._json_dict)
#
#     # .config contains the hyperparameters.
#     #  We remove special values that start with _.
#     config_list.append(
#         {k: v for k,v in run.config.items()
#           if not k.startswith('_')})
#
#     # .name is the human-readable name of the run.
#     name_list.append(run.name)
#
# runs_df = pd.DataFrame({
#     "summary": summary_list,
#     "config": config_list,
#     "name": name_list
#     })
#
# runs_df.to_csv("project.csv")
# ```

def runs_params_formatter(runs):
    df_list = []
    for run in runs: 
        summary_dict = flatten_dict(run.summary._json_dict, sep='.')
        config_dict = flatten_dict(run.config, sep='.')
        # [NOTE]: to remove complitity
        config_dict['model.peft.target_modules'] = '+'.join(config_dict['model.peft.target_modules'])
        name_dict = {'name': run.name}
        all_dict = {**summary_dict, **config_dict, **name_dict}
        df = pd.DataFrame.from_dict(all_dict, orient='index').T
        df_list.append(df)

    sweep_summary = pd.concat(df_list)
    sweep_summary['best_ttt_enhancement'] = sweep_summary['best_ttt_enhancement'].astype(float)
    return sweep_summary


sweep_summary = runs_params_formatter(runs)
sweep_summary


# 平均化するセルの値が NaN の場合は、その値は平均値に考慮されない
def average_duplicate_rows(df, columns_to_group, column_to_average):
    # 重複している行を特定
    duplicated_rows = df[df.duplicated(subset=columns_to_group, keep=False)]

    # 重複している行の特定のカラムの平均値を計算
    mean_values = duplicated_rows.groupby(columns_to_group).agg({column_to_average: 'mean'}).reset_index()

    # 平均値を持つ行で元の行を置き換える
    for idx, row in mean_values.iterrows():
        filter_condition = df[columns_to_group].eq(row[columns_to_group]).all(axis=1)
        df.loc[filter_condition, column_to_average] = row[column_to_average]

    # 重複を削除
    df.drop_duplicates(subset=columns_to_group, keep='first', inplace=True)

    return df


# +
# ablation_parameters
p = ['name', 'model.peft.r', 'model.peft.target_modules', 'reconst', 'best_ttt_enhancement']
df = sweep_summary[p]
print(f'number of total runs: {len(df)}')
# print(df[(df['reconst'] == 'feature') & (df['model.peft.r'] == 8) & (df['model.peft.target_modules'] == 'k_proj+v_proj+q_proj+out_proj')])

# extract finished runs
df = df.dropna(subset=['best_ttt_enhancement'])
print(f'number of finished runs: {len(df)}')

# name to order
order_mapping = {'k_proj+v_proj+q_proj+out_proj': 'kvqo', 'v_proj+q_proj': 'vq', 'q_proj': 'q'}
df['model.peft.target_modules'] = df['model.peft.target_modules'].map(order_mapping)
order_mapping = {'kvqo': 1, 'vq': 2, 'q': 3}
df['model.peft.target_modules'] = df['model.peft.target_modules'].map(order_mapping)

df = average_duplicate_rows(df, ['model.peft.r', 'model.peft.target_modules', 'reconst'], 'best_ttt_enhancement')
df = df.sort_values(by=['model.peft.r', 'model.peft.target_modules'])
print(f'number of unique runs: {len(df)}')
print(df[df['reconst'] == 'feature'])

# +
reconst = 'feature' # pixel or feature
target = df[(df['reconst'] == reconst)]
print(target)

# create heatmap
heatmap_data = target.pivot(index='model.peft.r', columns='model.peft.target_modules', values='best_ttt_enhancement')

plt.figure()
sns.heatmap(heatmap_data, annot=True, cmap="Reds", fmt=".3f")
plt.title(f'{reconst} TTT Enhancement')
plt.xlabel('LoRA Target Modules')
plt.xticks([x-0.5 for x in list(order_mapping.values())], labels=order_mapping.keys())
plt.ylabel('Rank')
plt.show()

# rank vs ttt_enhancement
fig, ax = plt.subplots()

for k,v in order_mapping.items():
    tmp = target[target['model.peft.target_modules'] == v]
    ax.plot(tmp['model.peft.r'], tmp['best_ttt_enhancement'], 'o-', label=f'LoRA modules = {k}')
ax.set_ylabel('Top1 enhancement')
ax.set_xlabel('LoRA rank')
ax.set_ylim(-3,1)
ax.set_title(reconst)
plt.legend()

fig, ax = plt.subplots()
print(order_mapping.keys())

alignment = -0.3
for r in [1, 8, 64]: # LoRA rank
    tmp = target[target['model.peft.r'] == r]
    ax.bar(tmp['model.peft.target_modules'] + alignment, tmp['best_ttt_enhancement'], width=0.3, label=f'rank = {r}')
    alignment += 0.3
ax.set_ylabel('Top1 enhancement')
ax.set_xlabel('LoRA target_modules')
plt.xticks([x for x in list(order_mapping.values())], labels=order_mapping.keys())
ax.set_title(reconst)
plt.legend()
# -




