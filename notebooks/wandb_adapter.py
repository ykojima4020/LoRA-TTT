import wandb
import pandas as pd

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
        if run._state == 'failed':
            print(f'{run.name} is {run._state} run and excluded.')
            continue
        summary_dict = flatten_dict(run.summary._json_dict, sep='.')
        config_dict = flatten_dict(run.config, sep='.')
        # [NOTE]: to remove complitity
        config_dict['model.peft.target_modules'] = '+'.join(config_dict['model.peft.target_modules'])
        name_dict = {'name': run.name}
        all_dict = {**summary_dict, **config_dict, **name_dict}
        df = pd.DataFrame.from_dict(all_dict, orient='index').T
        df_list.append(df)

    sweep_summary = pd.concat(df_list)
    # sweep_summary['best_ttt_enhancement'] = sweep_summary['best_ttt_enhancement'].astype(float)
    return sweep_summary

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

