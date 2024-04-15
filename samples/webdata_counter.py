import sys
sys.path.append('../')

from tqdm import tqdm

from mae_clip_training import get_args_parser
from misc.config import get_config

from braceexpand import braceexpand
import os.path as osp
import webdataset as wds


def get_webdataset(cfg, mode='train'): 
    dataset_type = None
    tar_file_list = []
    total_length = 0
    for ds in cfg.dataset[mode]:
        ds_meta = cfg.dataset.meta[ds]
        if dataset_type is None:
            dataset_type = ds_meta.type
        else:
            assert dataset_type == ds_meta.type, \
                'All datasets must be of the same type'

        prefix = ds_meta.prefix
        path = ds_meta.path
        length = ds_meta.length
        cur_tar_file_list = []
        for tar_file in braceexpand(osp.join(path, prefix)):
            if osp.exists(tar_file):
                cur_tar_file_list.append(tar_file)
        print(f'Found {len(cur_tar_file_list)} files for dataset {ds}')
        tar_file_list.extend(cur_tar_file_list)
        total_length += length
    print(f'Found {len(tar_file_list)} files in total for split {mode}')

    dataset = wds.WebDataset(tar_file_list)
    return dataset

args = get_args_parser()
args = args.parse_args()
cfg = get_config(args)

dataset = get_webdataset(cfg.data)

n_data = 0
for batch in tqdm(dataset):
    n_data += 1

print(n_data)

