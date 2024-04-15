# This file is based on https://github.com/NVlabs/GroupViT/blob/main/utils/config.py. 
# -------------------------------------------------------------------------
# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/GroupViT/blob/main/LICENSE
#
# Written by Jiarui Xu
# -------------------------------------------------------------------------

import os
import os.path as osp

from omegaconf import OmegaConf

def load_config(cfg_file):
    cfg = OmegaConf.load(cfg_file)
    if '_base_' in cfg:
        if isinstance(cfg._base_, str):
            base_cfg = OmegaConf.load(osp.join(osp.dirname(cfg_file), cfg._base_))
        else:
            base_cfg = OmegaConf.merge(OmegaConf.load(f) for f in cfg._base_)
        cfg = OmegaConf.merge(base_cfg, cfg)
    return cfg


def get_config(args):
    cfg = load_config(args.cfg)
    # https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#struct-flag
    # prevent the creation of fields that do not exist
    OmegaConf.set_struct(cfg, True)

    if args.opts is not None:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.opts))

    if hasattr(args, 'batch_size') and args.batch_size:
        cfg.data.batch_size = args.batch_size

    if hasattr(args, 'output') and args.output:
        cfg.output = args.output
    else:
        cfg.output = osp.join('./tmp', cfg.model_name)

    if hasattr(args, 'wandb') and args.wandb:
        cfg.wandb = args.wandb

    if hasattr(args, 'test') and args.test:
        cfg.test = args.test

    if hasattr(args, 'device') and args.device:
        cfg.device = args.device

    if hasattr(args, 'epochs') and args.epochs:
        cfg.train.epochs = args.epochs

    if hasattr(args, 'type') and args.type:
        cfg.type = args.type

    if hasattr(args, 'reconst') and args.reconst:
        cfg.reconst = args.reconst

    return cfg
