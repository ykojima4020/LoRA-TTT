# -------------------------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE
#
# Written by Ze Liu, Zhenda Xie
# Modified by Jiarui Xu
# Modified by Y Kojima
# -------------------------------------------------------------------------

import torch.optim as optim
import timm.optim.optim_factory as optim_factory

def build_optimizer(config, model):
    """Build optimizer, set weight decay of normalization to 0 by default."""

    opt_name = config.optimizer.name
    optimizer = None
    if opt_name == 'adamw':
        if config.optimizer.param_groups:
            param_groups = optim_factory.param_groups_weight_decay(model, config.weight_decay)
            optimizer = optim.AdamW(
                param_groups,
                eps=config.optimizer.eps,
                betas=config.optimizer.betas,
                lr=config.lr)
        else:
            optimizer = optim.AdamW(
                model.parameters(),
                eps=config.optimizer.eps,
                betas=config.optimizer.betas,
                lr=config.lr,
                weight_decay=config.weight_decay)
    else:
        raise ValueError(f'Unsupported optimizer: {opt_name}')

    return optimizer