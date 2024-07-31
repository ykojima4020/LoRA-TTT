# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: mae_clip
#     language: python
#     name: mae_clip
# ---

# %%
import torch
from imagenetv2_pytorch import ImageNetV2Dataset

import sys
sys.path.append('../')
from factory import RILSMAECLIPFactory, PretrainedOpenCLIPFactory, PretrainedHFOpenCLIPFactory
from evaluator.evaluator import ZeroShotImageNetEvaluator
from evaluator.imagenet_config import simple_prompts, ensemble_prompts, imagenet_classes

# %%
from misc.config import load_config
from omegaconf import OmegaConf

config = '../config/default_local.yaml'
cfg = load_config(config)
OmegaConf.set_struct(cfg, True)

model_path = '/path/to/your/model'
reconst = 'feature'
device = 'cuda'

# %%
factory = PretrainedHFOpenCLIPFactory(cfg.model, mae=reconst)
model, tokenizer, transform = factory.create()
model = model.to(device)

status = torch.load(model_path, map_location="cuda")

model.load_state_dict(status['model'])
model.eval()

# %%
# imagenetV2 evaluation
dataset = ImageNetV2Dataset(transform=transform('valid'))
evaluator = ZeroShotImageNetEvaluator(tokenizer, dataset, simple_prompts, imagenet_classes, device)
eval_res = evaluator(model.clip, update=False)
print(eval_res)

# %%
