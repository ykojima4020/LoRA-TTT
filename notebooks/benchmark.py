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
#     display_name: mae_clip
#     language: python
#     name: mae_clip
# ---

# %% [markdown]
# # Evaluation on zero-shot baseline performance on various datasets

# %%
import torch
import torchvision
import torchvision.transforms as transforms

import sys
sys.path.append('../')
from factory import PretrainedHFOpenCLIPFactory
from evaluator.evaluator import ZeroShotImageNetEvaluator
from evaluator.imagenet_config import simple_prompts, ensemble_prompts, imagenet_classes
from evaluator.imagenet_variant_config import imagenet_a_classes, imagenet_r_classes
from misc.config import load_config
from omegaconf import OmegaConf

# %%
# configurations
config = '../config/default_local.yaml'
reconst = 'feature'
device = 'cuda'

imagenet_a = '/path/to/imagenet-a'
imagenet_v2 = '/path/to/imagenet-v2'
imagenet_r = '/path/to/imagenet-r'
imagenet_sketch = '/path/to/imagenet-sketch'

# %%
cfg = load_config(config)
OmegaConf.set_struct(cfg, True)

factory = PretrainedHFOpenCLIPFactory(cfg.model, mae=reconst)
model, tokenizer, transform = factory.create()
model = model.to(device)

# %%
# ImageNet-V2
dataset = torchvision.datasets.ImageFolder(root=imagenet_v2, transform=transform('valid'))
evaluator = ZeroShotImageNetEvaluator(tokenizer, dataset, ensemble_prompts, imagenet_classes, device)
eval_res = evaluator(model.clip, update=False)
eval_res

# %%
# ImageNet-Sketch
dataset = torchvision.datasets.ImageFolder(root=imagenet_sketch, transform=transform('valid'))
evaluator = ZeroShotImageNetEvaluator(tokenizer, dataset, ensemble_prompts, imagenet_classes, device)
eval_res = evaluator(model.clip, update=False)
eval_res

# %%
# ImageNet-A
dataset = torchvision.datasets.ImageFolder(root=imagenet_a, transform=transform('valid'))
evaluator = ZeroShotImageNetEvaluator(tokenizer, dataset, ensemble_prompts, imagenet_a_classes, device)
eval_res = evaluator(model.clip, update=False)
eval_res

# %%
# ImageNet-R
dataset = torchvision.datasets.ImageFolder(root=imagenet_r, transform=transform('valid'))
evaluator = ZeroShotImageNetEvaluator(tokenizer, dataset, ensemble_prompts, imagenet_r_classes, device)
eval_res = evaluator(model.clip, update=False)
eval_res

# %% [markdown]
# ### OpenAI CLIP-ViT-B/16 baseline

# %%
import clip
from tqdm import tqdm

model, preprocess = clip.load("ViT-B/16", device=device)

# %%
def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

def evaluate(model, dataset, classes, device):
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classes]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    loader = torch.utils.data.DataLoader(dataset, batch_size=128, num_workers=4)

    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for i, (image_input, target) in enumerate(tqdm(loader)):
            image_input = image_input.to(device)
            target = target.to(device)
            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += image_input.size(0)
    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100
    return top1, top5


# %%
# ImageNetV2
dataset = torchvision.datasets.ImageFolder(root=imagenet_v2, transform=preprocess)
top1, top5 = evaluate(model, dataset, imagenet_classes, device)
print(top1, top5)

# %%
# ImageNet-A
dataset = torchvision.datasets.ImageFolder(root=imagenet_a, transform=preprocess)
top1, top5 = evaluate(model, dataset, imagenet_a_classes, device)
print(top1, top5)

# %%
# ImageNet-R
dataset = torchvision.datasets.ImageFolder(root=imagenet_r, transform=preprocess)
top1, top5 = evaluate(model, dataset, imagenet_r_classes, device)
print(top1, top5)

# %%
# ImageNet-Sketch
dataset = torchvision.datasets.ImageFolder(root=imagenet_sketch, transform=preprocess)
top1, top5 = evaluate(model, dataset, imagenet_classes, device)
print(top1, top5)

# %%
