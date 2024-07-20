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

# %%
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

import sys
sys.path.append('../../../')

from factory import PretrainedHFOpenCLIPFactory
from tta import MEMLoss, zeroshot_weights, build_tta_optimizer, accuracy
from evaluator.imagenet_config import simple_prompts, ensemble_prompts, imagenet_classes
from misc.config import load_config
from omegaconf import OmegaConf
from misc.tpt_transforms import AugMixAugmenter

sys.path.append('../../../external/CLIP_Explainability/code/')
from image_utils import show_cam_on_image

# %%
# configurations
config = '../../../config/mae_clip_run.yaml'
device = 'cuda'
config = load_config(config)
OmegaConf.set_struct(config, True)

factory = PretrainedHFOpenCLIPFactory(config.model, mae=config.reconst)
model, tokenizer, transform = factory.create()
model = model.to(device)
# without fine-tuning
status = model.mae.state_dict()

# %%
# [NOTE]: Data augmentation
normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])
base_transform = transforms.Compose([
                transforms.Resize(224, interpolation=BICUBIC),
                transforms.CenterCrop(224)])
preprocess = transforms.Compose([
                transforms.ToTensor(),
                normalize])
batch_size = 64
tta_transform = AugMixAugmenter(base_transform, preprocess, n_views=batch_size-1,
                               augmix=False)

# %%
# optimizer
optimizer = build_tta_optimizer(model, config.tta['peft'])

# Loss
loss = MEMLoss()

# %%
image_path = '/path/to/image'
image = Image.open(image_path)
images = tta_transform(image)
target = torch.tensor([0])
classes = imagenet_classes
prompts = ensemble_prompts

for k in range(len(images)):
    images[k] = images[k].to(device)
target = target.to(device)
image = images[0].unsqueeze(0)
images = torch.stack(images)

# %%
text_embeddings = zeroshot_weights(model.clip, tokenizer, classes, prompts, device)

# %%
# [NOTE]: inference before TTA
model.mae.load_state_dict(status)
model.eval()
model.clip = model.clip.to(device) # why?
with torch.no_grad():
    with torch.cuda.amp.autocast():
        image_features = model.clip.image_encode(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        output = image_features @ text_embeddings
acc1, acc5 = accuracy(output, target, topk=(1, 5))
print(acc1, acc5)

# %%
cam_image = clip_grad_cam(model, image, target[0])
cam_image

# %%
# Single Instance TTA
for name, param in model.image_encoder.named_parameters():
    if 'lora' in name:
        param.requires_grad = True

# [TODO]: should load only LoRA and Decoder, not update text_encoder
model.mae.load_state_dict(status)
model.train()
for j in range(1): # epoch is once
    l = loss(model, images, text_embeddings)
    optimizer.zero_grad()
    l.backward()
    optimizer.step()

# %%
# [NOTE]: inference
model.eval()
model.clip = model.clip.to(device) # why?
with torch.no_grad():
    with torch.cuda.amp.autocast():
        image_features = model.clip.image_encode(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        output = image_features @ text_embeddings
acc1, acc5 = accuracy(output, target, topk=(1, 5))
print(acc1, acc5)


# %%
def clip_grad_cam(model, image, target):

    model.train()
    # [NOTE]: the model shoud be trainable
    for name, param in model.image_encoder.named_parameters():
        param.requires_grad = True
    
    image_feature = model.clip.image_encode(image)
    image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)

    similarity = image_feature @ text_embeddings
    objective = similarity[0][target]

    # [NOTE]: calculate gradiation of the intermediate tensor
    attention_weight = model.image_encoder.get_attn()[11]
    attention_weight.retain_grad()
    
    model.zero_grad()
    objective.backward()

    # [NOTE]: GradCAM calculation
    last_attn = attention_weight.detach()
    last_attn = last_attn.reshape(-1, last_attn.shape[-1], last_attn.shape[-1])

    last_grad = attention_weight.grad.detach()
    last_grad = last_grad.reshape(-1, last_grad.shape[-1], last_grad.shape[-1])

    cam = last_grad * last_attn
    cam = cam.clamp(min=0).mean(dim=0)
    image_relevance = cam[0, 1:]

     # [NOTE]: create image
    image_relevance = image_relevance.reshape(1, 1, 14, 14)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bilinear')
    image_relevance = image_relevance.reshape(224, 224).data.cpu().numpy()
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    image = image[0].permute(1, 2, 0).data.cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())
    vis = show_cam_on_image(image, image_relevance, neg_saliency=False)
    vis = vis[..., ::-1] # BGR RGB convert

    # [NOTE]: the model shoud be froze 
    for name, param in model.image_encoder.named_parameters():
        param.requires_grad = False

    return Image.fromarray(np.uint8(255 * vis))


# %%
cam_image = clip_grad_cam(model, image, target[0])

# %%
cam_image

# %%
