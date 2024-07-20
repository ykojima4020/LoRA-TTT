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
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

import sys
sys.path.append('../../../external/CLIP_Explainability/code/')
from image_utils import show_cam_on_image

# %%
device='cuda'

# %%
model_vit = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", output_attentions=True).to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# %%
vision_encoder = model_vit.vision_model

# %%
# image_path = '/data2/yuto/dataset/imagenetv2-c/original/0/0000/4804.jpg'
image_path = '/data2/yuto/mae_clip/notebooks/loss/gradcam/horse&dog.jpg'
img = Image.open(image_path)
text = 'a horse'

# %%
inputs = processor(text=[text], images=img, return_tensors="pt", padding=True)
inputs['input_ids'] = inputs['input_ids'].to(device)
inputs['attention_mask'] = inputs['attention_mask'].to(device)
inputs['pixel_values'] = inputs['pixel_values'].to(device)

# %%
image = inputs['pixel_values']
image.shape

# %%
outputs = model_vit(**inputs)

# %%
attention_weight = outputs.vision_model_output.attentions[11]
attention_weight.retain_grad()

# %%
image_features = outputs.image_embeds
target_features = outputs.text_embeds

image_features_norm = image_features.norm(dim=-1, keepdim=True)
image_features_new = image_features / image_features_norm
target_features_norm = target_features.norm(dim=-1, keepdim=True)
target_features_new = target_features / target_features_norm

objective = image_features_new[0].dot(target_features_new[0])
print(objective)

# %%
model_vit.zero_grad()
objective.backward()

# %%
last_attn = attention_weight.detach()
print(last_attn.shape)
last_attn = last_attn.reshape(-1, last_attn.shape[-1], last_attn.shape[-1])
print(last_attn.shape)

last_grad = attention_weight.grad.detach()
last_grad = last_grad.reshape(-1, last_grad.shape[-1], last_grad.shape[-1])

print(last_grad.shape)

# %%
cam = last_grad * last_attn
cam = cam.clamp(min=0).mean(dim=0) 
image_relevance = cam[0, 1:]

# %%
# image_relevance = image_relevance.reshape(1, 1, 14, 14)
image_relevance = image_relevance.reshape(1, 1, 7, 7)
image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bilinear')
image_relevance = image_relevance.reshape(224, 224).data.cpu().numpy()
image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
image = image[0].permute(1, 2, 0).data.cpu().numpy()
image = (image - image.min()) / (image.max() - image.min())

# %%
vis = show_cam_on_image(image, image_relevance, neg_saliency=False)
vis = np.uint8(255 * vis)
vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)

plt.imshow(vis)


# %%
