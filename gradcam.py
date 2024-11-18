import torch
from PIL import Image
import numpy as np
from pathlib import Path
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

import cv2
def show_cam_on_image(img, mask, neg_saliency=False):

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)

    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam


def clip_grad_cam(model, image, target, text_embeddings):

    # model.train()
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

    model.zero_grad()
    return Image.fromarray(np.uint8(255 * vis))

