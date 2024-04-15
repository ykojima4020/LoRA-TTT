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
#     display_name: venv
#     language: python
#     name: venv
# ---

# %%
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

import sys
sys.path.append('/home/ykojima/Desktop/clip/mae_clip/external/robustness/ImageNet-C/imagenet_c')
sys.path.append('../')
from imagenet_c import corrupt

from imagenetv2_pytorch import ImageNetV2Dataset

from misc.transforms import Corruption, get_corruption_transform

# %%
corruptions = ['brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog',
               'frost', 'gaussian_noise', 'glass_blur', 'impulse_noise', 'jpeg_compression',
               'motion_blur', 'pixelate', 'shot_noise', 'snow', 'zoom_blur']


# %%
def show_all_corruptions(images, severity, row=3, col=5):
    n_data = row * col # 表示するデータ数
    # row=3 # 行数
    # col=5 # 列数
    fig, ax = plt.subplots(nrows=row, ncols=col, figsize=(8,6))
    for i, (image, corruption) in enumerate(zip(images, corruptions)):
        _r= i//col
        _c= i%col
        ax[_r,_c].set_title(corruption, fontsize=8)
        ax[_r,_c].imshow(image)
        ax[_r,_c].axes.xaxis.set_visible(False) # X軸を非表示に
        ax[_r,_c].axes.yaxis.set_visible(False)
    plt.subplots_adjust(wspace=0.05, hspace=-0.2)
    fig.suptitle(f"severity={severity}", fontsize=10, y=0.15)
    # plt.title(f"severity={severity}")


# %%
image_id = 100

transform = get_corruption_transform(Corruption(severity=0, corruption_name='defocus_blur'))
transform = transforms.Compose(transform.transforms[:2])
dataset = ImageNetV2Dataset(transform=transform)
print(transform)
display(dataset[image_id][0])

for s in range(1, 6):
    images = []
    for c in corruptions:
        transform = get_corruption_transform(Corruption(severity=s, corruption_name=c))
        # [NOTE]: remove ToTensor, Normalize
        transform = transforms.Compose(transform.transforms[:3])
        dataset = ImageNetV2Dataset(transform=transform)
        image = Image.fromarray(dataset[image_id][0])
        images.append(image)
    show_all_corruptions(images, s)

# %%
image_id = 100
images = []
severities = list(range(1, 6))
corruption = 'gaussian_noise'
for s in severities:
    transform = get_corruption_transform(Corruption(severity=s, corruption_name=corruption))
    # [NOTE]: remove ToTensor, Normalize
    transform = transforms.Compose(transform.transforms[:3])
    dataset = ImageNetV2Dataset(transform=transform)
    image = Image.fromarray(dataset[image_id][0])
    images.append(image)
fig, ax = plt.subplots(nrows=1, ncols=len(images), figsize=(8,6))
for i, (image, severity) in enumerate(zip(images, severities)):
    ax[i].set_title(f'{severity}', fontsize=8)
    ax[i].imshow(image)
    ax[i].axes.xaxis.set_visible(False) # X軸を非表示に
    ax[i].axes.yaxis.set_visible(False)
plt.subplots_adjust(wspace=0.05, hspace=-0.2)
fig.suptitle(f"{corruption}", fontsize=10, y=0.38)

# %%
image = Image.fromarray(dataset[0][0])
image

# %%

# %%
