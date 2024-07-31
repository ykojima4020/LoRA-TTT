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
import torch
import torchvision
from torchvision import transforms as transforms
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.append('../')

from factory import RILSMAECLIPFactory, PretrainedOpenCLIPFactory, \
                    PretrainedOpenCLIPDecoderFineTuneFactory, PretrainedOpenCLIPDecoderEncoderFineTuneFactory, \
                    PretrainedHFOpenCLIPFactory
from evaluator import imagenet_config
from peft import LoraConfig, get_peft_model

# %%
from misc.config import load_config
from omegaconf import OmegaConf

config = '../config/default_local.yaml'
cfg = load_config(config)
OmegaConf.set_struct(cfg, True)

# global params
device = 'cuda'
mask_ratio = 0.75

# %%
inverse = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.26862954, 1/0.26130258, 1/0.27577711 ]),
                                transforms.Normalize(mean = [-0.48145466, -0.4578275, -0.40821073],
                                                     std = [ 1., 1., 1. ]),
                                transforms.ToPILImage() # transform tensor to pillow for simple visualization
                               ])


# %%
def zeroshot_classifier(model, classnames, templates):
        with torch.no_grad():
            zeroshot_weights = []
            for classname in tqdm(classnames):
                # 80 patterns per class
                texts = [template.format(classname) for template in templates] #format with class
                max_length = 15
                tokens = tokenizer(texts, padding=True, truncation=True, max_length=max_length)
                batch = {key: values.to(device) for key, values in tokens.items()}
                class_embeddings = model.text_encode(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]) #embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True) # the norm shape is torch.Size([80, 1])
                class_embedding = class_embeddings.mean(dim=0) # the mean shape is torch.Size([256])
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        return zeroshot_weights

def get_score(model, zeroshot_weights, target_image, target, logit_scale=100):
    with torch.no_grad():
        image_features = model.clip.image_encode(target_image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        scores = (logit_scale * image_features @ zeroshot_weights).softmax(dim=-1)
    return scores[0][target].item()


# %% [markdown]
# ### Model loading here

# %%
# example
model_path = '/home/ykojima/Desktop/clip/mae_clip/output/20240401_decoder_layer8_encoder_lora_finetune_pix_recon_hf_open_clip_cc3m_wd_0001_blr_1e5/checkpoint.pth'

factory = PretrainedHFOpenCLIPFactory(cfg.model, mae='pixel')
model, tokenizer, transform = factory.create()

# LoRA settings
model = model.to(device)

status = torch.load(model_path, map_location="cuda")
model.load_state_dict(status['model'])

# %%
# parameters fixed
for name, param in model.named_parameters():
    if ('decoder' in name):
        param.requires_grad = False
    if ('text_model.encoder' in name):
        param.requires_grad = False
    print(name, param.requires_grad)

# %%
zeroshot_weights = zeroshot_classifier(model.clip, imagenet_config.imagenet_classes, imagenet_config.imagenet_templates)

# %%
severity = 5
corruption = "zoom_blur"
dataset = torchvision.datasets.ImageFolder(root=f'/home/ykojima/dataset/imagenetv2-c/{corruption}/{severity}', transform=transform('valid'))
data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=2, shuffle=False)

# %%
# target image
target_image, target = dataset[3880]
print(target)
target_image = target_image.unsqueeze(0)
display(inverse(target_image[0]))

# just for visualization
model.load_state_dict(status['model'])
model.eval()

target_image = target_image.to(device)

with torch.no_grad():
    mae_loss, reconstruction, mask = model.mae(target_image)

initial_loss = mae_loss.item()
initial_score = get_score(model, zeroshot_weights, target_image, target)


# %%
def one_sample_ttt(model, optimizer, images, device, mask_ratio=0.75):
    images = images.to(device)
    mae_loss, reconstruction, mask = model.mae(images)
    mae_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return mae_loss.item()


# %%
# initialization
model.load_state_dict(status['model'])

# training parameters
lr = 1e-3
eps =  1e-8
weight_decay = 0.2
optimizer = torch.optim.AdamW(model.image_encoder.parameters(),
            eps=eps, lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay)
optimizer = torch.optim.SGD(model.image_encoder.parameters(), lr=lr, weight_decay=weight_decay)

epochs = [-1]
losses = [initial_loss]
scores = [initial_score]

for epoch in tqdm(range(0, 500)):
    model.train()
    loss = one_sample_ttt(model, optimizer, target_image, device)
    epochs.append(epoch+1)
    losses.append(loss)
    model.eval()
    score = get_score(model, zeroshot_weights, target_image, target)
    scores.append(score)
    
result_df = pd.DataFrame({'epoch': epochs, 'loss': losses, 'score': scores})

# %%
# pd.set_option('display.max_rows', None)
result_df

# %%
df = result_df[1:]
fig, ax1 = plt.subplots(figsize=(7,4))
ax2 = ax1.twinx()
df.plot(x='epoch', y='loss', ax=ax1, label='MAE Loss')
ax1.hlines(result_df['loss'][0], xmin=0, xmax=500, color='#1f77b4', linestyle='dashed', lw=1, label='Initial Loss')
ax1.set_ylabel('Loss')

df.plot(x='epoch', y='score', ax=ax2, color='orange', label='CLIP Score')
ax2.hlines(result_df['score'][0], xmin=0, xmax=500, color='orange', linestyle='dashed', lw=1, label='Initial Score')
ax2.set_yscale('log')
ax2.set_ylabel('CLIP Score')

handler1, label1 = ax1.get_legend_handles_labels()
handler2, label2 = ax2.get_legend_handles_labels()
ax2.get_legend().remove()
# 凡例をまとめて出力する
ax1.legend(handler1 + handler2, label1 + label2, loc=1, borderaxespad=0.)

ax1.set_xlim([1, 500])
plt.title(f'One Sample TTT (SGD, LR={lr}, WD={weight_decay})')

# %%
