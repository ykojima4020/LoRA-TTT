import torch
from torch import nn

class MAECLIP(nn.Module):
    def __init__(self, image_encoder, clip, mae, alpha=1.0):
        super().__init__()
        self.image_encoder = image_encoder
        self.clip = clip
        self.mae = mae
        self._alpha = alpha

    def forward(self, batch):
        clip_loss, logit_scale = self.clip(batch)
        mae_loss, reconstruction, mask = self.mae(batch['image'])
        total_loss = clip_loss + self._alpha * mae_loss
        return total_loss, clip_loss, mae_loss, reconstruction, logit_scale
