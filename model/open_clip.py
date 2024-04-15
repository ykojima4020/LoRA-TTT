import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F

import diffdist.functional as diff_dist

import open_clip

from einops import rearrange

class OpenCLIPImageEncoder(nn.Module):

    def __init__(self, model):
        super().__init__()
        if not isinstance(model, open_clip.model.CLIP):
            raise TypeError

        self.patchify = model.visual.conv1
        self.cls_token = model.visual.class_embedding          # torch.Size([768])
        self.pos_embedding = model.visual.positional_embedding # torch.Size([197, 768]) 
        self.transformer = model.visual.transformer
        self.ln_pre = model.visual.ln_pre
        self.ln_post = model.visual.ln_post

        # [NOTE]: proj is not needed in OpenCLIPImageEncoder 
        self.proj = model.visual.proj

    def forward(self, x, shuffler=None): 
        x = self.patchify(x)				# torch.Size([B, 768, 14, 14]) 
        x = rearrange(x, 'b c h w -> b (h w) c')	# torch.Size([B, 196, 768])
        x = x + self.pos_embedding[1:]

        if shuffler:
             x = rearrange(x, 'b t c -> t b c')
             x, forward_indexes, backward_indexes = shuffler(x)
             x = rearrange(x, 't b c -> b t c')
        else:
             backward_indexes = None

        cls_token_pos = self.cls_token + self.pos_embedding[0]	# torch.Size([768])
        x = torch.cat([cls_token_pos.expand(x.shape[0], 1, -1), x], dim=1) # torch.Size([B, 197, 768])

        x = self.ln_pre(x)

        x = rearrange(x, 'b t c -> t b c')		# torch.Size([197, B, 768])
        x = self.transformer(x)
        x = rearrange(x, 't b c -> b t c')
        x = self.ln_post(x)			# NLD

        x = rearrange(x, 'b t c -> t b c')
        return x, backward_indexes

class OpenCLIPImageProjector(nn.Module):
    def __init__(self, model):
        super().__init__()
        if not isinstance(model, open_clip.model.CLIP):
            raise TypeError
        self.proj = model.visual.proj

    def forward(self, x):
        return x @ self.proj

class OpenCLIP(nn.Module):
    def __init__(self, image_encoder=None, image_projector=None, clip=None, temperature=0.07):
        super().__init__()
        if not isinstance(clip, open_clip.model.CLIP):
            raise TypeError
        self._image_encoder = image_encoder
        self._image_projector = image_projector
        self.clip = clip

        self._temperature = temperature
        self.logit_scale = nn.Parameter(torch.ones([]) *  self._temperature)
        self.cross_entropy = nn.CrossEntropyLoss()

    def loss(self, image_x, text_x):
        batch_size = image_x.shape[0]
        # get label globally
        labels = torch.arange(batch_size, dtype=torch.long, device=image_x.device) + batch_size * dist.get_rank()

        # [B, C]
        image_x = F.normalize(image_x, dim=-1)
        text_x = F.normalize(text_x, dim=-1)

        logits_per_img = image_x @ dist_collect(text_x).t()
        logits_per_text = text_x @ dist_collect(image_x).t()

        logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
        loss_img = self.cross_entropy(logits_per_img * logit_scale, labels)
        loss_text = self.cross_entropy(logits_per_text * logit_scale, labels)

        loss = 0.5 * (loss_img + loss_text)
        return loss, logit_scale

    def image_encode(self, image):
        # Getting Image and Text Features
        image_features = self._image_encoder(image)[0][0, :, :]
        image_embeddings = self._image_projector(image_features)
        return image_embeddings 

    def text_encode(self, input_ids, attention_mask=None):
        text_x = self.clip.encode_text(input_ids)
        return text_x

    def forward(self, batch):
        image_x = self.image_encode(batch['image'])
        text_x = self.text_encode(batch['input_ids'])
        loss, logit_scale = self.loss(image_x, text_x)

        return loss, logit_scale



#[NOTE]: https://github.com/NVlabs/GroupViT/blob/main/models/multi_label_contrastive.py#L24C1-L34C51
def dist_collect(x):
    """ collect all tensor from all GPUs
    args:
        x: shape (mini_batch, ...)
    returns:
        shape (mini_batch * num_gpu, ...)
    """
    x = x.contiguous()
    out_list = [torch.zeros_like(x, device=x.device, dtype=x.dtype).contiguous() for _ in range(dist.get_world_size())]
    out_list = diff_dist.all_gather(out_list, x)
    return torch.cat(out_list, dim=0).contiguous()


