import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F

import diffdist.functional as diff_dist

class CLIP(nn.Module):
    def __init__(self, image_encoder, text_encoder, image_projection, text_projection, temperature=0.07):
        super().__init__()
        self._image_encoder = image_encoder
        self._text_encoder = text_encoder
        self._image_projection = image_projection
        self._text_projection = text_projection
        self._temperature = temperature
        self.logit_scale = nn.Parameter(torch.ones([]) *  self._temperature)
        self.cross_entropy = nn.CrossEntropyLoss()

    def image_encode(self, image):
        # Getting Image and Text Features
        image_features = self._image_encoder(image)[0][0, :, :]
        image_embeddings = self._image_projection(image_features)
        return image_embeddings 

    def text_encode(self, input_ids, attention_mask):
        text_features = self._text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_embeddings = self._text_projection(text_features)
        return text_embeddings

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


    def forward(self, batch):
        image_x = self.image_encode(batch['image'])
        text_x = self.text_encode(batch['input_ids'], batch['attention_mask'])
        loss, logit_scale = self.loss(image_x, text_x)

        return loss, logit_scale

    def get_transforms(self, mode):
        return self._image_encoder.get_transforms(mode)

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


if __name__ == '__main__':
    images = torch.randn(8, 3, 224, 224)
    input_ids = torch.randint(5, 300, size=(8, 25))
    attention_mask = torch.ones(8, 25)
    batch = {
        'image': images,
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }

    # clip = CLIP()
    # loss = clip(batch)
