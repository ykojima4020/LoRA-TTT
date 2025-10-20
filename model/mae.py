'''
This code refers to https://github.com/ykojima4020/tta_mae_detection/blob/main/model/model.py.
'''

import torch
import timm
from timm.utils import ModelEmaV2
import numpy as np

from einops import repeat, rearrange
from einops.layers.torch import Rearrange

from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block

def random_indexes(size : int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes) # return indexes of the sorted tensor
    return forward_indexes, backward_indexes

def eff_random_indexes(size: int, batch_size: int, device):
    forward_indexes = torch.stack([torch.randperm(size, device=device) for _ in range(batch_size)], dim=1)
    backward_indexes = torch.argsort(forward_indexes, dim=0)
    return forward_indexes, backward_indexes

def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))

class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, patches : torch.Tensor):
        '''
        Returns:
            forward_indexes: shuffled indexes 
            backward_indexes: index of the shuffled indexes 
        '''
        T, B, C = patches.shape
        remain_T = int(T * (1 - self.ratio))

        # indexes = [random_indexes(T) for _ in range(B)]
        # forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)	# torch.Size([196, B])
        # backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)	# torch.Size([196, B])
        # 適切なデバイス上でインデックスを一括生成
        forward_indexes, backward_indexes = eff_random_indexes(T, B, patches.device)

        patches = take_indexes(patches, forward_indexes)	# torch.Size([196, B, 768])
        patches = patches[:remain_T]				# torch.Size([49, B, 768])

        return patches, forward_indexes, backward_indexes

class MyMasking(torch.nn.Module):
    def __init__(self, ratio, mask_indices) -> None:
        super().__init__()
        self.ratio = ratio
        self.mask_indices = mask_indices

    def forward(self, patches : torch.Tensor):
        T, B, C = patches.shape
        remain_T = int(T * (1 - self.ratio))

        forward_indexes = None
        backward_indexes = None

        patches = take_indexes(patches, self.mask_indices)		#  torch.Size([196, B, 768])
        patches = patches[:remain_T]

        return patches, forward_indexes, backward_indexes


class ImageEncoder(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=12,
                 num_head=3
                 ) -> None:
        super().__init__()

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2, 1, emb_dim))

        self.patchify = torch.nn.Conv2d(3, emb_dim, patch_size, patch_size)

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, img, shuffler=None):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding

        if shuffler:
             patches, forward_indexes, backward_indexes = shuffler(patches)
        else:
             backward_indexes = None

        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')

        return features, forward_indexes, backward_indexes

class MAEFeatureDecoder(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=4,
                 num_head=3,
                 ) -> None:
        super().__init__()

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2 + 1, 1, emb_dim))

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.head = torch.nn.Linear(emb_dim, 3 * patch_size ** 2)
        self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=image_size//patch_size)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, backward_indexes):
        T = features.shape[0]
        backward_indexes = torch.cat([torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1], dim=0)
        features = torch.cat([features, self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)], dim=0)
        features = take_indexes(features, backward_indexes)
        features = features + self.pos_embedding

        features = rearrange(features, 't b c -> b t c')
        features = self.transformer(features)
        features = rearrange(features, 'b t c -> t b c')

        return features, None

class MAEPixelDecoder(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=4,
                 num_head=3,
                 ) -> None:
        super().__init__()

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2 + 1, 1, emb_dim))

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.head = torch.nn.Linear(emb_dim, 3 * patch_size ** 2)
        self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=image_size//patch_size)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, backward_indexes):
        T = features.shape[0]
        backward_indexes = torch.cat([torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1], dim=0)
        features = torch.cat([features, self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)], dim=0)
        features = take_indexes(features, backward_indexes)
        features = features + self.pos_embedding

        features = rearrange(features, 't b c -> b t c')
        features = self.transformer(features)
        features = rearrange(features, 'b t c -> t b c')
        features = features[1:] # remove global feature

        patches = self.head(features)
        mask = torch.zeros_like(patches)
        mask[T-1:] = 1
        mask = take_indexes(mask, backward_indexes[1:] - 1)
        img = self.patch2img(patches)
        mask = self.patch2img(mask)

        return img, mask

class PixelMAE(torch.nn.Module):
    def __init__(self, encoder, decoder, mask_ratio) -> None: 
        super().__init__()

        self._shuffler = PatchShuffle(mask_ratio)
        self.encoder = encoder 
        self.decoder = decoder
        self.mask_ratio = mask_ratio

    def forward(self, image):
        features, _, backward_indexes = self.encoder(image, self._shuffler)
        reconstruction, mask = self.decoder(features,  backward_indexes)
        loss = torch.mean((reconstruction - image) ** 2 * mask) / self.mask_ratio
        return loss, reconstruction, mask

# [NOTE]: This EMA function will be deprecated soon.
class FeatureEMAMAE(torch.nn.Module):
    def __init__(self, encoder, decoder, mask_ratio, decay=0.9999) -> None: 
        super().__init__()

        self._shuffler = PatchShuffle(mask_ratio)
        self.encoder = encoder 
        self.ema = ModelEmaV2(self.encoder, decay=decay)
        self.decoder = decoder
        self.mask_ratio = mask_ratio

    def forward(self, image):
        # [TODO]: only proceed if model.train()
        if self.training:
            self.ema.update(self.encoder)

        # [NOTE]: extract class token features from EMA encoder
        image_features = self.ema.module(image)[0][0, :, :]

        features, _, backward_indexes = self.encoder(image, self._shuffler)
        reconstruction, mask = self.decoder(features,  backward_indexes)

        loss = torch.mean((reconstruction - image_features) ** 2) / self.mask_ratio
        # [NOTE]: output image as dummy
        return loss, image, mask

class CLSTokenMAE(torch.nn.Module):
    def __init__(self, encoder, decoder, mask_ratio) -> None:
        super().__init__()

        self._shuffler = PatchShuffle(mask_ratio)
        self.encoder = encoder
        self.decoder = decoder
        self.mask_ratio = mask_ratio

    def forward(self, image):
        # [NOTE]: extract class token features from EMA encoder
        with torch.no_grad():
            cls_token = self.encoder(image)[0][0, :, :]

        features, _, backward_indexes = self.encoder(image, self._shuffler)
        reconstruct_features, mask = self.decoder(features,  backward_indexes)
        reconstruct_cls_token = reconstruct_features[0]

        loss = torch.mean((reconstruct_cls_token - cls_token) ** 2) / self.mask_ratio
        # [NOTE]: output image as dummy
        return loss, image, mask

class CLSTokenMAEWithoutDecoder(torch.nn.Module):
    def __init__(self, encoder, decoder, mask_ratio) -> None:
        super().__init__()

        if not isinstance(decoder, type(None)):
            print(f'{type(decoder)} is not supported and should be None.')
            raise TypeError
        self._shuffler = PatchShuffle(mask_ratio)
        self.encoder = encoder
        self.decoder = torch.nn.Module()
        self.mask_ratio = mask_ratio

    def forward(self, image, coefficient=None):
        # [NOTE]: extract class token features from EMA encoder
        with torch.no_grad():
            cls_token = self.encoder(image)[0][0, :, :]					# torch.Size([B, 768])

        mask_cls_token = self.encoder(image, self._shuffler)[0][0, :, :]		# torch.Size([B, 768])

        per_sample_loss = torch.mean((mask_cls_token - cls_token) ** 2, dim=1) / self.mask_ratio  # 各サンプルごとの損失

        if coefficient is not None:
            per_sample_loss = per_sample_loss * coefficient

        # [NOTE]: バッチ全体の平均損失を計算
        loss = torch.mean(per_sample_loss)

        # [NOTE]: output image as dummy
        return loss, image, None

class FeatureAllMAEWithoutDecoder(torch.nn.Module):
    def __init__(self, encoder, decoder, mask_ratio) -> None:
        super().__init__()

        if not isinstance(decoder, type(None)):
            print(f'{type(decoder)} is not supported and should be None.')
            raise TypeError
        self._shuffler = PatchShuffle(mask_ratio)
        self.encoder = encoder
        self.decoder = torch.nn.Module()
        self.mask_ratio = mask_ratio

    def forward(self, image):
        # [NOTE]: extract class token features from EMA encoder
        with torch.no_grad():
            target_features = self.encoder(image)[0][1:]					# torch.Size([196, B, 768])
        T, B, C = target_features.shape
        remain_T = int(T * (1 - self.mask_ratio))

        image_features, forward_indexes, _ = self.encoder(image, self._shuffler)		# features.shape is torch.Size([50, 64, 768])
        image_features = image_features[1:, :, :]						# torch.Size([49, B, 768])

        target_features = take_indexes(target_features, forward_indexes)			# torch.Size([196, 6, 768])
        visible_target_features = target_features[:remain_T]					# torch.Size([49, 6, 768])

        loss = torch.mean((image_features - visible_target_features) ** 2) / self.mask_ratio
        # [NOTE]: output image as dummy
        return loss, image, None


class UncertaintyMaskingMAE(torch.nn.Module):
    def __init__(self, encoder, decoder, mask_ratio) -> None:
        super().__init__()

        if not isinstance(decoder, type(None)):
            print(f'{type(decoder)} is not supported and should be None.')
            raise TypeError
        self.encoder = encoder
        # [NOTE] add dropout in order to perform MC Dropout
        self.encoder.transformer.layers[0].mlp.fc1 = torch.nn.Sequential(
                                                     self.encoder.transformer.layers[0].mlp.fc1,
                                                     torch.nn.Dropout(p=0.2)
                                                     )
        self.decoder = torch.nn.Module()
        self.mask_ratio = mask_ratio
        self.n_forward = 10

    def forward(self, image):
        if not self.encoder.training:
            raise RuntimeError('Model is not in trainig mode.')

        n_outputs = []
        for i in range(self.n_forward):
            with torch.no_grad():
                image_features = self.encoder(image)[0]				# torch.Size([197, B, 768])
                visual_tokens = image_features[1:, :, :]			# torch.Size([196, B, 768])
                visual_tokens = visual_tokens.mean(dim=2)			# torch.Size([196, B])
                n_outputs.append(visual_tokens)
        stacked_outputs_ = torch.stack(n_outputs, dim=0)			# torch.Size([10, 196, B])

        variance = torch.var(stacked_outputs_, dim=0)				# torch.Size([196, 6])

        # smaller is first
        sorted_data, sorted_indices = torch.sort(variance, dim=0, descending=True)

        # calculate mask index here.
        masking = MyMasking(self.mask_ratio, sorted_indices)

        # mask_image_patches_batch(image, sorted_indices, './images/')

        with torch.no_grad():
            ref_cls_token = self.encoder(image)[0][0, :, :]			# torch.Size([B, 768])

        mask_cls_token = self.encoder(image, masking)[0][0, :, :]		# torch.Size([B, 768])

        loss = torch.mean((mask_cls_token - ref_cls_token) ** 2) / self.mask_ratio
        # [NOTE]: output image as dummy
        return loss, image, None
