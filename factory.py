import torch
import torch.nn as nn

import open_clip
from transformers import DistilBertTokenizer
from transformers import CLIPModel, CLIPProcessor

from model.clip import CLIP
from model.modules import TextEncoder, ProjectionHead
from model.mae import ImageEncoder, MAEPixelDecoder, MAEFeatureDecoder, PixelMAE, CLSTokenMAE, CLSTokenMAEWithoutDecoder
from model.mae_clip import MAECLIP

from model.models_rils import RILSMAEEncoder
from model.open_clip import OpenCLIPImageEncoder, OpenCLIPImageProjector, OpenCLIP
from model.hf_open_clip import HFOpenCLIPImageEncoder, HFOpenCLIPImageProjector, HFOpenCLIP
from model.tpt_hf_open_clip import TPTHFOpenCLIP, TPTTextEncoder
from model.tokenizer import BertTokenizer, OpenCLIPTokenizer

from misc.transforms import get_original_vit_image_encoder_transforms, get_open_clip_vitb16_transforms

# [NOTE]: LoRA
from peft import LoraConfig, get_peft_model

class Factory:
    def __init__(self, cfg):
        pass

    def create(self):
        raise NotImplementedError

class OriginalViTCLIPFactory(Factory):

    def __init__(self, cfg):
        # [NOTE]: the following parameters are set by a configuration file.
        self._image_size = cfg.image.encoder.size
        self._patch_size = cfg.image.encoder.patch_size
        self._emb_dim = cfg.image.encoder.embeddings
        self._encoder_layer = cfg.image.encoder.layer
        self._encoder_head = cfg.image.encoder.head
        self._cfg = cfg
 
    def create(self):
        image_encoder = ImageEncoder(self._image_size, self._patch_size, self._emb_dim, self._encoder_layer, self._encoder_head)
        text_encoder = TextEncoder(self._cfg.text.encoder.name, pretrained=True, trainable=False)
        image_projection = ProjectionHead(embedding_dim=self._cfg.image.encoder.embeddings, projection_dim=self._cfg.clip.projection, dropout=self._cfg.clip.dropout)
        text_projection = ProjectionHead(embedding_dim=self._cfg.text.encoder.embeddings, projection_dim=self._cfg.clip.projection, dropout=self._cfg.clip.dropout)
        model = CLIP(image_encoder, text_encoder, image_projection, text_projection, self._cfg.clip.temperature)
        return model, None, None

class RILSMAECLIPFactory(Factory):

    def __init__(self, cfg):
        self._cfg = cfg

        self._image_size = cfg.image.encoder.size
        self._patch_size = cfg.image.encoder.patch_size

        self._emb_dim = cfg.image.encoder.embeddings
        self._encoder_layer = cfg.image.encoder.layer
        self._encoder_head = cfg.image.encoder.head 
        self._decoder_layer = cfg.image.decoder.layer 
        self._decoder_head = cfg.image.decoder.head 

        self._alpha = cfg.loss.alpha
        self._mask_ratio = cfg.mae.mask_ratio
        self._temperature = cfg.clip.temperature

    def create(self):
        image_encoder = RILSMAEEncoder(
            img_size=self._image_size, patch_size=self._patch_size,
            embed_dim=self._emb_dim, depth=self._encoder_layer,
            num_heads=self._encoder_head)

        image_decoder = MAEPixelDecoder(
            image_size=self._image_size, patch_size=self._patch_size,
            emb_dim=self._emb_dim, num_layer=self._decoder_layer,
            num_head=self._decoder_head)
 
        text_encoder = TextEncoder(self._cfg.text.encoder.name, pretrained=self._cfg.text.encoder.pretrained, trainable=self._cfg.text.encoder.trainable)
        image_projector = ProjectionHead(embedding_dim=self._emb_dim, projection_dim=self._cfg.clip.projection, dropout=self._cfg.clip.dropout)
        text_projector = ProjectionHead(embedding_dim=self._cfg.text.encoder.embeddings, projection_dim=self._cfg.clip.projection, dropout=self._cfg.clip.dropout)

        
        clip = CLIP(image_encoder, text_encoder, image_projector, text_projector, self._temperature)
        mae = PixelMAE(image_encoder, image_decoder, self._mask_ratio)

        model = MAECLIP(image_encoder, clip, mae, self._alpha)

        # [NOTE]: all the trainable parameters are requires_grad = False.
        for name, param in model.named_parameters():
            param.requires_grad = False

        tokenizer = DistilBertTokenizer.from_pretrained(self._cfg.text.encoder.name)
        tokenizer = BertTokenizer(tokenizer)

        transform = get_original_vit_image_encoder_transforms

        return model, tokenizer, transform


class PretrainedOpenCLIPFactory(Factory):
    def __init__(self, cfg, mae='pixel'):
        # [NOTE]: need to create a decoder
        self._image_size = cfg.image.encoder.size
        self._patch_size = cfg.image.encoder.patch_size
        self._emb_dim = cfg.image.encoder.embeddings
        self._decoder_layer = cfg.image.decoder.layer 
        self._decoder_head = cfg.image.decoder.head 

        self._alpha = cfg.loss.alpha
        self._mask_ratio = cfg.mae.mask_ratio
        self._temperature = cfg.clip.temperature

        self._mae_loss_type = cfg.mae.reconst
        self._mae_decoder = cfg.mae.decoder

        if 'peft' in cfg:
            raise NotImplementedError(f'PEFT cannot be applied in {self.__class__.__name__}.')

    def create(self):

        open_clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='datacomp_l_s1b-b8k')

        image_encoder = OpenCLIPImageEncoder(open_clip_model)
        image_projector = OpenCLIPImageProjector(open_clip_model)
        clip = OpenCLIP(image_encoder, image_projector, open_clip_model, self._temperature)

        if self._mae_loss_type == 'pixel':
            image_decoder = MAEPixelDecoder(
                image_size=self._image_size, patch_size=self._patch_size,
                emb_dim=self._emb_dim, num_layer=self._decoder_layer,
                num_head=self._decoder_head)
            mae = PixelMAE(image_encoder, image_decoder, self._mask_ratio)
        elif self._mae_loss_type == 'feature':
            image_decoder = MAEFeatureDecoder(
                image_size=self._image_size, patch_size=self._patch_size,
                emb_dim=self._emb_dim, num_layer=self._decoder_layer,
                num_head=self._decoder_head)
            if self._mae_decoder:
                mae = CLSTokenMAE(image_encoder, image_decoder, self._mask_ratio)
            else:
                mae = CLSTokenMAEWithoutDecoder(image_encoder, None, self._mask_ratio)
            print(f"{type(mae).__name__} is created.")
        else:
            raise TypeError(f'{self._mae_loss_type} is invalid.')

        # [NOTE]: creating model...
        model = MAECLIP(image_encoder, clip, mae, alpha=self._alpha)

        # [NOTE]: all the trainable parameters are requires_grad = False.
        for name, param in model.named_parameters():
            param.requires_grad = False

        tokenizer = open_clip.get_tokenizer('ViT-B-16')
        tokenizer = OpenCLIPTokenizer(tokenizer)
        transform = get_open_clip_vitb16_transforms

        return model, tokenizer, transform


class PretrainedHFOpenCLIPFactory(Factory):
    def __init__(self, cfg, tpt=False):
        # [NOTE]: need to create a decoder
        self._vit_type = cfg.image.encoder.name
        valid_type = ['vit-b', 'vit-l']
        if self._vit_type not in valid_type:
            raise ValueError(f'The value must be one of {valid_type}, but got {self._vit_type}')

        self._image_size = cfg.image.encoder.size
        self._patch_size = cfg.image.encoder.patch_size
        self._emb_dim = cfg.image.encoder.embeddings
        self._decoder_layer = cfg.image.decoder.layer
        self._decoder_head = cfg.image.decoder.head

        self._alpha = cfg.loss.alpha
        self._mask_ratio = cfg.mae.mask_ratio

        self._mae_loss_type = cfg.mae.reconst
        self._mae_decoder = cfg.mae.decoder

        self._tpt = tpt
        self._peft = None
        if 'peft' in cfg:
            self._peft = cfg.peft

    def create(self, tpt=False):

        # [NOTE]: I don't know what kinds of dataset are used in the model.
        if self._vit_type == 'vit-b':
            hf_open_clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        elif self._vit_type == 'vit-l':
            hf_open_clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        else:
            raise NotImplementedError

        image_encoder = HFOpenCLIPImageEncoder(hf_open_clip_model.vision_model)
        image_projector = HFOpenCLIPImageProjector(hf_open_clip_model.visual_projection)

        if self._peft.name == 'lora':
            config = LoraConfig(r=self._peft.r,
                                target_modules=self._peft.target_modules,
                                lora_alpha=(self._peft.r * self._peft.alpha_r_scale),
                                lora_dropout=self._peft.dropout,
                                layers_to_transform=self._peft.layers_to_transform
                               )
            image_encoder = get_peft_model(image_encoder, config)

            # [NOTE]: Trainable LoRA scaling
            '''
            for name, module in image_encoder.named_modules():
                if 'lora_B.default' in name:
                    scaled_module = DynamicScaledLinear(self._peft.r, self._emb_dim)
                    parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                    setattr(dict(image_encoder.named_modules())[parent_name], name.rsplit('.', 1)[-1], scaled_module)
            '''
        else:
            print(f'{self._peft.name} is not supported.')

        if self._tpt:
            text_encoder = TPTTextEncoder(hf_open_clip_model.text_model, hf_open_clip_model.dtype)
            # [NOTE]: ImageProjector is also used for TextProjector
            text_projector = HFOpenCLIPImageProjector(hf_open_clip_model.text_projection)
            clip = TPTHFOpenCLIP(image_encoder, image_projector, text_encoder, text_projector, hf_open_clip_model)
        else:
            clip = HFOpenCLIP(image_encoder, image_projector, hf_open_clip_model)


        if self._mae_loss_type == 'pixel':
            image_decoder = MAEPixelDecoder(
                image_size=self._image_size, patch_size=self._patch_size,
                emb_dim=self._emb_dim, num_layer=self._decoder_layer,
                num_head=self._decoder_head)
            mae = PixelMAE(image_encoder, image_decoder, self._mask_ratio)
        elif self._mae_loss_type == 'feature':
            image_decoder = MAEFeatureDecoder(
                image_size=self._image_size, patch_size=self._patch_size,
                emb_dim=self._emb_dim, num_layer=self._decoder_layer,
                num_head=self._decoder_head)
            if self._mae_decoder:
                mae = CLSTokenMAE(image_encoder, image_decoder, self._mask_ratio)
            else:
                mae = CLSTokenMAEWithoutDecoder(image_encoder, None, self._mask_ratio)
            print(f"{type(mae).__name__} is created.")
        else:
            raise TypeError(f'{self._mae_loss_type} is invalid.')

        # [NOTE]: creating model...
        model = MAECLIP(image_encoder, clip, mae, alpha=self._alpha)

        # [NOTE]: all the trainable parameters are requires_grad = False.
        for name, param in model.named_parameters():
            param.requires_grad = False

        tokenizer = processor.tokenizer
        tokenizer = BertTokenizer(tokenizer)

        # [NOTE]: the transform should be the same for ViT-B and ViT-L
        transform = get_open_clip_vitb16_transforms
        return model, tokenizer, transform


class ScaledLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=False):
        super(ScaledLinear, self).__init__(in_features, out_features, bias)
        # 学習可能なスケーリングパラメータを初期化
        self.scale = nn.Parameter(torch.zeros(1))
        min_value = -2
        max_value = 2
        self.scale.data.clamp_(min_value, max_value)

        # 重みとバイアスを0で初期化
        self.reset_parameters()

    def reset_parameters(self):
        # 重みを0で初期化
        nn.init.constant_(self.weight, 0)
        if self.bias is not None:
            # バイアスも0で初期化
            nn.init.constant_(self.bias, 0)

    def forward(self, input):
        # nn.Linear のフォワードパス
        output = super(ScaledLinear, self).forward(input)
        # スケーリングパラメータで出力をスケーリング
        return output * torch.exp(self.scale)


class DynamicScaledLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=False):
        super(DynamicScaledLinear, self).__init__(in_features, out_features, bias)

        self.scale = 1

        # 重みとバイアスを0で初期化
        self.reset_parameters()

    def reset_parameters(self):
        # 重みを0で初期化
        nn.init.constant_(self.weight, 0)
        if self.bias is not None:
            # バイアスも0で初期化
            nn.init.constant_(self.bias, 0)

    def set_scale(self, scale):
        self.scale = scale

    def forward(self, input):
        # nn.Linear のフォワードパス
        output = super(DynamicScaledLinear, self).forward(input)
        # スケーリングパラメータで出力をスケーリング
        return output * self.scale
