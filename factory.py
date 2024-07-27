import open_clip
from transformers import DistilBertTokenizer
from transformers import CLIPModel, CLIPProcessor

from model.clip import CLIP
from model.modules import TextEncoder, ProjectionHead
from model.mae import ImageEncoder, MAEPixelDecoder, MAEFeatureDecoder, PixelMAE, FeatureMAE, FeatureMAEWithoutDecoder
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
        self._cfg = cfg
        self._image_size = 224 
        self._patch_size = 14
        self._emb_dim = 512
        self._encoder_layer = 12
        self._encoder_head = 4 
 
    def create(self):
        image_encoder = ImageEncoder(self._image_size, self._patch_size, self._emb_dim, self._encoder_layer, self._encoder_head)
        text_encoder = TextEncoder(self._cfg.text_encoder_name, pretrained=True, trainable=False)
        image_projection = ProjectionHead(embedding_dim=self._cfg.image_embedding, projection_dim=self._cfg.projection_dim, dropout=self._cfg.dropout)
        text_projection = ProjectionHead(embedding_dim=self._cfg.text_embedding, projection_dim=self._cfg.projection_dim, dropout=self._cfg.dropout)
        return CLIP(image_encoder, text_encoder, image_projection, text_projection, self._cfg.temperature)

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

        # [TODO]: this parameter should be managed in the config.
        self._mae = mae

    def create(self):

        open_clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='datacomp_l_s1b-b8k')

        image_encoder = OpenCLIPImageEncoder(open_clip_model)
        image_projector = OpenCLIPImageProjector(open_clip_model)
        clip = OpenCLIP(image_encoder, image_projector, open_clip_model, self._temperature)

        if self._mae == 'pixel': 
            image_decoder = MAEPixelDecoder(
                image_size=self._image_size, patch_size=self._patch_size,
                emb_dim=self._emb_dim, num_layer=self._decoder_layer,
                num_head=self._decoder_head)
            mae = PixelMAE(image_encoder, image_decoder, self._mask_ratio)
        elif self._mae == 'feature':
            image_decoder = MAEFeatureDecoder(
                image_size=self._image_size, patch_size=self._patch_size,
                emb_dim=self._emb_dim, num_layer=self._decoder_layer,
                num_head=self._decoder_head)
            mae = FeatureMAE(image_encoder, image_decoder, self._mask_ratio)
        else:
            raise TypeError(f'{self._mae} is invalid.')

        # [NOTE]: creating model...
        model = MAECLIP(image_encoder, clip, mae, alpha=self._alpha)
        tokenizer = open_clip.get_tokenizer('ViT-B-16')
        tokenizer = OpenCLIPTokenizer(tokenizer)
        transform = get_open_clip_vitb16_transforms

        return model, tokenizer, transform

class PretrainedOpenCLIPDecoderFineTuneFactory(Factory):
    def __init__(self, cfg, mae='pixel'):
        self._factory = PretrainedOpenCLIPFactory(cfg, mae)

    def create(self, frozen_encoder_layers=7):
        model, tokenizer, transform = self._factory.create()
        # [NOTE]: all the weigths are frozen
        for name, param in model.named_parameters():
            param.requires_grad = False

        for name, param in model.named_parameters():
            if ('decoder' in name):
                param.requires_grad = True

        return model, tokenizer, transform

class PretrainedOpenCLIPDecoderEncoderFineTuneFactory(Factory):
    def __init__(self, cfg, mae='pixel'):
        self._factory = PretrainedOpenCLIPFactory(cfg, mae)

    def create(self, frozen_encoder_layers=7):
        model, tokenizer, transform = self._factory.create()
        # [NOTE]: all the weigths are frozen
        for name, param in model.named_parameters():
            param.requires_grad = False

        # [TODO]: choose trainable parameters flexibly
        for name, param in model.named_parameters():
            if ('decoder' in name):
                param.requires_grad = True
            elif ('image_encoder' in name):
                if ('ln_post' in name):
                    param.requires_grad = True
                elif ('transformer' in name):
                    names = name.split('.')
                    layer_number = int(names[3])
                    if layer_number > frozen_encoder_layers:
                        param.requires_grad = True
                else:
                    param.requires_grad = False

        '''
        for name, param in model.named_parameters():
            if ('decoder' in name):
                param.requires_grad = True
            elif ('image_encoder' in name):
                if ('ln_pre' in name):
                    param.requires_grad = True
                elif ('transformer' in name):
                    names = name.split('.')
                    layer_number = int(names[3])
                    if layer_number < 4:
                        param.requires_grad = True
                else:
                    param.requires_grad = False
        '''

        return model, tokenizer, transform


class PretrainedHFOpenCLIPFactory(Factory):
    def __init__(self, cfg, tpt=False):
        # [NOTE]: need to create a decoder
        self._image_size = cfg.image.encoder.size
        self._patch_size = cfg.image.encoder.patch_size
        self._emb_dim = cfg.image.encoder.embeddings
        self._decoder_layer = cfg.image.decoder.layer
        self._decoder_head = cfg.image.decoder.head

        self._alpha = cfg.loss.alpha
        self._mask_ratio = cfg.mae.mask_ratio

        # [TODO]: this parameter should be managed in the config.
        self._mae_loss_type = cfg.mae.reconst
        self._mae_decoder = cfg.mae.decoder

        self._tpt = tpt
        self._peft = None
        if 'peft' in cfg:
            self._peft = cfg.peft

    def create(self, tpt=False):

        # [NOTE]: I don't know what kinds of dataset are used in the model.
        hf_open_clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

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
        else:
            raise TypeError(f'{self._peft.name} is not supported.')

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
                mae = FeatureMAE(image_encoder, image_decoder, self._mask_ratio)
            else:
                mae = FeatureMAEWithoutDecoder(image_encoder, None, self._mask_ratio)
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

        # [TODO]: should be set transform designed for
        transform = get_open_clip_vitb16_transforms
        return model, tokenizer, transform

