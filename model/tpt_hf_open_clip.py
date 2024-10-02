import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask

import diffdist.functional as diff_dist

from model.tpt import PromptLearner
from evaluator.imagenet_config import imagenet_classes

class TPTHFOpenCLIP(nn.Module):
    def __init__(self, image_encoder=None, image_projector=None, text_encoder=None, text_projector=None, clip=None):
        super().__init__()
        if not isinstance(clip, transformers.models.clip.modeling_clip.CLIPModel):
            raise TypeError
        self._image_encoder = image_encoder
        self._image_projector = image_projector
        self.text_encoder = text_encoder
        self.text_projector = text_projector
        self.clip = clip

        # [NOTE]: PromptLearner should be created in a factory. and then given as a argument here.
        n_ctx = 4
        ctx_init = 'a_photo_of_a'
        ctx_position = 'end' 
        learned_cls = False
        classnames = imagenet_classes
        # [NOTE]: Batch size should be None here
        batch_size = None 
        self.prompt_learner = PromptLearner(clip, classnames, batch_size,
                                            n_ctx, ctx_init, ctx_position, learned_cls)

        self.logit_scale = clip.logit_scale 
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

    @property
    def dtype(self):
        return self._image_encoder.patchify.weight.dtype

    # restore the initial state of the prompt_learner (tunable prompt)
    def reset(self):
        self.prompt_learner.reset()

    def reset_classnames(self, classnames, arch):
        self.prompt_learner.reset_classnames(classnames, arch)

    def image_encode(self, image):
        # Getting Image and Text Features
        image_features = self._image_encoder(image)[0][0, :, :]
        image_embeddings = self._image_projector(image_features)
        return image_embeddings 

    def text_encode(self, input_ids, attention_mask=None):
        text_x = self.clip.get_text_features(input_ids)
        return text_x

    def get_text_features(self):
        '''This function comes from https://github.com/azshue/TPT/blob/main/clip/custom_clip.py#L300-L308
        '''
        text_features = []
        prompts = self.prompt_learner()					# torch.Size([1000, 77, 512])
        tokenized_prompts = self.prompt_learner.tokenized_prompts	# torch.Size([1000, 77])
        t_features = self.text_encoder(prompts, tokenized_prompts)
        t_features = self.text_projector(t_features)

        text_features.append(t_features / t_features.norm(dim=-1, keepdim=True))
        text_features = torch.stack(text_features, dim=0)		# [1, 1000, 512]

        return torch.mean(text_features, dim=0)

    def forward(self, batch):
        # [TODO]: TPT CLIP change the behavior in train or test. but, TTA should be test
        #         should't be determined depending on the type of the input

        if isinstance(batch, dict):
            image_x = self.image_encode(batch['image'])
            text_x = self.text_encode(batch['input_ids'])
            loss, logit_scale = self.loss(image_x, text_x)
            return loss, logit_scale

        elif isinstance(batch, torch.Tensor):
            # with torch.no_grad():
            #     image_features = self.image_encode(batch.type(self.dtype))
            image_features = self.image_encode(batch.type(self.dtype))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = self.get_text_features()
            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()
            return logits

        else:
            raise TypeError



class TPTTextEncoder(nn.Module):
    '''
    This class is based on https://github.com/azshue/TPT/blob/main/clip/custom_clip.py#L39-L59
    '''
    def __init__(self, model, dtype):
        super().__init__()
        if not isinstance(model, transformers.models.clip.modeling_clip.CLIPTextTransformer):
            raise TypeError

        self.transformer = model.encoder
        self.positional_embedding = model.embeddings.position_embedding.weight
        self.ln_final = model.final_layer_norm
        self.dtype = dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)[:77, :] # x.shape torch.Size([1000, 77, 512])
        # [NOTE]: confirmed that x is the same as original TPT at this point.

        # [NOTE]: this process is refered as
        #         https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_clip.py#L702-L704
        causal_attention_mask = _create_4d_causal_attention_mask(x.shape[:2], self.dtype, device=x.device)
        x = self.transformer(x, causal_attention_mask=causal_attention_mask).last_hidden_state
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]

        return x


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

