import torch
import torchvision
import numpy as np

import sys
sys.path.append('../')
from evaluator.evaluator import ZeroShotEvaluator
from tta import TTAHandlerIF, build_tta_optimizer, zeroshot_weights, accuracy 

from copy import deepcopy
from tqdm import tqdm
from misc.utils import AvgMeter, Summary, AverageMeter, ProgressMeter

from pathlib import Path

def measure_memory_usage():
    return torch.cuda.memory_allocated() / (1024 ** 3)


class ImageEncoderTTAMeasurer(TTAHandlerIF):

    def __init__(self, factory, status, loss, config, device='cuda', lora=True):
        super().__init__(loss)
        self.before_model_mem = AverageMeter('before model load', ':2.3f', Summary.AVERAGE)
        self.after_model_mem = AverageMeter('after model load', ':2.3f', Summary.AVERAGE)
        self.before_forward_mem = AverageMeter('before forward', ':2.3f', Summary.AVERAGE)
        self.after_forward_mem = AverageMeter('after forward', ':2.3f', Summary.AVERAGE)
        self.before_backward_mem = AverageMeter('before backward', ':2.3f', Summary.AVERAGE)
        self.after_backward_mem = AverageMeter('after backward', ':2.3f', Summary.AVERAGE)

        self.factory = factory
        self.status = status
        self.config = config
        self.device = device
        self.lora = lora
        self.text_embeddings = None


    def set_trainable(self):
        for name, param in self.model.named_parameters():
            param.requires_grad = self.requires_grad_states[name]

    def set_freeze(self):
        for name, param in self.model.named_parameters():
            if self.requires_grad_states[name]: # 元々 True だったものだけ変更
                param.requires_grad = False

    def reset_dataset(self, classes, prompts, device='cuda'):

        self.before_model_mem.update(measure_memory_usage(), 1)       
        self.model, self.tokenizer, _ = self.factory.create()
        self.model = self.model.to(device)
        self.text_embeddings = zeroshot_weights(self.model.clip, self.tokenizer, classes, prompts, device)
        del self.model.clip.clip.text_model
        del self. model.clip.clip.text_projection
        self.after_model_mem.update(measure_memory_usage(), 1)       

        if self.lora:
            for name, param in self.model.image_encoder.named_parameters():
                if 'lora' in name:
                    param.requires_grad = True
        else:
            # model.clip._image_projector.proj.weight.requires_grad = True
            for i, layer in enumerate(self.model.image_encoder.transformer.layers):
                if i in self.config.layers:
                    for name, param in layer.named_parameters():
                        if not 'lora' in name:
                            if 'self_attn' in name:
                                param.requires_grad = True
        self.requires_grad_states = {name: param.requires_grad for name, param in self.model.named_parameters()}

        self.optimizer = build_tta_optimizer(self.model.image_encoder.parameters(), self.config)
        self.optim_state = deepcopy(self.optimizer.state_dict())
        # setup automatic mixed-precision (Amp) loss scaling
        self.scaler = torch.GradScaler(init_scale=1000)

        for name, param in self.model.named_parameters():
            print(f'{name}: {param.requires_grad}')
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'Trainable parameters: {trainable_params}')


    def reset_model(self):
        self.model.mae.load_state_dict(self.status)

    def reset_optim(self):
        self.optimizer.load_state_dict(self.optim_state)

    def update(self, images):
        self.model.train()
        for j in range(self.config.epochs):
            if self.config.reset:
                self.reset_optim()
            self.before_forward_mem.update(measure_memory_usage(), 1)       
            with torch.autocast(device_type='cuda', enabled=self.amp):
                loss = self.loss(self.model, images, self.text_embeddings)
            self.after_forward_mem.update(measure_memory_usage(), 1)       
            self.optimizer.zero_grad()
            self.before_backward_mem.update(measure_memory_usage(), 1)       
            # compute gradient and do SGD step
            self.scaler.scale(loss).backward()
            self.after_backward_mem.update(measure_memory_usage(), 1)       
            # Unscales the gradients of optimizer's assigned params in-place
            self.scaler.step(self.optimizer)
            self.scaler.update()
        return loss

    def accuracy(self, image, target, score=False):
        # [NOTE]: inference
        self.model.eval()
        with torch.no_grad():
            with torch.autocast(device_type='cuda', enabled=False):
                image_features = self.model.clip.image_encode(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                output = image_features @ self.text_embeddings
                if score:
                    scores = (self.model.clip.logit_scale.exp() * output).softmax(dim=-1) # logit_scale.exp() is 100
                    target_score = scores[0][target[0]]
                    max_score = torch.max(scores[0])
                else:
                    target_score = None
                    max_score = None
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        return acc1, acc5, target_score, max_score

    def __del__(self):
        print(self.before_model_mem)
        print(self.after_model_mem)
        print(self.before_forward_mem)
        print(self.after_forward_mem)
        print(self.before_backward_mem)
        print(self.after_backward_mem)


class TextPromptTTAMeasurer(TTAHandlerIF):

    def __init__(self, factory, status, loss, config, device='cuda'):
        super().__init__(loss)

        self.before_model_mem = AverageMeter('before model load', ':2.3f', Summary.AVERAGE)
        self.after_model_mem = AverageMeter('after model load', ':2.3f', Summary.AVERAGE)
        self.before_forward_mem = AverageMeter('before forward', ':2.3f', Summary.AVERAGE)
        self.after_forward_mem = AverageMeter('after forward', ':2.3f', Summary.AVERAGE)
        self.before_backward_mem = AverageMeter('before backward', ':2.3f', Summary.AVERAGE)
        self.after_backward_mem = AverageMeter('after backward', ':2.3f', Summary.AVERAGE)

        self.factory = factory
        self.status = status
        self.config = config
        self.device = device
        self.text_embeddings = None

    def set_trainable(self):
        for name, param in self.model.named_parameters():
            param.requires_grad = self.requires_grad_states[name]

    def set_freeze(self):
        for name, param in self.model.named_parameters():
            if self.requires_grad_states[name]: # 元々 True だったものだけ変更
                param.requires_grad = False

    def reset_dataset(self, classes, prompts, device='cuda'):
        self.before_model_mem.update(measure_memory_usage(), 1)       
        self.model, self.tokenizer, _ = self.factory.create()
        self.model = self.model.to(device)
        self.after_model_mem.update(measure_memory_usage(), 1)       

        # [NOTE]: TPT
        self.model.clip.prompt_learner.ctx.requires_grad = True
        self.requires_grad_states = {name: param.requires_grad for name, param in self.model.named_parameters()}
        trainable_param = self.model.clip.prompt_learner.parameters()
        self.optimizer = build_tta_optimizer(trainable_param, self.config)
        self.optim_state = deepcopy(self.optimizer.state_dict())

        # setup automatic mixed-precision (Amp) loss scaling
        self.scaler = torch.GradScaler(init_scale=1000)

        for name, param in self.model.named_parameters():
            print(f'{name}: {param.requires_grad}')
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'Trainable parameters: {trainable_params}')

        arch = 'ViT-B/16'
        self.model.clip.reset_classnames(classes, arch)
        self.text_embeddings = None

        self.model = self.model.to(device)
        self.model.eval()
        with torch.no_grad():
            self.model.clip.reset()

    def reset_model(self):
        with torch.no_grad():
            self.model.clip.reset()

    def reset_optim(self):
        self.optimizer.load_state_dict(self.optim_state)

    def update(self, images):
        self.model.train()
        for j in range(self.config.epochs):
            if self.config.reset:
                self.reset_optim()
            self.before_forward_mem.update(measure_memory_usage(), 1)       
            with torch.autocast(device_type='cuda', enabled=self.amp):
                loss = self.loss(self.model, images, self.text_embeddings)
            self.after_forward_mem.update(measure_memory_usage(), 1)       
            self.optimizer.zero_grad()
            self.before_backward_mem.update(measure_memory_usage(), 1)       
            # compute gradient and do SGD step
            self.scaler.scale(loss).backward()
            self.after_backward_mem.update(measure_memory_usage(), 1)       
            # Unscales the gradients of optimizer's assigned params in-place
            self.scaler.step(self.optimizer)
            self.scaler.update()

    def accuracy(self, image, target, score=False):
        # [NOTE]: inference
        self.model.eval()
        with torch.no_grad():
            with torch.autocast(device_type='cuda', enabled=False):
                output = self.model.clip(image)
                if score:
                    scores = (self.model.clip.logit_scale.exp() * output).softmax(dim=-1) # logit_scale.exp() is 100
                    target_score = scores[0][target[0]]
                    max_score = torch.max(scores[0])
                else:
                    target_score = None
                    max_score = None
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        return acc1, acc5, target_score, max_score

    def __del__(self):
        print(self.before_model_mem)
        print(self.after_model_mem)
        print(self.before_forward_mem)
        print(self.after_forward_mem)
        print(self.before_backward_mem)
        print(self.after_backward_mem)
