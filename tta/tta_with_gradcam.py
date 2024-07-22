import torch
import torchvision
import numpy as np
from pathlib import Path

import sys
sys.path.append('../')

from copy import deepcopy
from tqdm import tqdm
from misc.utils import AvgMeter, Summary, AverageMeter, ProgressMeter

from tta import MAELoss, MEMLoss, MAEMEMLoss, zeroshot_weights, accuracy, build_tta_optimizer
from gradcam import clip_grad_cam

class LoRATTARunnerWithGradCAM():

    def __init__(self, config, loss, save_dir=False):
        print(f'{self} created.')
        print(config)
        self._config = config

        if isinstance(loss, MEMLoss):
            self._loss = loss
        elif isinstance(loss, MAELoss):
            self._loss = loss
        elif isinstance(loss, MAEMEMLoss):
            self._loss = loss
        else:
            raise TypeError

        if save_dir:
            self.save_dir = Path(save_dir)
            self.result_file = self.save_dir / 'result.txt'
            print(self.save_dir)
        else:
            self.save_dir = False

    def _lora_trainable(self, model):
        for name, param in model.image_encoder.named_parameters():
            if 'lora' in name:
                param.requires_grad = True

    def __call__(self, factory, status,
                 tta_dataset, prompts, classes, device='cuda'):
        model, tokenizer, _ = factory.create()
        model = model.to(device)

        # [NOTE]: trainable parameters
        self._lora_trainable(model)

        for name, param in model.named_parameters():
            print(f'{name}: {param.requires_grad}')

        text_embeddings = zeroshot_weights(model.clip, tokenizer, classes, prompts, device)

        optimizer = build_tta_optimizer(model, self._config)

        top1_before_tta = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
        top5_before_tta = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
        top1_after_tta = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
        top5_after_tta = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)

        progress = ProgressMeter(
            len(tta_dataset),
            [top1_before_tta, top1_after_tta],
            prefix='Test: ')

        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            if self.result_file.exists():
                self.result_file.unlink()
            self.result_file.touch()
            before_save_dir = self.save_dir / 'before'
            after_save_dir = self.save_dir / 'after'
            before_save_dir.mkdir(parents=True, exist_ok=True)
            after_save_dir.mkdir(parents=True, exist_ok=True)

        for i, (images, target, path) in tqdm(enumerate(tta_dataset)):

            if self.save_dir:
                p = Path(path)
                file_name = p.name
                cls = p.parent.parts[-1]
                before_cls_save_dir = before_save_dir / cls
                after_cls_save_dir = after_save_dir / cls
                before_cls_save_dir.mkdir(parents=True, exist_ok=True)
                after_cls_save_dir.mkdir(parents=True, exist_ok=True)

            for k in range(len(images)):
                images[k] = images[k].to(device)
            target = torch.tensor([target]).to(device)
            image = images[0].unsqueeze(0)
            images = torch.stack(images)

            # [TODO]: should load only LoRA and Decoder, not update text_encoder
            model.mae.load_state_dict(status)

            # [NOTE]: inference before TTA
            model.eval()
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    image_features = model.clip.image_encode(image)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    output = image_features @ text_embeddings
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1_before_tta.update(acc1[0], image.size(0))
            top5_before_tta.update(acc5[0], image.size(0))
            before_res = True if acc1 == 100.0 else False

            if self.save_dir:
                before_image = clip_grad_cam(model, image, target[0], text_embeddings)
                before_image.save(before_save_dir / cls / file_name)

            self._lora_trainable(model)
            model.train()
            for j in range(self._config.epochs):
                loss = self._loss(model, images, text_embeddings)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # [NOTE]: inference after TTA
            model.eval()
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    image_features = model.clip.image_encode(image)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    output = image_features @ text_embeddings
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1_after_tta.update(acc1[0], image.size(0))
            top5_after_tta.update(acc5[0], image.size(0))
            after_res = True if acc1 == 100.0 else False

            if self.save_dir:
                after_image = clip_grad_cam(model, image, target[0], text_embeddings)
                after_image.save(after_save_dir / cls / file_name)

            with open(self.result_file, 'a') as f:
                f.write(f"{cls + '/' + file_name},{before_res},{after_res}\n")

            if (i+1) % 200 == 0:
                progress.display(i)

        return top1_before_tta.avg.item(), top1_after_tta.avg.item()

