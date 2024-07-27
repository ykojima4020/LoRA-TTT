import torch
import torchvision
import numpy as np

import sys
sys.path.append('../')
from evaluator.evaluator import ZeroShotEvaluator

from copy import deepcopy
from tqdm import tqdm
from misc.utils import AvgMeter, Summary, AverageMeter, ProgressMeter

import time

def build_tta_optimizer(model, config):
    #[NOTE]: all the trainable parameters is in an image encoder
    if config.optimizer == 'adam':
        optimizer = torch.optim.AdamW(model.image_encoder.parameters(),
            eps=config.eps, lr=config.lr, betas=config.betas,
            weight_decay=config.weight_decay)
    elif config.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.image_encoder.parameters(), lr=config.lr,
            weight_decay=config.weight_decay)
    else:
        raise TypeError
    return optimizer

class MEMLoss():
    def __init__(self, tpt=False):
        self._selection_p = 0.1
        self._tpt = tpt

    def __call__(self, model, images, text_embeddings):
        if self._tpt:
            output = model.clip(images)
        else:
            image_features = model.clip.image_encode(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            output = model.clip.logit_scale.exp() * (image_features @ text_embeddings)
        output, _ = select_confident_samples(output, self._selection_p)
        loss = avg_entropy(output)
        return loss


class MAELoss():
    def __init__(self):
        pass

    def __call__(self, model, images, text_embeddings):
        loss, reconstruction, mask = model.mae(images)
        return loss

class MAEMEMLoss():
    def __init__(self, mae_weight, mem_weight):
        self._maew = mae_weight
        self._memw = mem_weight
        self._mae_loss = MAELoss()
        self._mem_loss = MEMLoss()

    def __call__(self, model, images, text_embeddings):
        mem_loss = self._mem_loss(model, images, text_embeddings)
        mae_loss = self._mae_loss(model, images, text_embeddings)
        return (self._memw * mem_loss) + (self._maew * mae_loss)

class LoRATTARunner():

    def __init__(self, config, loss):
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

        self.tta_avg_time = None
        self.tta_infer_time = None

    def get_time(self):
        return {'time': {'average TTA': self.tta_avg_time, 'average inference': self.tta_infer_time}}

    def __call__(self, factory, status,
                 tta_dataset, prompts, classes,
                 num_workers=4, pin_memory=True, device='cuda'):
        model, tokenizer, _ = factory.create()
        model = model.to(device)

        # [NOTE]: trainable parameters
        for name, param in model.image_encoder.named_parameters():
            if 'lora' in name:
                param.requires_grad = True

        for name, param in model.named_parameters():
            print(f'{name}: {param.requires_grad}')

        text_embeddings = zeroshot_weights(model.clip, tokenizer, classes, prompts, device)

        optimizer = build_tta_optimizer(model, self._config)

        tta_data_loader = torch.utils.data.DataLoader(
                    tta_dataset, batch_size=1, shuffle=False,
                    num_workers=num_workers, pin_memory=pin_memory)

        top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
        top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)

        progress = ProgressMeter(
            len(tta_data_loader),
            [top1, top5],
            prefix='Test: ')

        tta_times = []
        infer_times = []

        for i, (images, target) in tqdm(enumerate(tta_data_loader)):

            for k in range(len(images)):
                images[k] = images[k].to(device)
            target = target.to(device)
            image = images[0]
            images = torch.cat(images, dim=0)

            torch.cuda.synchronize()
            start_tta = time.time()

            # [TODO]: should load only LoRA and Decoder, not update text_encoder
            model.mae.load_state_dict(status)
            model.train()
            for j in range(self._config.epochs):
                loss = self._loss(model, images, text_embeddings)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            torch.cuda.synchronize()
            end_tta = time.time()

            torch.cuda.synchronize()
            start_infer = time.time()

            # [NOTE]: inference
            model.eval()
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    image_features = model.clip.image_encode(image)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    output = image_features @ text_embeddings

            torch.cuda.synchronize()
            end_infer = time.time()
            tta_times.append(end_tta - start_tta)
            infer_times.append(end_infer - start_infer)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))

            if (i+1) % 200 == 0:
                progress.display(i)

        self.tta_avg_time = sum(tta_times) / len(tta_times)
        self.tta_infer_time = sum(infer_times) / len(infer_times)

        return top1.avg.item(), top5.avg.item()


class TextPromptTTARunner():

    def __init__(self, config, loss):
        print(f'{self} created.')
        print(config)
        self._config = config

        # [NOTE]: MEM loss only here.
        if isinstance(loss, MEMLoss):
            self._loss = loss
        else:
            raise TypeError

        self.tta_avg_time = None
        self.tta_infer_time = None

    def get_time(self):
        return {'time': {'average TTA': self.tta_avg_time, 'average inference': self.tta_infer_time}}


    def __call__(self, factory, status,
                 tta_dataset, prompts, classes,
                 num_workers=4, pin_memory=True, device='cuda'):
        model, tokenizer, _ = factory.create()
        model = model.to(device)

        # [NOTE]: trainable parameters
        model.clip.prompt_learner.ctx.requires_grad = True

        for name, param in model.named_parameters():
            print(f'{name}: {param.requires_grad}')

        # [NOTE]: fixed parameters for TPT
        arch = 'ViT-B/16'
        trainable_param = model.clip.prompt_learner.parameters()
        optimizer = torch.optim.AdamW(trainable_param, self._config.lr)
        model.clip.reset_classnames(classes, arch)
        model = model.to(device)

        tta_data_loader = torch.utils.data.DataLoader(
                    tta_dataset,
                    batch_size=1, shuffle=False,
                    num_workers=num_workers, pin_memory=pin_memory)

        top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
        top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)

        progress = ProgressMeter(
            len(tta_data_loader),
            [top1, top5],
            prefix='Test: ')

        model.eval()
        with torch.no_grad():
            model.clip.reset()

        tta_times = []
        infer_times = []

        for i, (images, target) in tqdm(enumerate(tta_data_loader)):

            for k in range(len(images)):
                images[k] = images[k].to(device)
            target = target.to(device)
            image = images[0]
            images = torch.cat(images, dim=0)

            torch.cuda.synchronize()
            start_tta = time.time()

            # reset the tunable prompt to its initial state
            with torch.no_grad():
                model.clip.reset()

            model.train()
            for j in range(self._config.epochs):
                loss = self._loss(model, images, text_embeddings=None)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            torch.cuda.synchronize()
            end_tta = time.time()

            torch.cuda.synchronize()
            start_infer = time.time()

            # [NOTE]: inference
            model.eval()
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    output = model.clip(image)

            torch.cuda.synchronize()
            end_infer = time.time()
            tta_times.append(end_tta - start_tta)
            infer_times.append(end_infer - start_infer)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))

            if (i+1) % 200 == 0:
                progress.display(i)

        self.tta_avg_time = sum(tta_times) / len(tta_times)
        self.tta_infer_time = sum(infer_times) / len(infer_times)

        return top1.avg.item(), top5.avg.item()


def test_time_tuning(clip, inputs, optimizer, scaler, config):

    # [NOTE]: fixed parameters for the reproduction
    selection_p = 0.1

    selected_idx = None
    for j in range(config.epochs):
        with torch.cuda.amp.autocast():
            clip = clip.to('cuda')
            output = clip(inputs)

            if selected_idx is not None:
                output = output[selected_idx]
            else:
                output, selected_idx = select_confident_samples(output, selection_p)

            loss = avg_entropy(output)

        optimizer.zero_grad()
        # compute gradient and do SGD step
        scaler.scale(loss).backward()
        del loss

        # Unscales the gradients of optimizer's assigned params in-place
        scaler.step(optimizer)
        scaler.update()
    return

def select_confident_samples(logits, top):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
    return logits[idx], idx

def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True) # logits = outputs.log_softmax(dim=1) [N, 1000]
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0]) # avg_logits = logits.mean(0) [1, 1000]
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k
       This function comes from https://github.com/azshue/TPT/blob/main/utils/tools.py#L88-L102
    """

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def zeroshot_weights(model, tokenizer, classnames, templates, device):
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

