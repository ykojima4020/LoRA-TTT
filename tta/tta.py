import torch
import torchvision
import numpy as np

import sys
sys.path.append('../')
from evaluator.evaluator import ZeroShotEvaluator

from copy import deepcopy
from tqdm import tqdm
from misc.utils import AvgMeter, Summary, AverageMeter, ProgressMeter

from pathlib import Path

def build_tta_optimizer(param, config):
    #[NOTE]: all the trainable parameters is in an image encoder
    if config.optimizer == 'adam':
        optimizer = torch.optim.AdamW(param,
            eps=config.eps, lr=config.lr, betas=config.betas,
            weight_decay=config.weight_decay)
    elif config.optimizer == 'sgd':
        optimizer = torch.optim.SGD(param, lr=config.lr,
            weight_decay=config.weight_decay)
    else:
        raise TypeError
    return optimizer

class MEMLoss():
    def __init__(self, tpt=False, selection_p=0.1):
        self._selection_p = selection_p
        self._tpt = tpt

    def __call__(self, model, images, text_embeddings):
        if self._tpt:
            output = model.clip(images)
        else:
            image_features = model.clip.image_encode(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            output = model.clip.logit_scale.exp() * (image_features @ text_embeddings)
        loss_output, _ = select_confident_samples(output, self._selection_p)
        loss = avg_entropy(loss_output)
        return loss

class MAELoss():
    def __init__(self):
        pass

    def __call__(self, model, images, text_embeddings):
        loss, reconstruction, mask = model.mae(images)
        return loss

class SelectionMAELoss():
    def __init__(self):
        self._selection_p = 0.1

    def __call__(self, model, images, text_embeddings):
        with torch.no_grad():
            image_features = model.clip.image_encode(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        output = model.clip.logit_scale.exp() * (image_features @ text_embeddings)
        _, selected_idx = select_confident_samples(output, self._selection_p)
        images = images[selected_idx]
        loss, reconstruction, mask = model.mae(images)
        return loss

from model.mae import PatchShuffle
import torch.nn.functional as F
class SelectionMAEConsistencyLoss():
    def __init__(self):
        self._selection_p = 0.1
        mask_ratio = 0.5
        self._shuffler = PatchShuffle(mask_ratio)

    def __call__(self, model, images, text_embeddings):
        with torch.no_grad():
            image_features = model.clip.image_encode(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        sims = model.clip.logit_scale.exp() * (image_features @ text_embeddings)
        sims, selected_idx = select_confident_samples(sims, self._selection_p)
        images = images[selected_idx]

        mask_image_features = model.clip.image_encode(images, self._shuffler)
        mask_image_features = mask_image_features / mask_image_features.norm(dim=-1, keepdim=True)
        mask_sims = model.clip.logit_scale.exp() * (mask_image_features @ text_embeddings)

        prob = F.softmax(sims, dim=1)
        mask_prob = F.softmax(mask_sims, dim=1)

        log_mask_prob = torch.log(mask_prob)
        loss = -torch.sum(prob * log_mask_prob, dim=1)
        loss = torch.mean(loss)
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

class SelectionMAEMEMLoss():
    def __init__(self, mae_weight, mem_weight):
        self._maew = mae_weight
        self._memw = mem_weight
        self._mae_loss = SelectionMAELoss()
        self._mem_loss = MEMLoss()

    def __call__(self, model, images, text_embeddings):
        mem_loss = self._mem_loss(model, images, text_embeddings)
        mae_loss = self._mae_loss(model, images, text_embeddings)
        return (self._memw * mem_loss) + (self._maew * mae_loss)


class TTARunnerIF():

    def __init__(self, config, loss, tp=False, lora=True):
        print(f'{self} created.')
        print(config)
        self._config = config

        # [NOTE]: When using MAELoss, due to specific implementation constraints,
        #         enabling AMP prevents the loss from ebing differentiable.
        #         https://discuss.pytorch.org/t/autocast-and-torch-no-grad-unexpected-behaviour/93475
        self._amp = False

        print(f"{type(loss).__name__} is used.")
        if isinstance(loss, MEMLoss):
            self._loss = loss
            self._amp = True
        elif isinstance(loss, MAELoss):
            self._loss = loss
        elif isinstance(loss, SelectionMAELoss):
            self._loss = loss
        elif isinstance(loss, SelectionMAEConsistencyLoss):
            self._loss = loss
        elif isinstance(loss, MAEMEMLoss):
            self._amp = True
            self._loss = loss
        elif isinstance(loss, SelectionMAEMEMLoss):
            self._amp = True
            self._loss = loss
        else:
            raise TypeError

        self._tp = tp
        self._lora = lora

    def __call__(self, factory, status,
                 tta_dataset, prompts, classes,
                 num_workers=4, pin_memory=True, device='cuda'):
        raise NotImplementedError

class TTARunner(TTARunnerIF):

    def __init__(self, config, loss, tp=False, lora=True):
        super().__init__(config, loss, tp=tp, lora=lora)

    def __call__(self, factory, status,
                 tta_dataset, prompts, classes,
                 num_workers=4, pin_memory=True, device='cuda'):
        model, tokenizer, _ = factory.create()
        model = model.to(device)

        # [NOTE]: Set trainable parameters

        # [NOTE]: TPT
        if self._tp:
            model.clip.prompt_learner.ctx.requires_grad = True
            arch = 'ViT-B/16'
            trainable_param = model.clip.prompt_learner.parameters()
            optimizer = build_tta_optimizer(trainable_param, self._config)
            optim_state = deepcopy(optimizer.state_dict())
            model.clip.reset_classnames(classes, arch)
            model = model.to(device)
            model.eval()
            with torch.no_grad():
                model.clip.reset()
            text_embeddings = None

        # [NOTE]: Image Encoder Tuning
        else:
            if self._lora:
                for name, param in model.image_encoder.named_parameters():
                    if 'lora' in name:
                        param.requires_grad = True
            else:
                # model.clip._image_projector.proj.weight.requires_grad = True
                for i, layer in enumerate(model.image_encoder.transformer.layers):
                    if i in self._config.layers:
                        for name, param in layer.named_parameters():
                            if not 'lora' in name:
                                if 'self_attn' in name:
                                    param.requires_grad = True
            text_embeddings = zeroshot_weights(model.clip, tokenizer, classes, prompts, device)
            optimizer = build_tta_optimizer(model.image_encoder.parameters(), self._config)
            optim_state = deepcopy(optimizer.state_dict())

        for name, param in model.named_parameters():
            print(f'{name}: {param.requires_grad}')
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Trainable parameters: {trainable_params}')

        # setup automatic mixed-precision (Amp) loss scaling
        scaler = torch.GradScaler(init_scale=1000)

        tta_data_loader = torch.utils.data.DataLoader(
                    tta_dataset, batch_size=1, shuffle=True,
                    num_workers=num_workers, pin_memory=pin_memory)

        top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
        top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)

        progress = ProgressMeter(
            len(tta_data_loader),
            [top1, top5],
            prefix='Test: ')

        for i, (images, target) in tqdm(enumerate(tta_data_loader)):

            for k in range(len(images)):
                images[k] = images[k].to(device)
            target = target.to(device)
            image = images[0]
            images = torch.cat(images, dim=0)

            if self._tp:
                # reset the tunable prompt to its initial state
                with torch.no_grad():
                    model.clip.reset()
            else:
                # [TODO]: should load only LoRA and Decoder, not update text_encoder
                model.mae.load_state_dict(status)

            model.train()
            for j in range(self._config.epochs):
                if self._config.reset:
                    optimizer.load_state_dict(optim_state)
                with torch.autocast(device_type='cuda', enabled=self._amp):
                    loss = self._loss(model, images, text_embeddings)
                optimizer.zero_grad()
                # compute gradient and do SGD step
                scaler.scale(loss).backward()
                # Unscales the gradients of optimizer's assigned params in-place
                scaler.step(optimizer)
                scaler.update()

            if self._config.adaptive:
                set_scale(model.image_encoder, loss.item())

            # [NOTE]: inference
            model.eval()
            with torch.no_grad():
                with torch.autocast(device_type='cuda'):
                    if self._tp:
                        output = model.clip(image)
                    else:
                        image_features = model.clip.image_encode(image)
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                        output = image_features @ text_embeddings

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))

            if (i+1) % 200 == 0:
                progress.display(i)

        return top1.avg.item(), top5.avg.item()


class TTARunnerAnalyser(TTARunnerIF):

    def __init__(self, config, loss, tp=False, lora=True, file=None):
        super().__init__(config, loss, tp=tp, lora=lora)

        if file:
            self.file_path = Path(file)
        else:
            self.file_path = Path(f'analysis.txt')
        if not self.file_path.exists():
            self.file_path.touch()
        else:
            raise NotImplementedError('File already exists. No new file created.')

    def __call__(self, factory, status,
                 tta_dataset, prompts, classes,
                 num_workers=4, pin_memory=True, device='cuda'):
        model, tokenizer, _ = factory.create()
        model = model.to(device)

        # [NOTE]: Set trainable parameters

        # [NOTE]: TPT
        if self._tp:
            model.clip.prompt_learner.ctx.requires_grad = True
            arch = 'ViT-B/16'
            trainable_param = model.clip.prompt_learner.parameters()
            optimizer = build_tta_optimizer(trainable_param, self._config)
            optim_state = deepcopy(optimizer.state_dict())
            model.clip.reset_classnames(classes, arch)
            model = model.to(device)
            model.eval()
            with torch.no_grad():
                model.clip.reset()
            text_embeddings = None


        # [NOTE]: Image Encoder Tuning
        else:
            if self._lora:
                for name, param in model.image_encoder.named_parameters():
                    if 'lora' in name:
                        param.requires_grad = True
            else:
                # model.clip._image_projector.proj.weight.requires_grad = True
                for i, layer in enumerate(model.image_encoder.transformer.layers):
                    if i in self._config.layers:
                        for name, param in layer.named_parameters():
                            if not 'lora' in name:
                                if 'self_attn' in name:
                                    param.requires_grad = True
            text_embeddings = zeroshot_weights(model.clip, tokenizer, classes, prompts, device)
            optimizer = build_tta_optimizer(model.image_encoder.parameters(), self._config)
            optim_state = deepcopy(optimizer.state_dict())

        for name, param in model.named_parameters():
            print(f'{name}: {param.requires_grad}')
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Trainable parameters: {trainable_params}')

        # setup automatic mixed-precision (Amp) loss scaling
        scaler = torch.GradScaler(init_scale=1000)

        # [NOTE]: Dataset is not shuffled in the Analysis class
        tta_data_loader = torch.utils.data.DataLoader(
                    tta_dataset, batch_size=1, shuffle=False,
                    num_workers=num_workers, pin_memory=pin_memory)

        top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
        top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)

        progress = ProgressMeter(
            len(tta_data_loader),
            [top1, top5],
            prefix='Test: ')

        for i, (images, target) in tqdm(enumerate(tta_data_loader)):

            for k in range(len(images)):
                images[k] = images[k].to(device)
            target = target.to(device)
            image = images[0]
            images = torch.cat(images, dim=0)

            if self._tp:
                # reset the tunable prompt to its initial state
                with torch.no_grad():
                    model.clip.reset()
            else:
                # [TODO]: should load only LoRA and Decoder, not update text_encoder
                model.mae.load_state_dict(status)

            # [NOTE]: inference
            model.eval()
            with torch.no_grad():
                with torch.autocast(device_type='cuda'):
                    image_features = model.clip.image_encode(image)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    output = image_features @ text_embeddings
                    scores = (100 * output).softmax(dim=-1) # logit_scale.exp() is 100
                    socore_before_tta = scores[0][target[0]]

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            res_before_tta = True if acc1 == 100.0 else False


            model.train()
            for j in range(self._config.epochs):
                if self._config.reset:
                    optimizer.load_state_dict(optim_state)
                with torch.autocast(device_type='cuda', enabled=self._amp):
                    loss = self._loss(model, images, text_embeddings)
                optimizer.zero_grad()
                # compute gradient and do SGD step
                scaler.scale(loss).backward()
                # Unscales the gradients of optimizer's assigned params in-place
                scaler.step(optimizer)
                scaler.update()

            if self._config.adaptive:
                set_scale(model.image_encoder, loss.item())

            # [NOTE]: inference
            model.eval()
            with torch.no_grad():
                with torch.autocast(device_type='cuda'):
                    if self._tp:
                        output = model.clip(image)
                    else:
                        image_features = model.clip.image_encode(image)
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                        output = image_features @ text_embeddings
                    scores = (100 * output).softmax(dim=-1) # logit_scale.exp() is 100
                    socore_after_tta = scores[0][target[0]]


            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            res_after_tta = True if acc1 == 100.0 else False

            if (i+1) % 200 == 0:
                progress.display(i)

            with open(self.file_path, 'a') as f:
                f.write(f'{loss.item()} {res_before_tta} {res_after_tta} {socore_before_tta.item()} {socore_after_tta.item()}\n')

        return top1.avg.item(), top5.avg.item()




def test_time_tuning(clip, inputs, optimizer, scaler, config):

    # [NOTE]: fixed parameters for the reproduction
    selection_p = 0.1

    selected_idx = None
    for j in range(config.epochs):
        with torch.autocast():
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
            max_length = 77
            tokens = tokenizer(texts, padding=True, truncation=True, max_length=max_length)
            batch = {key: values.to(device) for key, values in tokens.items()}
            class_embeddings = model.text_encode(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True) # the norm shape is torch.Size([80, 1])
            class_embedding = class_embeddings.mean(dim=0) # the mean shape is torch.Size([256])
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights


def set_scale(model, scale):
    # scale = min(scale, 1)
    for name, module in model.named_modules():
        if 'lora_B.default' in name:
            module.set_scale(scale)
