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
    def __init__(self, selection_p=0.1):
        self._selection_p = selection_p

    def __call__(self, model, images, text_embeddings):
        # [NOTE]: confidence selection
        if self._selection_p != 1:
            with torch.no_grad():
                image_features = model.clip.image_encode(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            output = model.clip.logit_scale.exp() * (image_features @ text_embeddings)
            _, selected_idx = select_confident_samples(output, self._selection_p)
            images = images[selected_idx]

        loss, reconstruction, mask = model.mae(images)
        return loss


class WeightedMAELoss():
    def __init__(self, selection_p=0.1):
        self._selection_p = selection_p

    def __call__(self, model, images, text_embeddings):
        # [NOTE]: confidence selection
        with torch.no_grad():
            image_features = model.clip.image_encode(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        output = model.clip.logit_scale.exp() * (image_features @ text_embeddings)
        selected_logits, selected_idx = select_confident_samples(output, self._selection_p)
        images = images[selected_idx]

        # [NOTE]: weight calculation
        coefficient, _ = weighted_entropy(selected_logits)
        loss, _, _ = model.mae(images, coefficient=coefficient)

        return loss


from model.mae import PatchShuffle
import torch.nn.functional as F
class MAEConsistencyLoss():
    def __init__(self, selection_p=0.1):
        self._selection_p = selection_p
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
    def __init__(self, mae_weight, mem_weight, selection_p=0.1):
        self._maew = mae_weight
        self._memw = mem_weight
        self._mae_loss = MAELoss(selection_p=selection_p)
        self._mem_loss = MEMLoss(selection_p=selection_p)

    def __call__(self, model, images, text_embeddings):
        mem_loss = self._mem_loss(model, images, text_embeddings)
        mae_loss = self._mae_loss(model, images, text_embeddings)
        return (self._memw * mem_loss) + (self._maew * mae_loss)


class MAEMEMLossV2():
    def __init__(self, mae_weight, mem_weight, selection_p=0.1):
        self._maew = mae_weight
        self._memw = mem_weight
        self._selection_p = selection_p

    def __call__(self, model, images, text_embeddings):

        image_features = model.clip.image_encode(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        output = model.clip.logit_scale.exp() * (image_features @ text_embeddings)

        loss_output, selected_idx = select_confident_samples(output, self._selection_p)
        mem_loss = avg_entropy(loss_output)

        images = images[selected_idx]
        mae_loss, reconstruction, mask = model.mae(images)

        loss = (self._memw * mem_loss) + (self._maew * mae_loss)

        return loss


from misc.seed_util import g
def seed_worker(worker_id):
    worker_seed = g.initial_seed() + worker_id
    torch.manual_seed(worker_seed)

class TTARunner():

    def __init__(self, tta_handler):
        # [NOTE]: tta_handler includes Image Encoder Tuning, LoRA, TPT, and both.
        self.tta = tta_handler

    def __call__(self, tta_dataset, classes, prompts, num_workers=4, pin_memory=True, device='cuda'):

        self.tta.reset_dataset(classes, prompts)

        # [NOTE]: TTA preparation
        tta_data_loader = torch.utils.data.DataLoader(
                    tta_dataset, batch_size=1, shuffle=False, worker_init_fn=seed_worker, generator=g,
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

            self.tta.reset_model()
            loss = self.tta.update(images)
            acc1, acc5, _, _ = self.tta.accuracy(image, target)

            # measure accuracy and record loss
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))

            if (i+1) % 200 == 0:
                progress.display(i)

        return top1.avg.item(), top5.avg.item()


class TTARunnerAnalyser():

    def __init__(self, tta_handler, file=None):
        # [NOTE]: tta_handler includes Image Encoder Tuning, LoRA, TPT, and both.
        self.tta = tta_handler

    def __call__(self, tta_dataset, classes, prompts, dname=None, num_workers=4, pin_memory=True, device='cuda'):

        if dname:
            if isinstance(self.tta.get_loss(), MEMLoss):
                loss_type = 'mem'
            elif isinstance(self.tta.get_loss(), MAELoss):
                loss_type = 'mae'
            else:
                raise ValueError
            self.file_path = Path(f'./calib/{dname}_{loss_type}.txt')
        else:
            self.file_path = Path(f'analysis.txt')

        if not self.file_path.exists():
            self.file_path.touch()
        else:
            raise NotImplementedError('File already exists. No new file created.')


        self.tta.reset_dataset(classes, prompts)

        # [NOTE]: TTA preparation
        tta_data_loader = torch.utils.data.DataLoader(
                    tta_dataset, batch_size=1, shuffle=False, worker_init_fn=seed_worker, generator=g,
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

            self.tta.reset_model()
            acc1, acc5, score_before_tta, score_max_before_tta = self.tta.accuracy(image, target, score=True)
            res_before_tta = True if acc1 == 100.0 else False
            loss = self.tta.update(images)
            acc1, acc5, score_after_tta, score_max_after_tta = self.tta.accuracy(image, target, score=True)
            res_after_tta = True if acc1 == 100.0 else False

            # measure accuracy and record loss
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))

            with open(self.file_path, 'a') as f:
                f.write(f'{loss.item()} {res_before_tta} {res_after_tta} {score_before_tta.item()} {score_max_before_tta.item()} {score_after_tta.item()} {score_max_after_tta.item()}\n')

            if (i+1) % 200 == 0:
                progress.display(i)

        return top1.avg.item(), top5.avg.item()


class ParallelTTARunner():

    def __init__(self, tta_handler_1, tta_handler_2):
        # [NOTE]: 2 tta handelers are only supported.
        self.tta_1 = tta_handler_1
        self.tta_2 = tta_handler_2

    def __call__(self, tta_dataset, classes, prompts, num_workers=4, pin_memory=True, device='cuda'):

        self.tta_1.reset_dataset(classes, prompts)
        self.tta_2.reset_dataset(classes, prompts)

        # [NOTE]: TTA preparation
        tta_data_loader = torch.utils.data.DataLoader(
                    tta_dataset, batch_size=1, shuffle=False, worker_init_fn=seed_worker, generator=g,
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

            self.tta_1.reset_model()
            self.tta_2.reset_model()

            self.tta_2.set_freeze()
            self.tta_1.set_trainable()
            self.tta_1.update(images)

            self.tta_1.set_freeze()
            self.tta_2.set_trainable()
            self.tta_2.update(images)

            # [NOTE]: only tta 1 for accuracy
            acc1, acc5, _, _ = self.tta_1.accuracy(image, target)

            # measure accuracy and record loss
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))

            if (i+1) % 200 == 0:
                progress.display(i)

        return top1.avg.item(), top5.avg.item()



class TTAHandlerIF():

    def __init__(self, loss):
        # [NOTE]: When using MAELoss, due to specific implementation constraints,
        #         enabling AMP prevents the loss from ebing differentiable.
        #         https://discuss.pytorch.org/t/autocast-and-torch-no-grad-unexpected-behaviour/93475
        self.amp = False

        print(f"{type(loss).__name__} is used.")
        if isinstance(loss, MEMLoss):
            self.loss = loss
            self.amp = True
        elif isinstance(loss, MAELoss):
            self.loss = loss
        elif isinstance(loss, MAEMEMLoss):
            self.amp = True
            self.loss = loss
        elif isinstance(loss, MAEMEMLoss):
            self.amp = True
            self.loss = loss
        elif isinstance(loss, MAEMEMLossV2):
            self.amp = True
            self.loss = loss
        else:
            raise TypeError

    def update(self, images):
        raise NotImplementedError

    def get_loss(self):
        return self.loss

class ImageEncoderTTA(TTAHandlerIF):

    def __init__(self, model, tokenizer, status, loss, config, device='cuda', lora=True):
        super().__init__(loss)
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.status = status
        self.config = config
        self.device = device
        self.lora = lora
        self.text_embeddings = None

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

        for name, param in model.named_parameters():
            print(f'{name}: {param.requires_grad}')
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Trainable parameters: {trainable_params}')

    def set_trainable(self):
        for name, param in self.model.named_parameters():
            param.requires_grad = self.requires_grad_states[name]

    def set_freeze(self):
        for name, param in self.model.named_parameters():
            if self.requires_grad_states[name]: # 元々 True だったものだけ変更
                param.requires_grad = False

    def reset_dataset(self, classes, prompts, device='cuda'):
        self.text_embeddings = zeroshot_weights(self.model.clip, self.tokenizer, classes, prompts, device)

    def reset_model(self):
        self.model.mae.load_state_dict(self.status)

    def reset_optim(self):
        self.optimizer.load_state_dict(self.optim_state)

    def update(self, images):
        self.model.train()
        for j in range(self.config.epochs):
            if self.config.reset:
                self.reset_optim()
            with torch.autocast(device_type='cuda', enabled=self.amp):
                loss = self.loss(self.model, images, self.text_embeddings)
            self.optimizer.zero_grad()
            # compute gradient and do SGD step
            self.scaler.scale(loss).backward()
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


class TextPromptTTA(TTAHandlerIF):

    def __init__(self, model, tokenizer, status, loss, config, device='cuda'):
        super().__init__(loss)
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.status = status
        self.config = config
        self.device = device
        self.text_embeddings = None

        # [NOTE]: TPT
        model.clip.prompt_learner.ctx.requires_grad = True
        self.requires_grad_states = {name: param.requires_grad for name, param in self.model.named_parameters()}
        trainable_param = model.clip.prompt_learner.parameters()
        self.optimizer = build_tta_optimizer(trainable_param, self.config)
        self.optim_state = deepcopy(self.optimizer.state_dict())

        # setup automatic mixed-precision (Amp) loss scaling
        self.scaler = torch.GradScaler(init_scale=1000)

        for name, param in model.named_parameters():
            print(f'{name}: {param.requires_grad}')
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Trainable parameters: {trainable_params}')

    def set_trainable(self):
        for name, param in self.model.named_parameters():
            param.requires_grad = self.requires_grad_states[name]

    def set_freeze(self):
        for name, param in self.model.named_parameters():
            if self.requires_grad_states[name]: # 元々 True だったものだけ変更
                param.requires_grad = False

    def reset_dataset(self, classes, prompts, device='cuda'):
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
            with torch.autocast(device_type='cuda', enabled=self.amp):
                loss = self.loss(self.model, images, self.text_embeddings)
            self.optimizer.zero_grad()
            # compute gradient and do SGD step
            self.scaler.scale(loss).backward()
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

def weighted_entropy(logits, scaling_factor=0.4):
    log_probs = logits - logits.logsumexp(dim=-1, keepdim=True)			# torch.Size([N, C])
    min_val = torch.finfo(log_probs.dtype).min
    log_probs = torch.clamp(log_probs, min=min_val)
    probs = torch.exp(log_probs)
    entropy = -torch.sum(probs * log_probs, dim=1)

    # prevent overflow
    max_entropy = 88.0
    entropy = torch.clamp(entropy, max=max_entropy)
    coefficient = 1 / torch.exp(entropy.clone().detach() - scaling_factor)
    return coefficient, entropy

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
