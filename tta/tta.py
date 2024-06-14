import torch
import torchvision
import numpy as np

import sys
sys.path.append('../')
from evaluator.evaluator import ZeroShotImageNetEvaluator

from copy import deepcopy
from tqdm import tqdm
from misc.utils import AvgMeter, Summary, AverageMeter, ProgressMeter

class TPTMAETTARunner():

    def __init__(self, mae=True, tpt=False):
        self._mae = mae
        self._tpt = tpt

    def __call__(self, factory, status,
                 tta_dataset, prompts, classes, config,
                 num_workers=4, pin_memory=True, device='cuda'):
        model, tokenizer, _ = factory.create()
        model = model.to(device)

        # [NOTE]: trainable parameters
        if self._tpt:
            model.clip.prompt_learner.ctx.requires_grad = True
        if self._mae:
            for name, param in model.image_encoder.named_parameters():
                if 'lora' in name:
                    param.requires_grad = True

        for name, param in model.named_parameters():
            print(f'{name}: {param.requires_grad}')


        if self._tpt:
            # [NOTE]: fixed parameters for TPT
            arch = 'ViT-B/16'
            trainable_param = model.clip.prompt_learner.parameters()
            optimizer = torch.optim.AdamW(trainable_param, config.tpt.lr)
            optim_state = deepcopy(optimizer.state_dict())
            model.clip.reset_classnames(classes, arch)

            # setup automatic mixed-precision (Amp) loss scaling
            scaler = torch.cuda.amp.GradScaler(init_scale=1000)

        if self._mae:
            text_embeddings = zeroshot_weights(model.clip, tokenizer, classes, prompts, device)
            # [NOTE]: MAE optimizer, update only image encoder
            if config.mae.optimizer == 'adam':
                eps = 1e-8
                mae_optimizer = torch.optim.AdamW(model.image_encoder.parameters(),
                        eps=eps, lr=config.mae.lr, betas=(0.9, 0.95), weight_decay=config.mae.weight_decay)
            elif config.mae.optimizer == 'sgd':
                mae_optimizer = torch.optim.SGD(model.image_encoder.parameters(), lr=config.mae.lr, weight_decay=config.mae.weight_decay)
            else:
                raise TypeError


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

        for i, (images, target) in tqdm(enumerate(tta_data_loader)):

            for k in range(len(images)):
                images[k] = images[k].to(device)
            target = target.to(device)
            image = images[0]
            images = torch.cat(images, dim=0)

            if self._tpt:
                # reset the tunable prompt to its initial state
                if config.tpt.epochs > 0:
                    with torch.no_grad():
                        model.clip.reset()

                # [NOTE]: I don't know why optimizer is loaded here.
                optimizer.load_state_dict(optim_state)
                test_time_tuning(model.clip, images, optimizer, scaler, config.tpt)

            # [NOTE]: MAE TTA
            if self._mae:
                # [TODO]: should load only LoRA and Decoder, not update text_encoder
                model.mae.load_state_dict(status)
                for j in range(config.mae.epochs):
                    loss, reconstruction, mask = model.mae(images)
                    mae_optimizer.zero_grad()
                    loss.backward()
                    mae_optimizer.step()


            # [NOTE]: inference
            model.eval()
            model.clip = model.clip.to(device) # why?
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    if self._tpt:
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


class MEMLoRATTARunner():

    def __init__(self, mae=True, tpt=False):
        self._mae = mae
        self._tpt = tpt

    def __call__(self, factory, status,
                 tta_dataset, prompts, classes, config,
                 num_workers=4, pin_memory=True, device='cuda'):
        model, tokenizer, _ = factory.create()
        model = model.to(device)

        # [NOTE]: trainable parameters
        if self._tpt:
            model.clip.prompt_learner.ctx.requires_grad = True
        if self._mae:
            for name, param in model.image_encoder.named_parameters():
                if 'lora' in name:
                    param.requires_grad = True

        for name, param in model.named_parameters():
            print(f'{name}: {param.requires_grad}')


        if self._tpt:
            # [NOTE]: fixed parameters for TPT
            arch = 'ViT-B/16'
            trainable_param = model.clip.prompt_learner.parameters()
            optimizer = torch.optim.AdamW(trainable_param, config.tpt.lr)
            optim_state = deepcopy(optimizer.state_dict())
            model.clip.reset_classnames(classes, arch)

            # setup automatic mixed-precision (Amp) loss scaling
            scaler = torch.cuda.amp.GradScaler(init_scale=1000)

        if self._mae:
            text_embeddings = zeroshot_weights(model.clip, tokenizer, classes, prompts, device)
            # [NOTE]: MAE optimizer, update only image encoder
            if config.mae.optimizer == 'adam':
                eps = 1e-8
                mae_optimizer = torch.optim.AdamW(model.image_encoder.parameters(),
                        eps=eps, lr=config.mae.lr, betas=(0.9, 0.95), weight_decay=config.mae.weight_decay)
            elif config.mae.optimizer == 'sgd':
                mae_optimizer = torch.optim.SGD(model.image_encoder.parameters(), lr=config.mae.lr, weight_decay=config.mae.weight_decay)
            else:
                raise TypeError


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

        selection_p = 0.1

        model.eval()
        with torch.no_grad():
            model.clip.reset()

        for i, (images, target) in tqdm(enumerate(tta_data_loader)):

            for k in range(len(images)):
                images[k] = images[k].to(device)
            target = target.to(device)
            image = images[0]
            images = torch.cat(images, dim=0)

            if self._tpt:
                # reset the tunable prompt to its initial state
                if config.tpt.epochs > 0:
                    with torch.no_grad():
                        model.clip.reset()

                # [NOTE]: I don't know why optimizer is loaded here.
                optimizer.load_state_dict(optim_state)
                test_time_tuning(model.clip, images, optimizer, scaler, config.tpt)

            # [NOTE]: MAE TTA
            if self._mae:

                # [TODO]: should load only LoRA and Decoder, not update text_encoder
                model.mae.load_state_dict(status)
                for j in range(config.mae.epochs):
                    if self._tpt:
                        output = model.clip(image)
                    else:
                        image_features = model.clip.image_encode(images)
                        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                        output = model.clip.logit_scale.exp() * (image_features @ text_embeddings)
                    output, _ = select_confident_samples(output, selection_p)
                    loss = avg_entropy(output)
                    mae_optimizer.zero_grad()
                    loss.backward()
                    mae_optimizer.step()

 
            # [NOTE]: inference
            model.eval()
            model.clip = model.clip.to(device) # why?
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    if self._tpt:
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

class TTARunner():

    def __init__(self, single):
        if single:
            self._ttadapter = SingleSampleAdapter()
        else:
            self._ttadapter = AllSampleAdapter()

    def __call__(self, factory, status,
                 tta_train_data, tta_test_data, prompts, classes, config,
                 num_workers=4, pin_memory=True, device='cuda'):
        model, tokenizer, _ = factory.create()
        model = model.to(device)

        # [NOTE]: Preparation for TTA
        for name, param in model.named_parameters():
            if ('decoder' in name):
                param.requires_grad = False
            if ('text_model.encoder' in name):
                param.requires_grad = False

        for name, param in model.image_encoder.named_parameters():
            if 'lora' in name:
                param.requires_grad = True


        # [NOTE]: update only image encoder
        if config.optimizer == 'adam':
            eps = 1e-8 
            optimizer = torch.optim.AdamW(model.image_encoder.parameters(),
                    eps=eps, lr=config.lr, betas=(0.9, 0.95), weight_decay=config.weight_decay)
        elif config.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.image_encoder.parameters(), lr=config.lr, weight_decay=config.weight_decay) 
        else:
            raise TypeError

        # [NOTE]: STEP1: Evaluation of initial model before TTT.
        evaluator = ZeroShotImageNetEvaluator(tokenizer, tta_test_data, prompts, classes, device)
        before_tta = evaluator(model.clip)
        before_tta_top1 = before_tta['eval']['imagenet']['top1']
        before_tta_top5 = before_tta['eval']['imagenet']['top5']
    
        after_tta_top1, after_tta_top5 = self._ttadapter(model, status, tokenizer, optimizer,
                                                         tta_train_data, tta_test_data, prompts, classes, config,
                                                         num_workers, pin_memory, device) 
        return before_tta_top1, before_tta_top5, after_tta_top1, after_tta_top5
    
class AllSampleAdapter():

    def __init__(self):
        pass

    def __call__(self, model, status, tokenizer, optimizer,
                 tta_train_data, tta_test_data, prompts, classes, config,
                 num_workers=4, pin_memory=True, device='cuda'):
        # [NOTE]: initialization
        evaluator = ZeroShotImageNetEvaluator(tokenizer, tta_test_data, prompts, classes, device)

        train_loader = torch.utils.data.DataLoader(tta_train_data,
                           batch_size=config.batch_size, num_workers=num_workers, pin_memory=pin_memory)
        tttrainer = TestTimeTrainer(train_loader, optimizer, device)

        # [NOTE]: after culculation original zero-shot performance, load the finetuned weights.
        model.load_state_dict(status)

        # [NOTE]: STEP2: TTT
        for epoch in range(0, config.epochs):
            tttrainer(model.mae)

        # [NOTE]: STEP3: Evaluation of model after TTT.
        after_tta = evaluator(model.clip, update=False)
        after_tta_top1 = after_tta['eval']['imagenet']['top1']
        after_tta_top5 = after_tta['eval']['imagenet']['top5']
        return after_tta_top1, after_tta_top5

class SingleSampleAdapter():

    def __init__(self):
        pass

    def __call__(self, model, status, tokenizer, optimizer,
                 tta_train_data, tta_test_data, prompts, classes, config,
                 num_workers=4, pin_memory=True, device='cuda'):
 
        steps_per_example = config.epochs
        train_loader = iter(torch.utils.data.DataLoader(TTTTrainDataset(tta_train_data, steps_per_example, config.batch_size),
                                                        batch_size=config.batch_size, shuffle=False,
                                                        num_workers=num_workers, pin_memory=pin_memory))

        test_loader = torch.utils.data.DataLoader(tta_test_data, batch_size=1, shuffle=False,
                                                  num_workers=num_workers, pin_memory=pin_memory)

        model.load_state_dict(status)
        # [NOTE]: because the text encoder is not updated, text embeddings can be calculated at first.
        text_embeddings = zeroshot_weights(model.clip, tokenizer, classes, prompts, device)

        # top1, top5, n = 0., 0., 0.
        top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
        top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)


        for test_image, target in tqdm(test_loader):
            test_image = test_image.to(device)
            target = target.to(device)
            model.load_state_dict(status)
            model.train()
            # [NOTE]: optimizer should be initialized? => don't have to initialize because lr is constant.

            for step_per_example in range(steps_per_example):
                train_data = next(train_loader)
                train_image, _ = train_data
                train_image = train_image.to(device)
                loss, reconstruction, mask = model.mae(train_image) 
                loss.backward()
                optimizer.step()
                optimizer.zero_grad() 

            model.eval()
            with torch.no_grad():
                # predict
                image_features = model.clip.image_encode(test_image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                logits = image_features @ text_embeddings

            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1.update(acc1[0], test_image.size(0))
            top5.update(acc5[0], test_image.size(0))

        return top1.avg.item(), top5.avg.item()


class TestTimeTrainer():

    def __init__(self, data_loader, optimizer, device):
        self._data_loader = data_loader
        self._optimizer = optimizer
        self._device = device
        self._mae_loss_meter = AvgMeter()

    def __call__(self, model):
        model = model.train()
        tqdm_object = tqdm(self._data_loader, total=len(self._data_loader))
        for idx, (images, target) in enumerate(tqdm_object):
            images = images.to(self._device)
            loss, reconstruction, mask = model(images)
            loss.backward()
            self._optimizer.step()
            self._optimizer.zero_grad()
            count = images.size(0)
            self._mae_loss_meter.update(loss.item(), count)
        return self._mae_loss_meter.avg


class TTTTrainDataset():
    '''
    this dataset classs is based on https://github.com/yossigandelsman/test_time_training_mae/blob/main/data/tt_image_folder.py#L7
    '''

    def __init__(self, dataset, steps_per_example, batch_size):
        if not isinstance(dataset, torch.utils.data.Dataset):
            raise TypeError
        self.dataset = dataset
        self.batch_size = batch_size
        self.steps_per_example = steps_per_example

    def __len__(self):
        return self.batch_size * self.steps_per_example * len(self.dataset) 

    def __getitem__(self, index):
        real_index = (index // (self.steps_per_example * self.batch_size))
        image, target = self.dataset[real_index]
        return image, target


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

