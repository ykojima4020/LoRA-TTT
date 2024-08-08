import torch
from torch import nn

from tqdm import tqdm
from misc.utils import AvgMeter, get_lr


class Loss():
    def __init__(self, clip_weight=1, mae_weight=1):
        self._clipw = clip_weight
        self._maew = mae_weight

    def __call__(self, model, batch):
        if self._clipw and self._maew:
            clip_loss, logit_scale = model.module.clip(batch)
            mae_loss, reconstruction, mask = model.module.mae(batch['image'])
            loss = (self._clipw*clip_loss) + (self._maew*mae_loss)
        elif self._clipw:
            clip_loss, logit_scale = model.module.clip(batch)
            loss = clip_loss
            mae_loss = torch.tensor(0, requires_grad=False)
        elif self._maew:
            mae_loss, reconstruction, mask = model.module.mae(batch['image'])
            loss = mae_loss
            clip_loss = torch.tensor(0, requires_grad=False)
        else:
            raise NotImplementedError

        return loss, clip_loss, mae_loss

class Trainer():

    def __init__(self):
        pass

    def __call__(self):
        raise NotImplementedError

class SimpleTrainer(Trainer):

    def __init__(self, data_loader, optimizer, lr_scheduler, grad_norm, device):
        self._reset()
        self._data_loader = data_loader
        self._num_steps = len(data_loader)
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler
        self._device = device

        self._scaler = torch.cuda.amp.GradScaler()
        self._grad_norm = grad_norm

    def _reset(self):
        self._loss_meter = AvgMeter()
        self._clip_loss_meter = AvgMeter()
        self._mae_loss_meter = AvgMeter()

    def __call__(self, model, loss, epoch):
        self._reset()
        tqdm_object = tqdm(self._data_loader, total=len(self._data_loader))
        for idx, batch in enumerate(tqdm_object):
            self._optimizer.zero_grad()
            batch = {k: v.to(self._device) for k, v in batch.items() if k != "caption"}

            with torch.autocast(enabled=True, device_type='cuda'):
                total_loss, clip_loss, mae_loss = loss(model, batch)
            self._scaler.scale(total_loss).backward()
            self._scaler.unscale_(self._optimizer)

            nn.utils.clip_grad_norm_(model.parameters(), self._grad_norm)
            self._scaler.step(self._optimizer)
            self._scaler.update()
            self._lr_scheduler.step_update(epoch * self._num_steps + idx)

            count = batch["image"].size(0)
            self._loss_meter.update(total_loss.item(), count)
            self._clip_loss_meter.update(clip_loss.item(), count)
            self._mae_loss_meter.update(mae_loss.item(), count)
    
            tqdm_object.set_postfix(train_loss=self._loss_meter.avg, lr=get_lr(self._optimizer))
        lr = self._optimizer.param_groups[0]["lr"]
        stats = {'train': {'loss': self._loss_meter.avg,
                           'clip_loss': self._clip_loss_meter.avg,
                           'mae_loss': self._mae_loss_meter.avg},
                 'lr': lr}
        return stats