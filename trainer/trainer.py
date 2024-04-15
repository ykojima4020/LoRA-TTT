import torch
from torch import nn

from tqdm import tqdm
from misc.utils import AvgMeter, get_lr

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

    def __call__(self, model, epoch):
        self._reset()
        tqdm_object = tqdm(self._data_loader, total=len(self._data_loader))
        for idx, batch in enumerate(tqdm_object):
            self._optimizer.zero_grad()
            batch = {k: v.to(self._device) for k, v in batch.items() if k != "caption"}

            with torch.autocast(device_type='cuda'):
                loss, clip_loss, mae_loss, _, _ = model(batch)
            self._scaler.scale(loss).backward()

            self._scaler.unscale_(self._optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), self._grad_norm)
            self._scaler.step(self._optimizer)
            self._scaler.update()
            self._lr_scheduler.step_update(epoch * self._num_steps + idx)

            count = batch["image"].size(0)
            self._loss_meter.update(loss.item(), count)
            self._clip_loss_meter.update(clip_loss.item(), count)
            self._mae_loss_meter.update(mae_loss.item(), count)
    
            tqdm_object.set_postfix(train_loss=self._loss_meter.avg, lr=get_lr(self._optimizer))
        lr = self._optimizer.param_groups[0]["lr"]
        stats = {'train': {'loss': self._loss_meter.avg,
                           'clip_loss': self._clip_loss_meter.avg,
                           'mae_loss': self._mae_loss_meter.avg},
                 'lr': lr}
        return stats