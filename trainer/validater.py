import torch
from tqdm import tqdm
from misc.utils import AvgMeter, get_lr
import wandb

class Validater():

    def __init__(self):
        pass

    def __call__(self):
        raise NotImplementedError

class SimpleValidater(Validater):

    def __init__(self, data_loader, optimizer, device):
        self._reset()
        self._data_loader = data_loader
        self._optimizer = optimizer
        self._device = device

    def _reset(self):
        self._loss_meter = AvgMeter()
        self._clip_loss_meter = AvgMeter()
        self._mae_loss_meter = AvgMeter()

    def __call__(self, model):
        self._reset()
        tqdm_object = tqdm(self._data_loader, total=len(self._data_loader))
        for batch in tqdm_object:
            batch = {k: v.to(self._device) for k, v in batch.items() if k != "caption"}
            with torch.no_grad():
                loss, clip_loss, mae_loss, reconstructed_img, logit_scale = model(batch)
            count = batch["image"].size(0)
            self._loss_meter.update(loss.item(), count)
            self._clip_loss_meter.update(clip_loss.item(), count)
            self._mae_loss_meter.update(mae_loss.item(), count)
    
            tqdm_object.set_postfix(valid_loss=self._loss_meter.avg)
        columns = ['original_image', 'reconstructed_image']
        table = wandb.Table(columns=columns)
        for sample, reconstruct in zip(batch['image'], reconstructed_img):
            table.add_data(wandb.Image(sample), wandb.Image(reconstruct))
        stats = {'valid': {'loss': self._loss_meter.avg,
                           'clip_loss': self._clip_loss_meter.avg,
                           'mae_loss': self._mae_loss_meter.avg},
                 'logit_scale': logit_scale}
        return stats, table
    
    
    