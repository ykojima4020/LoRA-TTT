import torch
from tqdm import tqdm
from misc.utils import AvgMeter

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
