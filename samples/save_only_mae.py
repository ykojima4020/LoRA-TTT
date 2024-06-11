import torch

import sys
sys.path.append('../')
from factory import PretrainedTPTHFOpenCLIPFactory

from misc.config import load_config
from omegaconf import OmegaConf



checkpoint = 'load/model/path' 
config = '/load/config/path' 
device = 'cuda'

cfg = load_config(config)
OmegaConf.set_struct(cfg, True)

factory = PretrainedTPTHFOpenCLIPFactory(cfg.model, mae=cfg.reconst)
model, _, _ = factory.create()
model = model.to(device)

status = torch.load(checkpoint, map_location=device)
model.load_state_dict(status['model'])

print(model.mae)

save_state = {
    'model': model.mae.state_dict(),
}
torch.save(save_state, 'save/model/path')
 