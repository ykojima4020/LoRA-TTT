import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.utils import ModelEmaV2

class GarmentClassifier(nn.Module):
    def __init__(self):
        super(GarmentClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        return x

model = GarmentClassifier()
ema = ModelEmaV2(model, decay=0.9999)
input = torch.randn([3,1,32,32])

print(model(input))

ema.update(model)

print(ema.module(input))

