import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Net(nn.Module):
    def __init__(self, input_shape=[1, 28, 28], num_classes=10, bias=True, w=1, **kwargs):
        super().__init__()
        width = int(w * 512)
        self.flatten = nn.Flatten()
        self.input_dim = np.prod(input_shape)
        self.layer0 = nn.Linear(self.input_dim, width, bias=bias)
        self.relu1 = nn.ReLU(inplace=True)
        self.layer1 = nn.Linear(width, width, bias=bias)
        self.relu2 = nn.ReLU(inplace=True)
        self.layer2 = nn.Linear(width, width, bias=bias)
        self.relu3 = nn.ReLU(inplace=True)
        self.layer3 = nn.Linear(width, num_classes, bias=bias)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.layer0(x))
        x = self.relu2(self.layer1(x))
        x = self.relu3(self.layer2(x))
        x = self.layer3(x)

        return x
