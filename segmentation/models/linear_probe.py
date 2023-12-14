#
# Authors: Wouter Van Gansbeke
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.deeplabv3_model import ResnetDilated


class LinearProbe(nn.Module):
    def __init__(self, resnet, num_classes):
        super(LinearProbe, self).__init__()
        num_features = resnet.layer4[-1].conv1.in_channels
        self.backbone = ResnetDilated(resnet) # Resnet is Dilated
        self.head = nn.Conv2d(num_features, num_classes, kernel_size=1)

    def forward(self, x):
        out_size = x.size()[2:]
        with torch.no_grad():
            x = self.backbone(x)
        x = self.head(x)
        return F.interpolate(x, out_size, mode='bilinear', align_corners=False).squeeze()

    def parameters(self):
        return self.head.parameters()
