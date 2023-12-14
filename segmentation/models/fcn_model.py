#
# Authors: Wouter Van Gansbeke
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

from functools import partial
import torch.nn as nn
from torch.nn import functional as F


class Model(nn.Module):
    def __init__(self, resnet, num_classes, dilation=6):
        super(Model, self).__init__()
        # Backbone is ResNet-50.
        # The 3 x 3 convolutions in the conv5 (layer 4) blocks have dilation 2 and stride 1. 
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        
        resnet.layer4.apply(partial(self._apply_dilation_2_stride_1))
        self.layer4 = resnet.layer4

        # Decoder is based upon FCN-16.
        # We use two 3 x 3 convolutions of 256 channels with BN and ReLU. 
        # This is followed by a 1 x 1 convolution for per-pixel classification.
        # We set dilation 6 in the two extra 3 x 3 convolutions.
        num_features = resnet.layer4[-1].conv1.in_channels
        print(resnet)
        print(num_features)
        self.decoder = nn.Sequential(
                            nn.Conv2d(num_features, 256, kernel_size=3, dilation=dilation, stride=1, padding=dilation, bias=False),
                            nn.BatchNorm2d(256), nn.ReLU(),
                            nn.Conv2d(256, 256, kernel_size=3, dilation=dilation, stride=1, padding=dilation, bias=False),
                            nn.BatchNorm2d(256), nn.ReLU(),
                            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=False))


    def _apply_dilation_2_stride_1(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.kernel_size == (3, 3):
                m.dilation = (2, 2)
                m.padding = (2, 2)
                m.stride = (1, 1)

            if m.stride == (2, 2):
                m.stride = (1, 1)

    def forward(self, x):
        input_shape = x.shape[-2:]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x) 
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.decoder(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        
        return x
