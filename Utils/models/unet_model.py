""" Full assembly of the parts to form the complete network
Code from https://github.com/milesial/Pytorch-UNet
 """

import torch.nn.functional as F
import pdb
from torchvision import models

from .unet_parts import *


def set_parameter_requires_grad(model, feature_extraction):
    if feature_extraction:
        for param in model.parameters():
            param.requires_grad = False

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True,
                 feature_extraction = False, use_attention = False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.use_attention = use_attention
        factor = 2 if bilinear else 1

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)
            
        self.up1 = Up(1024, 512, bilinear, use_attention=self.use_attention)
        self.up2 = Up(512, 256, bilinear, use_attention=self.use_attention)
        self.up3 = Up(256, 128, bilinear, use_attention=self.use_attention)
        self.up4 = Up(128, 64 * factor, bilinear, use_attention=self.use_attention)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
