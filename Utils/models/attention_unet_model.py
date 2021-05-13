# -*- coding: utf-8 -*-
"""
Code from https://github.com/LeeJunHyun/Image_Segmentation
@author: jpeeples
"""
import torch
import torch.nn as nn
from .unet_parts import *
import pdb

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi
    
class UpAtt(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, use_attention=False,
                 attention_mechanism=None):
        super().__init__()

        self.use_attention = use_attention
        self.attention_mechanism = attention_mechanism
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            if self.use_attention and self.attention_mechanism is None:
                self.conv = DoubleConv(in_channels // 2, out_channels // 2, in_channels // 2)
            else:
                self.conv = DoubleConv(in_channels, out_channels // 2, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, 
                                             kernel_size=2, stride=2)
            if self.use_attention and self.attention_mechanism is None:
                self.conv = DoubleConv(in_channels // 2, out_channels)
            else:
                self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        if self.use_attention:
            if self.attention_mechanism is not None:
                x2 = self.attention_mechanism(g=x1,x=x2)
                x = torch.cat([x2, x1], dim=1)
            else:
                x = x2*x1
        else:
            x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
    
class AttUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True,
                 feature_extraction = False, use_attention = True):
        super(AttUNet,self).__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.use_attention = use_attention
        factor = 2 if bilinear else 1
        
        #TBD Pretrained network
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
      
        #From other UNET model
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)
        
        self.up1 = UpAtt(1024, 512, bilinear, use_attention=self.use_attention,
                      attention_mechanism=self.Att5)
        self.up2 = UpAtt(512, 256, bilinear, use_attention=self.use_attention,
                      attention_mechanism=self.Att4)
        self.up3 = UpAtt(256, 128, bilinear, use_attention=self.use_attention,
                      attention_mechanism=self.Att3)
        self.up4 = UpAtt(128, 64 * factor, bilinear, use_attention=self.use_attention,
                      attention_mechanism=self.Att2)
        self.outc = OutConv(64,self.n_classes)


    def forward(self,x):
        # encoding path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # decoding + concat path
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return logits
    
