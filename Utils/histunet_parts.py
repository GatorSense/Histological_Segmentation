""" Parts of the HistUNet model """

#Pytorch dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F

##Local external libraries
from Utils.RBFHistogramPooling import HistogramLayerUNET

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    
class DownHist(nn.Module):
    """Downscaling with histogram then double conv"""

    def __init__(self, in_channels, out_channels,num_bins,
                 normalize_count=True,normalize_bins = True,use_hist=True):
        super().__init__()
        
        #Use histogramlayer, else stick to max pooling
        #Preprocess input to make sure histogram output is same 
        #number of channels as input for double conv
        if use_hist:
            self.pool_conv = nn.Sequential(
                nn.Conv2d(in_channels,int(in_channels/num_bins),1),
                HistogramLayerUNET(int(in_channels/num_bins),kernel_size=2,
                                   num_bins=num_bins,
                                   normalize_count=normalize_count,
                                   normalize_bins=normalize_bins),
                DoubleConv(in_channels, out_channels)
            )
        else:
            self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels // 2, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, 
                                         kernel_size=2, stride=2)
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
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class UpHist(nn.Module):
    """Upscaling then double conv with histogram layer concatenation"""

    def __init__(self, in_channels, out_channels, num_bins,bilinear=True,
                 normalize_count=True,normalize_bins = True,use_hist=True,
                 up4=False,use_attention=False,add_bn=False):
        super().__init__()

        self.use_attention = use_attention
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            if self.use_attention:
                self.conv = DoubleConv(in_channels // 2, out_channels // 2, in_channels // 2)
            else:
                self.conv = DoubleConv(in_channels, out_channels // 2, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, 
                                             kernel_size=2, stride=2)
            if self.use_attention:
                self.conv = DoubleConv(in_channels // 2, out_channels)
            else:
                self.conv = DoubleConv(in_channels, out_channels)

        #Define histogram layer for concatenation, use if wanted
        #For last upsampling layer, need to account for fewer number of channels
        #When using bilinear mode
        if use_hist:
            if up4 and bilinear:
                if add_bn:
                    self.hist_skip = nn.Sequential(nn.Conv2d(out_channels//2,
                                                    int(out_channels/(2*num_bins)),1),
                                                    nn.BatchNorm2d(int(out_channels/(2*num_bins))),
                                                    HistogramLayerUNET(int(out_channels/(2*num_bins)),
                                                    kernel_size=2,num_bins=num_bins,
                                                    normalize_count=normalize_count,
                                                    normalize_bins=normalize_bins,
                                                    skip_connection=True))
                else:
                    self.hist_skip = nn.Sequential(nn.Conv2d(out_channels//2,
                                                   int(out_channels/(2*num_bins)),1),
                                                   HistogramLayerUNET(int(out_channels/(2*num_bins)),
                                                   kernel_size=2,num_bins=num_bins,
                                                   normalize_count=normalize_count,
                                                   normalize_bins=normalize_bins,
                                                   skip_connection=True))
            else:
                if add_bn:
                    self.hist_skip = nn.Sequential(nn.Conv2d(out_channels,
                                                    int(out_channels/num_bins),1),
                                                    nn.BatchNorm2d(int(out_channels/num_bins)),
                                                    HistogramLayerUNET(int(out_channels/num_bins),
                                                    kernel_size=2,num_bins=num_bins,
                                                    normalize_count=normalize_count,
                                                    normalize_bins=normalize_bins,
                                                    skip_connection=True))
                else:
                    self.hist_skip = nn.Sequential(nn.Conv2d(out_channels,
                                                   int(out_channels/num_bins),1),
                                                   HistogramLayerUNET(int(out_channels/num_bins),
                                                   kernel_size=2,num_bins=num_bins,
                                                   normalize_count=normalize_count,
                                                   normalize_bins=normalize_bins,
                                                   skip_connection=True))                    
        else:
            self.hist_skip = nn.Sequential()
            
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
        
        #Pass feature maps from encoder branch through histogram layer 
        x2 = self.hist_skip(x2)
        
        if self.use_attention:
            x = x2*x1
        else:
            x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
