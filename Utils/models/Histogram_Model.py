## PyTorch dependencies
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import torch

## Local external libraries
from .JOSHUA_parts import *

import pdb

def set_parameter_requires_grad(model, feature_extraction):
    if feature_extraction:
        for param in model.parameters():
            param.requires_grad = False

class JOSHUA(nn.Module):
    
    def __init__(self,n_channels,n_classes,skip=True,pool=True,bilinear=True,
                 num_bins=4,normalize_count=True,normalize_bins=True,
                 skip_locations=[True,True,True,True],
                 pool_locations=[True,True,True,True],use_attention=False,
                 feature_extraction = False, add_bn=True,analyze=False):
        
        #inherit nn.module
        super(JOSHUA,self).__init__()
        self.skip_locations = skip_locations
        self.pool_locations = pool_locations
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.analyze = analyze
     
        factor = 2 if self.bilinear else 1

    
        self.inc = DoubleConv(n_channels, 64)
        
        #Add histogram pooling in encoder branch if desired
        if pool: 
            self.down1 = DownHist(64,128,num_bins,
                                  normalize_count=normalize_count,
                                  normalize_bins=normalize_bins,
                                  use_hist=pool_locations[0])
            self.down2 = DownHist(128,256,num_bins,
                                  normalize_count=normalize_count,
                                  normalize_bins=normalize_bins,
                                  use_hist=pool_locations[1])
            self.down3 = DownHist(256,512,num_bins,
                                  normalize_count=normalize_count,
                                  normalize_bins=normalize_bins,
                                  use_hist=pool_locations[2])
            self.down4 = DownHist(512, 1024 // factor,num_bins,
                                  normalize_count=normalize_count,
                                  normalize_bins=normalize_bins,
                                  use_hist=pool_locations[3])
            
        else:
            self.down1 = Down(64, 128)
            self.down2 = Down(128, 256)
            self.down3 = Down(256, 512)
            self.down4 = Down(512, 1024 // factor)

        
        #Add histogram layer skip connection
        if skip: 
            self.up1 = UpHist(1024, 512, num_bins, bilinear=bilinear,
                              normalize_count=normalize_count, 
                              normalize_bins=normalize_bins,
                              use_hist=skip_locations[0],
                              use_attention=use_attention,
                              add_bn=add_bn)
            self.up2 = UpHist(512, 256, num_bins, bilinear=bilinear,
                              normalize_count=normalize_count, 
                              normalize_bins=normalize_bins,
                              use_hist=skip_locations[1],
                              use_attention=use_attention,
                              add_bn=add_bn)
            self.up3 = UpHist(256, 128, num_bins, bilinear=bilinear,
                              normalize_count=normalize_count, 
                              normalize_bins=normalize_bins,
                              use_hist=skip_locations[2],
                              use_attention=use_attention,
                              add_bn=add_bn)
            self.up4 = UpHist(128, 64 * factor, num_bins, bilinear,
                              normalize_count=normalize_count, 
                              normalize_bins=normalize_bins,
                              use_hist=skip_locations[3],up4=True,
                              use_attention=use_attention,
                              add_bn=add_bn,analyze=self.analyze)        
        else:
            self.up1 = Up(1024, 512, bilinear)
            self.up2 = Up(512, 256, bilinear)
            self.up3 = Up(256, 128, bilinear)
            self.up4 = Up(128, 64 * factor, bilinear)
        
        self.outc = OutConv(64, n_classes)


    def forward(self, x):
       
        if self.analyze:
            #Encoder
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            
            #Decoder
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            xhist4, x = self.up4(x, x1)
            logits = self.outc(x)
        
            #Concatenate features before histogram
            # pdb.set_trace()
            # before_hist_feats = torch.cat((x1,x2,x3,x4),dim=1)
            before_hist_feats = x1
            
            #Concatenate features after histogram
            # after_hist_feats = torch.cat((xhist1,xhist2,xhist3,xhist4),dim=1)
            after_hist_feats = xhist4
            
            return (logits, logits, torch.sigmoid(logits))
            # return logits
            
        else:
            #Encoder
            
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            
            #Decoder
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            logits = self.outc(x)
            return logits
      

        
        
        
        
        
        
        
        