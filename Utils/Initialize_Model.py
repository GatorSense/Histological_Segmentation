# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 10:57:56 2020

@author: jpeeples
"""
## Python standard libraries
from __future__ import print_function
from __future__ import division

## Local external libraries
from .models.Histogram_Model import JOSHUA
from .models.unet_model import UNet
from .models.attention_unet_model import AttUNet

       
def initialize_model(model_name, num_classes,Network_parameters):
 
    #Generate segmentation model 
    if (model_name == 'JOSHUA') or (model_name == 'JOSHUA+'):
            model = JOSHUA(Network_parameters['channels'],num_classes,
                             skip=Network_parameters['histogram_skips'],
                             pool=Network_parameters['histogram_pools'],
                             bilinear=Network_parameters['bilinear'],
                             num_bins=Network_parameters['numBins'],
                             normalize_count=Network_parameters['normalize_count'],
                             normalize_bins=Network_parameters['normalize_bins'],
                             skip_locations=Network_parameters['skip_locations'],
                             pool_locations=Network_parameters['pool_locations'],
                             use_attention=Network_parameters['use_attention'],
                             feature_extraction=Network_parameters['feature_extraction'],
                             add_bn=Network_parameters['add_bn'])
            
    #Base UNET model or UNET+ (our version of attention)
    elif (model_name == 'UNET') or (model_name == 'UNET+'): 
        model = UNet(Network_parameters['channels'],num_classes,
                     bilinear = Network_parameters['bilinear'],
                     feature_extraction = Network_parameters['feature_extraction'],
                     use_attention=Network_parameters['use_attention'])
    
    #Attetion UNET model introduced in 2018
    elif model_name == 'Attention_UNET':
            model = AttUNet(Network_parameters['channels'],num_classes,
                          bilinear = Network_parameters['bilinear'],
                          feature_extraction = Network_parameters['feature_extraction'],
                          use_attention=Network_parameters['use_attention'])

   
    else: #Show error that segmentation model is not available
        raise RuntimeError('Invalid model')


    return model
