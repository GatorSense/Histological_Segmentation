# -*- coding: utf-8 -*-
"""
Main demo script for histological segmentation models. 
Used to train all models (modify line 204 to select certain models)
@author: jpeeples
"""

## Python standard libraries
from __future__ import print_function
from __future__ import division
import numpy as np
import os
import argparse
import logging
import sys
import random
import matplotlib.pyplot as plt

## PyTorch dependencies
import torch
import torch.nn as nn

## Local external libraries
from Utils.Initialize_Model import initialize_model
from Demo_Parameters import Parameters
from Prepare_Data import Prepare_DataLoaders

#UNET functions
from Utils.train import train_net

import pdb

#Turn off plotting
plt.ioff()

def main(Params,args):
    # #Reproducibility
    torch.manual_seed(Params['random_state'])
    np.random.seed(Params['random_state'])
    random.seed(Params['random_state'])
    torch.cuda.manual_seed(Params['random_state'])
    torch.cuda.manual_seed_all(Params['random_state'])
    
    #Name of dataset
    Dataset = Params['Dataset']
    
    #Model(s) to be used
    model_name = Params['Model_name']
    
    #Number of classes in dataset
    num_classes = Params['num_classes'][Dataset]
                                     
    #Number of runs and/or splits for dataset
    numRuns = Params['Splits'][Dataset]

    # Detect if we have a GPU available
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    print()
    print('Starting Experiments...')
    
    # Create training and validation dataloaders
    print("Initializing Datasets and Dataloaders...")
    
    #Return indices of training/validation/test data
    indices = Prepare_DataLoaders(Params,numRuns)
    
    #Loop counter
    split = 0
    
    for split in range(0, numRuns):
        
        # Initialize the segmentation model for this run
        model = initialize_model(model_name, num_classes,Params)
        
        # Send the model to GPU if available, use multiple if available
        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
        model = model.to(device)
    
        # Send the model to GPU if available
        model = model.to(device)
        
        #Print number of trainable parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
        # Train and evaluate
        try:
            if torch.cuda.device_count() > 1:
                n_channels = model.module.n_channels
                n_classes = model.module.n_classes
                bilinear = model.module.bilinear
            else:
                n_channels = model.n_channels
                n_classes = model.n_classes
                bilinear = model.bilinear
                
            logging.basicConfig(level=logging.INFO,format='%(levelname)s: %(message)s')
            logging.info(f'Using device {device}')
            logging.info(f'Network:\n'
                 f'\t{n_channels} input channels\n'
                 f'\t{n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if bilinear else "Transposed conv"} upscaling\n'
                 f'\tTotal number of trainable parameters: {num_params}')
               
            train_net(net=model,device=device,indices=indices,
                      split=split,Network_parameters=Params,
                      epochs=Params['num_epochs'],
                      batch_size=Params['batch_size'],
                      lr=Params['lr_rate'],
                      save_cp = Params['save_cp'],
                      save_results = Params['save_results'],
                      save_epoch = Params['save_epoch'])
        
        except KeyboardInterrupt:
            torch.save(model.state_dict(), 'INTERRUPTED.pth')
            logging.info('Saved interrupt')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
      
        torch.cuda.empty_cache()
            
        if Params['hist_model'] is not None:
            print('**********Run ' + str(split + 1) + ' For ' +
                  Params['hist_model'] + ' Finished**********') 
        else:
            print('**********Run ' + str(split + 1) + ' For ' + model_name + 
                  ' Finished**********') 
            
        #Iterate counter
        split += 1
       
def parse_args():
    # 'UNET'
    # 'Attention UNET'
    # 'UNET+'
    # 'JOSHUA'
    # 'JOSHUA+'
    parser = argparse.ArgumentParser(description='Run segmentation models for dataset')
    parser.add_argument('--save_results', type=bool, default=True,
                        help='Save results of experiments(default: True')
    parser.add_argument('--save_cp', type=bool, default=False,
                        help='Save results of experiments at each checkpoint (default: False)')
    parser.add_argument('--save_epoch', type=int, default=5,
                        help='Epoch for checkpoint (default: 5')
    parser.add_argument('--folder', type=str, default='Saved_Models/',
                        help='Location to save models')
    parser.add_argument('--model', type=str, default='JOSHUA+',
                        help='Select model to train with (default: JOSHUA+')
    parser.add_argument('--data_selection', type=int, default=1,
                        help='Dataset selection:  1: SFBHI, 2: GlaS')
    parser.add_argument('--channels', type=int, default=3,
                        help='Input channels of network (default: 3, RGB images)')
    parser.add_argument('--bilinear', type=bool, default=True,
                        help='Upsampling feature maps, set to True to use bilinear interpolation. Set to False to learn transpose convolution (consume more memory)')
    parser.add_argument('--augment', type=bool, default=True,
                        help='Data augmentation (default: True)')
    parser.add_argument('--rotate', type=bool, default=True,
                        help='Training data will be rotated, random flip (p=.5), random patch extraction (default:True')
    parser.add_argument('-numBins', type=int, default=16,
                        help='Number of bins for histogram layer. Recommended values are 4, 8 and 16. (default: 16)')
    parser.add_argument('--feature_extraction', type=bool, default=True,
                        help='Flag for feature extraction. False, train whole model. True, only update fully connected and histogram layers parameters (default: True)')
    parser.add_argument('--use_pretrained', type=bool, default=False,
                        help='Flag to use pretrained model from ImageNet or train from scratch (default: False)')
    parser.add_argument('--train_batch_size', type=int, default=1,
                        help='input batch size for training (default: 8)')
    parser.add_argument('--val_batch_size', type=int, default=1,
                        help='input batch size for validation (default: 10)')
    parser.add_argument('--test_batch_size', type=int, default=1,
                        help='input batch size for testing (default: 10)')
    parser.add_argument('--num_epochs', type=int, default=2,
                        help='Number of epochs to train each model for (default: 150)')
    parser.add_argument('--random_state', type=int, default=1,
                        help='Set random state for K fold CV for repeatability of data/model initialization (default: 1)')
    parser.add_argument('--add_bn', type=bool, default=False,
                        help='Add batch normalization before histogram layer(s) (default: False)')
    parser.add_argument('--padding', type=int, default=0,
                        help='If padding is desired, enter amount of zero padding to add to each side of image  (default: 0)')
    parser.add_argument('--normalize_count',type=bool, default=True,
                        help='Set whether to use sum (unnormalized count) or average pooling (normalized count) (default: True)')
    parser.add_argument('--normalize_bins',type=bool, default=True,
                        help='Set whether to enforce sum to one constraint across bins (default: True)')
    parser.add_argument('--resize_size', type=int, default=None,
                        help='Resize the image before center crop. (default: 256)')
    parser.add_argument('--center_size', type=int, default=None,
                        help='Center crop image. (default: 256)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--parallelize_model', type=bool, default=True,
                        help='enables CUDA training')
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='enables CUDA training')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    #Trains all models
    model_list = ['JOSHUA+','UNET','UNET+','Attention_UNET', 'JOSHUA']
    args = parse_args()
    
    model_count = 0
    for model in model_list:
        setattr(args, 'model', model)
        params = Parameters(args)
        main(params,args)
        model_count += 1
        print('Finished Model {} of {}'.format(model_count,len(model_list)))
        
