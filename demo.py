# -*- coding: utf-8 -*-
"""
Demo for histogram layer segmentation networks (JOSHUA/JOSHUA+)
Current script is only for experiments on
single cpu/gpu. If you want to run the demo
on multiple gpus (two were used in paper), 
please contact me at jpeeples@ufl.edu 
for the parallel version of 
demo.py
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

## PyTorch dependencies
import torch

## Local external libraries
from Utils.Initialize_Model import initialize_model
from Demo_Parameters import Parameters
from Prepare_Data import Prepare_DataLoaders

#UNET functions
from Utils.train import train_net

import pdb

#Reproducibility
Network_parameters = Parameters()
torch.manual_seed(Network_parameters['random_state'])
np.random.seed(Network_parameters['random_state'])
random.seed(Network_parameters['random_state'])
torch.cuda.manual_seed(Network_parameters['random_state'])
torch.cuda.manual_seed_all(Network_parameters['random_state'])

#Name of dataset
Dataset = Network_parameters['Dataset']

#Model(s) to be used
model_name = Network_parameters['Model_name']

#Number of classes in dataset
num_classes = Network_parameters['num_classes'][Dataset]
                                 
#Number of runs and/or splits for dataset
numRuns = Network_parameters['Splits'][Dataset]

#Number of bins and input convolution feature maps after channel-wise pooling
numBins = Network_parameters['numBins']

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Location to store trained models
current_directory = os.getcwd()
final_directory = os.path.join(current_directory, Network_parameters['folder'])

print()
print('Starting Experiments...')

# Create training and validation dataloaders
print("Initializing Datasets and Dataloaders...")

#Return indices of training/validation/test data
indices = Prepare_DataLoaders(Network_parameters,numRuns)

#Loop counter
split = 0

for split in range(0, numRuns):
    
    # Initialize the segmentation model for this run
    model = initialize_model(model_name, num_classes,Network_parameters)

    # Send the model to GPU if available
    model = model.to(device)
    
    #Print number of trainable parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters: %d" % (num_params))  

    # Train and evaluate
    try:
        logging.basicConfig(level=logging.INFO,format='%(levelname)s: %(message)s')
        logging.info(f'Using device {device}')
        logging.info(f'Network:\n'
             f'\t{model.n_channels} input channels\n'
             f'\t{model.n_classes} output channels (classes)\n'
             f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling\n'
             f'\tTotal number of trainable parameters: {num_params}')
           
        train_net(net=model,device=device,indices=indices,
                  split=split,Network_parameters=Network_parameters,
                  epochs=Network_parameters['num_epochs'],
                  batch_size=Network_parameters['batch_size'],
                  lr=Network_parameters['lr_rate'],
                  save_cp = Network_parameters['save_cp'],
                  save_results = Network_parameters['save_results'],
                  save_epoch = Network_parameters['save_epoch'])
    
    except KeyboardInterrupt:
        torch.save(model.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
  
    torch.cuda.empty_cache()
        
    if Network_parameters['hist_model'] is not None:
        print('**********Run ' + str(split + 1) + ' For ' +
              Network_parameters['hist_model'] + ' Finished**********') 
    else:
        print('**********Run ' + str(split + 1) + ' For ' + model_name + 
              ' Finished**********') 
        
    #Iterate counter
    split += 1