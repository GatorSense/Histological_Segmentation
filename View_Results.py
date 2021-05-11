# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 10:15:33 2019
Generate results from saved models
@author: jpeeples
"""

## Python standard libraries
from __future__ import print_function
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import pandas as pd
import random
import os
from sklearn.metrics import matthews_corrcoef
from operator import itemgetter 
import pandas as pd
import pdb

## PyTorch dependencies
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate 

## Local external libraries
from Demo_Parameters import Parameters
from Utils.Initialize_Model import initialize_model
from Prepare_Data import Prepare_DataLoaders
from Utils.Create_Individual_Figures import Generate_Images
from Utils.Capture_Metrics import Get_Metrics
from Utils.Compute_Test_Metrics import Get_Test_Metrics
from Utils.create_dataloaders import Get_Dataloaders
from Utils.Create_Fat_Spreadsheet import Generate_Fat

plt.ioff()

Results_parameters = Parameters()

torch.manual_seed(Results_parameters['random_state'])
np.random.seed(Results_parameters['random_state'])
random.seed(Results_parameters['random_state'])
torch.cuda.manual_seed(Results_parameters['random_state'])
torch.cuda.manual_seed_all(Results_parameters['random_state'])


#Location of experimental results
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')
fig_size = Results_parameters['fig_size']
font_size = Results_parameters['font_size']

NumRuns = Results_parameters['Splits'][Results_parameters['Dataset']]

#Name of dataset
Dataset = Results_parameters['Dataset']

#Model(s) to be used
model_name = Results_parameters['Model_name']

#Number of classes in dataset
num_classes = Results_parameters['num_classes'][Dataset]
                                 
#Number of runs and/or splits for dataset
numRuns = Results_parameters['Splits'][Dataset]

#Number of bins and input convolution feature maps after channel-wise pooling
numBins = Results_parameters['numBins']

#Metrics to capture
metrics = {'dice': 'Dice Coefficent', 'overall_IOU': 'IOU','pos_IOU': 'Positive IOU',
           'haus_dist': 'Hausdorff distance', 'adj_rand': 'Adjusted Rand Index',
           'precision': 'Precision', 'recall': 'Recall', 'f1_score': 'F1 Score',
           'specificity': 'Specificity',
           'pixel_acc': 'Pixel Accuracy','loss': 'Binary Cross Entropy',
           'inf_time': 'Inference Time'}



seg_models = {0: 'UNET', 1: 'UNET+', 2: 'Attention UNET', 3:'JOSHUA', 4: 'JOSHUA+'}
model_selection = {0: 1, 1: 1, 2: 4, 3: 1, 4: 1}
hist_skips = {0: False, 1: False, 2: False, 3: True, 4:True}
hist_pools = {0: False, 1: False, 2: False, 3: False, 4: False}
attention = {0: False, 1: True, 2: True, 3: False, 4: True}

#Initialize array to save scores
max_val_dice = np.zeros((len(seg_models),numRuns))

#Number of images to view in training/validation set
num_imgs = 4

#Return datasets and indices of training/validation data
dataset, indices = Prepare_DataLoaders(Results_parameters,numRuns)

mask_type = torch.float32 if num_classes == 1 else torch.long


#Compute avg and std deviations of val and train metrics, save in spreadsheet
Get_Metrics(metrics,seg_models,model_selection,hist_skips,hist_pools,attention,folds=NumRuns)
# pdb.set_trace()

#Load dataframe containing fat information
fat_df = pd.read_excel(r'Labeled Image Reference Length.xlsx')

#Generate spreadsheet with fat information
folder = (Results_parameters['folder'] + '/'+ Results_parameters['mode'] 
                    + '/' + Results_parameters['Dataset'] + '/Fat_Measures/mm_results/')

Generate_Fat(dataset,indices,mask_type,seg_models,device,NumRuns,hist_skips,
              hist_pools,attention,model_selection,num_classes,fat_df,folder)

#Parse through files and plot results
for split in range(0, NumRuns):
    
    #Generate dataloaders and pos wt
    dataloaders, pos_wt = Get_Dataloaders(dataset,split,indices,Results_parameters,
                                          Results_parameters['batch_size'])

#     #Set number of images to plot 
    max_imgs = {'train': 90, 'val': 90, 'test': 90}
    
#     #Save figures for individual images
    Generate_Images(dataloaders,mask_type,seg_models,device,split,max_imgs,
                    hist_skips,hist_pools,attention,model_selection,num_classes,fat_df,
                    show_fat=True)

 
    print('**********Run ' + str(split+1) + ' Finished**********')
