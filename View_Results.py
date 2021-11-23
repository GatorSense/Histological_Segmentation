# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 10:15:33 2019
Generate results from saved models
Note: Script should be used if ALL models are saved out
If only interested in certain models, modify the "seg_models" (Line 69)
dictionary to only include models of interests
@author: jpeeples
"""

## Python standard libraries
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import pdb
import argparse

## PyTorch dependencies
import torch

## Local external libraries
from Demo_Parameters import Parameters
from Prepare_Data import Prepare_DataLoaders
from Utils.Create_Individual_RGB_Figures import Generate_Images
from Utils.Capture_Metrics import Get_Metrics
from Utils.create_dataloaders import Get_Dataloaders
from Utils.Create_Fat_Spreadsheet import Generate_Fat

plt.ioff()


def main(Params,args):
    torch.cuda.empty_cache()
    torch.manual_seed(Params['random_state'])
    np.random.seed(Params['random_state'])
    random.seed(Params['random_state'])
    torch.cuda.manual_seed(Params['random_state'])
    torch.cuda.manual_seed_all(Params['random_state'])
    
    
    #Location of experimental results
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
 
    #Name of dataset
    Dataset = Params['Dataset']
    
    #Number of classes in dataset
    num_classes = Params['num_classes'][Dataset]
                                     
    #Number of runs and/or splits for dataset
    numRuns = Params['Splits'][Dataset]
    
    #Metrics to capture
    if num_classes == 1:
        metrics = {'dice': 'Dice Coefficent', 'overall_IOU': 'IOU','pos_IOU': 'Positive IOU',
                    'haus_dist': 'Hausdorff distance', 'adj_rand': 'Adjusted Rand Index',
                    'precision': 'Precision', 'recall': 'Recall', 'f1_score': 'F1 Score',
                    'specificity': 'Specificity',
                    'pixel_acc': 'Pixel Accuracy','loss': 'Binary Cross Entropy',
                    'inf_time': 'Inference Time'}
    else:
        metrics = {'dice': 'F1 Score', 'jacc': 'Jaccard/IOU', 'pixel_acc': 'Overall Pixel Accuracy',
                   'class_acc': 'Pixel Class Accuarcy', 'mAP': 'Mean Average Precision',
                   'loss': 'Cross Entropy', 'inf_time': 'Inference Time'}
    
    seg_models = {0: 'UNET', 1: 'UNET+', 2: 'Attention_UNET', 3:'JOSHUA', 4: 'JOSHUA+'}
    
    
    #Return datasets and indices of training/validation data
    indices = Prepare_DataLoaders(Params,numRuns,data_type=args.data_split)
    
    mask_type = torch.float32 if num_classes == 1 else torch.long
    
    #Compute avg and std deviations of val and train metrics, save in spreadsheet
    Get_Metrics(metrics,seg_models,args,folds=numRuns)
    
    #Load dataframe containing fat and label information
    fat_df = pd.read_excel(r'Labeled Image Reference Length.xls')
    labels_df = pd.read_excel(r'Image Name, Week, and Condition.xls')
   
    #Generate spreadsheet with fat information
    folder = (Params['folder'] + '/'+ Params['mode'] 
                        + '/' + Params['Dataset'] + '/Fat_Measures/um_results/')
   
    if Params['show_fat']:
        Generate_Fat(indices,mask_type,seg_models,device,numRuns,
                  num_classes,fat_df,folder,args,temp_params=Params)
        
    #Parse through files and plot results
    for split in range(0, numRuns):
        
        #Generate dataloaders and pos wt
        dataloaders, pos_wt = Get_Dataloaders(split,indices,Params,Params['batch_size'])
    
        #Save figures for individual images
        Generate_Images(dataloaders,mask_type,seg_models,device,split,
                        num_classes,fat_df,args,alpha=.6,show_fat=Params['show_fat'])
    
        print('**********Run ' + str(split+1) + ' Finished**********')
        
def parse_args():
        # 'UNET'
    # 'Attention UNET'
    # 'UNET+'
    # 'JOSHUA'
    # 'JOSHUA+'
    parser = argparse.ArgumentParser(description='Get results for dataset')
    parser.add_argument('--save_results', type=bool, default=True,
                        help='Save results of experiments(default: True')
    parser.add_argument('--save_cp', type=bool, default=False,
                        help='Save results of experiments at each checkpoint (default: False)')
    parser.add_argument('--save_epoch', type=int, default=5,
                        help='Epoch for checkpoint (default: 5')
    parser.add_argument('--folder', type=str, default='HPG_Results/Journal_Data_Splits/',
                        help='Location to save models')
    parser.add_argument('--model', type=str, default='JOSHUA+',
                        help='Select model to train with (default: JOSHUA+')
    parser.add_argument('--data_selection', type=int, default=2,
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
    parser.add_argument('--feature_extraction', type=bool, default=False,
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
    parser.add_argument('--random_state', type=int, default=2,
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
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--parallelize_model', type=bool, default=True,
                        help='enables CUDA training')
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='enables CUDA training')
    parser.add_argument('--data_split', type=str, default='Random',
                help='Select data split SFBHI: Random (default), Time, Condition')
    parser.add_argument('--week', type=int, default=1,
                        help='Week for new images without labels. (default: 1)')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    args = parse_args()
    params = Parameters(args)
    main(params,args)

    
