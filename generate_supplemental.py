# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 14:10:59 2022
Generate supplemental figures (histogram and adipose poor images)
@author: jpeeples
"""


## Python standard libraries
from __future__ import print_function
from __future__ import division
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

## PyTorch dependencies
import torch

## Local external libraries
from Demo_Parameters import Parameters
from Prepare_Data import Prepare_DataLoaders
from Utils.create_dataloaders import Get_Dataloaders

import pdb

#Turn off plotting
plt.ioff()

def main(Params,args):
    #Name of dataset
    Dataset = Params['Dataset']
    
    #Number of runs and/or splits for dataset
    numRuns = Params['Splits'][Dataset]

    # Detect if we have a GPU available
    use_cuda = args.use_cuda and torch.cuda.is_available()
    
    print()
    # Create training and validation dataloaders
    print("Initializing Datasets and Dataloaders...")
    
    #Return indices of training/validation/test data
    indices = Prepare_DataLoaders(Params,numRuns)
    
    #Loop counter
    split = 0
    
    #Use GPU if available
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    num_bins = args.numBins
    
    R_pos = []
    G_pos = []
    B_pos = []
    R_neg = []
    G_neg = []
    B_neg = []
    
    img_list = []
    percentage = .01
    num_imgs = 0
    
    for split in range(0, numRuns):
        
        #Initialize dataloa
        dataloaders, pos_wt = Get_Dataloaders(split,indices,Params,Params['batch_size'])
        
        #Look at validation images and get RGB values, validation should cover all training images
        for phase in ['val']:
        
            # img_count = 0
            for batch in dataloaders[phase]:
               
                imgs, true_masks, idx = (batch['image'], batch['mask'],
                                                  batch['index'])
                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)
                
                #Compute histograms based on postive/negative labels
                R_pos_temp = torch.masked_select(imgs[:,0],true_masks.gt(0))
                G_pos_temp = torch.masked_select(imgs[:,1],true_masks.gt(0))
                B_pos_temp = torch.masked_select(imgs[:,2],true_masks.gt(0))
                R_neg_temp = torch.masked_select(imgs[:,0],~true_masks.gt(0))
                G_neg_temp = torch.masked_select(imgs[:,1],~true_masks.gt(0))
                B_neg_temp = torch.masked_select(imgs[:,2],~true_masks.gt(0))
                
                #Upate histogram counts
                R_pos.append(R_pos_temp.cpu().flatten().numpy())
                G_pos.append(G_pos_temp.cpu().flatten().numpy())
                B_pos.append(B_pos_temp.cpu().flatten().numpy())
                R_neg.append(R_neg_temp.cpu().flatten().numpy())
                G_neg.append(G_neg_temp.cpu().flatten().numpy())
                B_neg.append(B_neg_temp.cpu().flatten().numpy())
                
                #Get adipose poor images
                if args.data_selection == 1:
                    #Compute fat percentage in image
                    fat_pixel_count = torch.count_nonzero(true_masks.flatten(start_dim=1),dim=-1)
                    pixel_count = imgs.size(dim=2)*imgs.size(dim=3)
                    fat_percent = fat_pixel_count/pixel_count
    
                    #Check if pixel fat is greater than percentage
                    poor_imgs = torch.le(fat_percent,percentage)                 
                    
                    #Append image names 
                    img_count = 0
                    for img in poor_imgs:
                        if img:
                            img_list.append([idx[img_count],np.round_(fat_percent[img_count].item(),decimals=4)])
                        img_count +=1
                    
                num_imgs += imgs.size(0)
                print('Finished {} Images'.format(num_imgs))
            
            print('**********Run ' + str(split + 1) + ' Finished**********') 
            
        #Iterate counter
        split += 1
    
    #Plot Each Channel Histogram
    #G
    if args.data_selection == 2:
        pos_class_name = 'Cancerous Tissue'
        G_loc = 'upper right'
    else:
        pos_class_name = 'Adipose Tissue'
        G_loc = 'upper left'
        
    plt.close('all')
    plt.style.use('seaborn-deep')
    sns.color_palette("colorblind")
    show_density = True
    
    set_bins = np.linspace(0,1,num_bins)
    plt.figure()

    plt.hist([np.concatenate(R_pos, axis=0), np.concatenate(R_neg, axis=0)], bins=set_bins, 
             density=show_density, label=[pos_class_name, 'Background'])
    plt.title('Red Channel Intensity Histogram',fontdict = {'fontsize' : 20})
    plt.legend(loc='upper left', prop={'size': 14})
    plt.xlabel('Normalized Intensity Values', fontdict = {'fontsize' : 16})
    plt.ylabel('P(Normalized Intensity Values)', fontdict = {'fontsize' : 16})
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.subplots_adjust(bottom=0.15)
    plt.savefig('{}/{}_R_Channel_Histogram.png'.format(args.supplemental,Params['Dataset']))
    
    plt.figure()
    plt.hist([np.concatenate(G_pos, axis=0), np.concatenate(G_neg, axis=0)], num_bins, 
             density=show_density, label=[pos_class_name, 'Background'])
    plt.title('Green Channel Intensity Histogram', fontdict = {'fontsize' : 20})
    plt.legend(loc=G_loc, prop={'size': 14})
    plt.xlabel('Normalized Intensity Values', fontdict = {'fontsize' : 16})
    plt.ylabel('P(Normalized Intensity Values)', fontdict = {'fontsize' : 16})
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.savefig('{}/{}_G_Channel_Histogram.png'.format(args.supplemental,Params['Dataset']))
    
    plt.figure()
    plt.hist([np.concatenate(B_pos, axis=0), np.concatenate(B_neg, axis=0)], num_bins, 
             density=show_density, label=[pos_class_name, 'Background'])
    plt.title('Blue Channel Intensity Histogram', fontdict = {'fontsize' : 20})
    plt.legend(loc='upper left', prop={'size': 14})
    plt.xlabel('Normalized Intensity Values', fontdict = {'fontsize' : 16})
    plt.ylabel('P(Normalized Intensity Values)', fontdict = {'fontsize' : 16})
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.subplots_adjust(bottom=0.15)
    plt.savefig('{}/{}_B_Channel_Histogram.png'.format(args.supplemental,Params['Dataset']))
    
    #Return image list
    return img_list
    
       
def parse_args():
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
    
    #Create supplemental figures folder
    supplemental_figs = 'Supplemental Figures/'
    if not os.path.exists(supplemental_figs):
        os.makedirs(supplemental_figs)

    args = parse_args()
    setattr(args, 'supplemental', supplemental_figs)
    params = Parameters(args)
    
    #Generate figures
    imgs = main(params,args)
    
    if args.data_selection == 1:
        df = pd.DataFrame(imgs,columns=['Images','Fat Percentage'])
        writer = pd.ExcelWriter('{}/{}_Adipose_Poor_Samples.xlsx'.format(args.supplemental,
                                                                         params['Dataset']),
                                                                  engine='xlsxwriter')
        df.to_excel(writer,index=False)
        writer.save()