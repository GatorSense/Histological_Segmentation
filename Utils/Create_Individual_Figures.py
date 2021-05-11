# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 14:46:58 2021

@author: jpeeples
"""
## Python standard libraries
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import os
from operator import itemgetter 
from sklearn.metrics import jaccard_score as jsc
import pandas as pd
import pdb

## PyTorch dependencies
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


from View_Results_Parameters import Parameters
from Utils.Initialize_Model import initialize_model
from Utils.functional import *

#mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
def inverse_normalize(tensor, mean=(0,0,0), std=(1,1,1)):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def Generate_Dir_Name(split,Network_parameters):
    
    if Network_parameters['hist_model'] is not None:
        dir_name = (Network_parameters['folder'] + '/' + Network_parameters['mode'] 
                    + '/' + Network_parameters['Dataset'] + '/' 
                    + Network_parameters['hist_model'] + '/Run_' 
                    + str(split + 1) + '/')
    #Baseline model
    else:
        dir_name = (Network_parameters['folder'] + '/'+ Network_parameters['mode'] 
                    + '/' + Network_parameters['Dataset'] + '/' +
                    Network_parameters['Model_names'][Network_parameters['model_selection']] 
                    + '/Run_' + str(split + 1) + '/')  
    
    #Location to save figures
    fig_dir_name = (Network_parameters['folder'] + '/'+ Network_parameters['mode'] 
                    + '/' + Network_parameters['Dataset'] + '/')
        
    return dir_name, fig_dir_name

def Generate_Images(dataloaders,mask_type,seg_models,device,split,
                    max_imgs,hist_skips,hist_pools,attention,model_selection,num_classes,
                    fat_df,show_fat=False,alpha=.35,class_name='Fat'):

    model_names = []
    
    #Set names of models
    for key in seg_models:
        model_names.append(seg_models[key])
    
    hausdorff_pytorch = HausdorffDistance()
    for phase in ['test']:
    #for phase in ['val','test']:
        #Set the maximum number of images to visualize
        temp_max_imgs = max_imgs[phase]
        img_count = 0
        for batch in dataloaders[phase]:
           
            imgs, true_masks, idx = (batch['image'], batch['mask'],
                                              batch['index'])
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)
           
            for img in range(0,imgs.size(0)):
        
                #Create figure for each image
                temp_fig, temp_ax = plt.subplots(nrows=2,ncols=len(seg_models)+2,figsize=(16,8))
                # plt.subplots_adjust(wspace=.4,hspace=.4)
                
                #Initialize fat array
                if show_fat:
                    temp_fat = np.zeros(len(seg_models)+1)
                
                #Get conversion rate from pixels to fat
                # pdb.set_trace()
                if show_fat:
                    temp_rate = fat_df.loc[fat_df['Image Name (.tif)']==idx[img]]['New Reference Length (um/px)'].iloc[-1]
    
            
                #Plot images, hand labels, and masks
                temp_ax[0,0].imshow(inverse_normalize(imgs[img]).cpu().permute(1, 2, 0))
                temp_ax[0,0].tick_params(axis='both', labelsize=0, length = 0)
                temp_ax[0,1].imshow(inverse_normalize(imgs[img]).cpu().permute(1,2,0))
                temp_ax[0,1].imshow(true_masks[img][0].cpu(),'jet',interpolation=None,alpha=alpha)
                temp_ax[0,1].tick_params(axis='both', labelsize=0, length = 0)
                temp_ax[1,1].imshow(true_masks[img][0].cpu(),cmap='gray')
                temp_ax[1,1].tick_params(axis='both', labelsize=0, length = 0)
                axes = temp_ax
                        
                #Compute percentage of fat from ground truth
                if show_fat:
                    temp_fat[0] = np.count_nonzero(true_masks[img][0]) /(temp_rate)
                    # temp_fat[0] = np.count_nonzero(true_masks[img][0]) * (temp_rate)
                
                #Labels Rows and columns
                col_names = [idx[img], 'Ground Truth'] + model_names
                cols = ['{}'.format(col) for col in col_names]
                
                    
                for ax, col in zip(axes[0], cols):
                    ax.set_title(col)
            
                
                # Initialize the histogram model for this run
                for key in seg_models:
                    
                    temp_params = Parameters(histogram_skips=hist_skips[key],
                                             histogram_pools=hist_pools[key],
                                             use_attention=attention[key],
                                             model_selection=model_selection[key])
                    
                    model_name = temp_params['Model_names'][temp_params['model_selection']]
            
                    model = initialize_model(model_name, num_classes,
                                                   temp_params)
                    
                    # If parallelized, need to set change model
                    # if temp_params['Parallelize']:
                    model = nn.DataParallel(model)
            
                    # Send the model to GPU if available
                    model = model.to(device)
                    
                    #Get location of best weights
                    sub_dir, fig_dir = Generate_Dir_Name(split, temp_params)
                    
                    #Load weights for model
                    #pdb.set_trace()
                    model.load_state_dict(torch.load(sub_dir + 'best_wts.pt', 
                                                map_location=torch.device(device)))
                    
                    
                    #Get output and plot
                    model.eval()
                    # pdb.set_trace()
                    with torch.no_grad():
                        preds = model(imgs[img].unsqueeze(0))
                        preds = (torch.sigmoid(preds) > .5).float()
                    
                    #Plot images with masks overlaid (alpha = .15 for histological images)
                    temp_ax[0,key+2].imshow(inverse_normalize(imgs[img]).cpu().permute(1,2,0))
                    temp_ax[0,key+2].imshow(preds[0].cpu().permute(1,2,0)[:,:,0],'jet',
                                            interpolation=None,alpha=alpha)
                    temp_ax[0,key+2].tick_params(axis='both', labelsize=0, length = 0)
                    
                    #Plot masks only
                    temp_ax[1,key+2].imshow(preds[0].cpu().permute(1,2,0)[:,:,0],cmap='gray')
                    temp_ax[1,key+2].tick_params(axis='both', labelsize=0, length = 0)
                    
                    #Computed weighted IOU (account for class imbalance)
                    temp_IOU_pos = np.round(iou(preds,true_masks[img]).item(),decimals=2)
                    # pdb.set_trace()
                    try:
                        temp_haus = np.round(hausdorff_pytorch.compute(preds,true_masks[img].unsqueeze(0)).item(),decimals=2)
                    except:
                        temp_haus = 'inf'
                    f1_score = np.round(f_score(preds,true_masks[img]).item(),decimals=2)
                    temp_true_masks = true_masks[img].cpu().numpy().reshape(-1).astype(int)
                    temp_preds = preds[0].cpu().numpy().reshape(-1).astype(int)
                    temp_IOU_macro = np.round(jsc(temp_true_masks, temp_preds, average='macro'),decimals=2)
                    temp_ax[1,key+2].set_title('{} IOU: {}, \n Dice (F1): {}, \n Hausdorff: {}'.format(class_name,
                                                                                                          temp_IOU_pos, 
                                                                                                           f1_score,
                                                                                                           temp_haus))
                    del model
                    del temp_haus
                    torch.cuda.empty_cache()

                    #Compute estimated fat
                    if show_fat:
                        temp_fat[key+1] = np.count_nonzero(preds[0]) /(temp_rate)
                 
                #Plot estimated fat area and highlight model closest to actual value
                # pdb.set_trace()
                if show_fat:
                    y_pos = np.arange(len(temp_fat))
                    rects = temp_ax[1,0].bar(y_pos,temp_fat)
                    temp_ax[1,0].set_xticks(y_pos)
                    temp_ax[1,0].set_xticklabels(col_names[1:],rotation=90)
                    temp_ax[1,0].set_ylabel('Est Fat Area ($\u03BCm^2$)')
                    temp_ax[1,0].patches[np.argmin(abs(temp_fat[1:]-temp_fat[0]))+1].set_facecolor('#aa3333')
                    temp_ax[1,0].set_title('Est Fat Area ($\u03BCm^2$) for Each Model: \n' + str())
                else:
                    temp_ax[1,0].set_frame_on(False)
                    temp_ax[1,0].tick_params(axis='both', labelsize=0, length = 0)
        
                
                folder = fig_dir + '{}_Images/Run_{}/'.format(phase.capitalize(),split+1)
                #Create Training and Validation folder
                if not os.path.exists(folder):
                    os.makedirs(folder)
                 
                img_name = folder + idx[img] + '.png'
                
                temp_fig.savefig(img_name,dpi=temp_fig.dpi)
                plt.close(fig=temp_fig)
                
                img_count += 1
                print('Finished image {} of {}'.format(img_count,len(dataloaders[phase].sampler)))
                
                if img_count == temp_max_imgs:
                    break
                
            if img_count == temp_max_imgs:
                break
        
    
    
    
    
    
    
    