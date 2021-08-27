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
from sklearn.metrics import jaccard_score as jsc
import pdb

## PyTorch dependencies
import torch
import torch.nn as nn

## Local external libraries
from Demo_Parameters import Parameters
from Utils.initialize_model import initialize_model
from Utils.functional import *
from Utils.decode_segmentation import decode_segmap
from Utils.metrics import eval_metrics

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
                    Network_parameters['Model_name'] 
                    + '/Run_' + str(split + 1) + '/')  
    
    #Location to save figures
    fig_dir_name = (Network_parameters['folder'] + '/'+ Network_parameters['mode'] 
                    + '/' + Network_parameters['Dataset'] + '/')
        
    return dir_name, fig_dir_name

def inverse_normalize(tensor, mean=(0,0,0), std=(1,1,1)):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def Generate_Images(dataloaders,mask_type,seg_models,device,split,
                    num_classes,fat_df,args,show_fat=False,alpha=.35,class_name='Fat'):

    model_names = []
    
    #Set names of models
    for key in seg_models:
        model_names.append(seg_models[key])
    
    hausdorff_pytorch = HausdorffDistance()
    for phase in ['val']:
    
        img_count = 0
        for batch in dataloaders[phase]:
           
            imgs, true_masks, idx = (batch['image'], batch['mask'],
                                              batch['index'])
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)
           
            for img in range(0,imgs.size(0)):
        
                #Create figure for each image
                temp_fig, temp_ax = plt.subplots(nrows=2,ncols=len(seg_models)+2,figsize=(16,8))
                
                #Initialize fat array
                if show_fat:
                    temp_fat = np.zeros(len(seg_models)+1)
                
                #Get conversion rate from pixels to fat
                if show_fat:
                    temp_org_size = fat_df.loc[fat_df['Image Name (.tif)']==idx[img]]['# of Pixels'].iloc[-1]
                    temp_ds_size = fat_df.loc[fat_df['Image Name (.tif)']==idx[img]]['Down sampled # of Pixels'].iloc[-1]
                    temp_org_rate = fat_df.loc[fat_df['Image Name (.tif)']==idx[img]]['Reference Length (um/px)'].iloc[-1]
        
                #Plot images, hand labels, and masks
                temp_ax[0,0].imshow(imgs[img].cpu().permute(1, 2, 0))
                temp_ax[0,0].tick_params(axis='both', labelsize=0, length = 0)
                
                if num_classes == 1:
                    temp_ax[0,1].imshow(imgs[img].cpu().permute(1,2,0))
                    temp_ax[0,1].imshow(true_masks[img][0].cpu(),'jet',interpolation=None,alpha=alpha)
                    temp_ax[0,1].tick_params(axis='both', labelsize=0, length = 0)
                    temp_ax[1,1].imshow(true_masks[img][0].cpu(),cmap='gray')
                    temp_ax[1,1].tick_params(axis='both', labelsize=0, length = 0)
                else:
                    temp_ax[0,1].imshow(imgs[img].cpu().permute(1,2,0))
                    temp_true = decode_segmap(true_masks[img].cpu().numpy())
                    temp_ax[0,1].imshow(temp_true,interpolation=None,alpha=alpha)
                    temp_ax[0,1].tick_params(axis='both', labelsize=0, length = 0)
                    temp_mask = decode_segmap(true_masks[img].cpu().numpy(),nc=num_classes)
                    temp_ax[1,1].imshow(temp_mask)
                    temp_ax[1,1].tick_params(axis='both', labelsize=0, length = 0)
                axes = temp_ax
                        
                #Compute percentage of fat from ground truth
                if show_fat:
                    temp_fat[0] = true_masks[img][0].count_nonzero().item() * (temp_org_size/temp_ds_size) * (temp_org_rate)**2
                 
                #Labels Rows and columns
                if num_classes == 1:
                    col_names = [idx[img], 'Ground Truth'] + model_names
                else:
                    col_names = ['Input Image', 'Ground Truth'] + model_names
                cols = ['{}'.format(col) for col in col_names]
                
                for ax, col in zip(axes[0], cols):
                    ax.set_title(col)
            
                
                # Initialize the histogram model for this run
                for key in seg_models:
                    
                    setattr(args, 'model', seg_models[key])
                    temp_params = Parameters(args)
                    
            
                    model = initialize_model(seg_models[key], num_classes,temp_params)
                    
                    sub_dir, fig_dir = Generate_Dir_Name(split, temp_params)
                    
                    #If parallelized, need to set model
                      # Send the model to GPU if available
                    try:
                        model = nn.DataParallel(model)
                        model = model.to(device)
                        model.load_state_dict(torch.load(sub_dir + 'best_wts.pt', 
                                                map_location=torch.device(device)))
                    except:
                        model = model.to(device)
                        model.load_state_dict(torch.load(sub_dir + 'best_wts.pt', 
                                                map_location=torch.device(device)))
                    
                    #Get location of best weights
                    sub_dir, fig_dir = Generate_Dir_Name(split, temp_params)
                    
                    #Get output and plot
                    model.eval()
                    
                    with torch.no_grad():
                        preds = model(imgs[img].unsqueeze(0))
                    
                    #Plot images with masks overlaid (alpha = .15 for histological images)
                    temp_ax[0,key+2].imshow(imgs[img].cpu().permute(1,2,0))
                    
                    if num_classes == 1:
                        preds = (torch.sigmoid(preds) > .5).float()
                        temp_ax[0,key+2].imshow(preds[0].cpu().permute(1,2,0)[:,:,0],'jet',
                                                interpolation=None,alpha=alpha)
                        temp_ax[0,key+2].tick_params(axis='both', labelsize=0, length = 0)
                        
                        #Plot masks only
                        temp_ax[1,key+2].imshow(preds[0].cpu().permute(1,2,0)[:,:,0],cmap='gray')
                        temp_ax[1,key+2].tick_params(axis='both', labelsize=0, length = 0)
                        
                        #Computed weighted IOU (account for class imbalance)
                        temp_IOU_pos = iou(preds,true_masks[img]).item()
                        try:
                            temp_haus = hausdorff_pytorch.compute(preds,true_masks[img].unsqueeze(0)).item()
                        except:
                            temp_haus = 'inf'
                        f1_score = np.round(f_score(preds,true_masks[img]).item(),decimals=2)
                        temp_true_masks = true_masks[img].cpu().numpy().reshape(-1).astype(int)
                        temp_preds = preds[0].cpu().numpy().reshape(-1).astype(int)
                        temp_ax[1,key+2].set_title('{} IOU: {:.2f}, \n Dice (F1): {:.2f}, \n Hausdorff: {:.2f}'.format(class_name,
                                                                                                              temp_IOU_pos, 
                                                                                                               f1_score,
                                                                                                               temp_haus))
                        
                        del temp_haus
                        
                    else:
                        temp_pred = torch.argmax(preds[0], dim=0).detach().cpu().numpy()
                        temp_pred = decode_segmap(temp_pred,nc=num_classes)
                        temp_ax[0,key+2].imshow(temp_pred,
                                                interpolation=None,alpha=alpha)
                        temp_ax[0,key+2].tick_params(axis='both', labelsize=0, length = 0)
                        
                        #Plot masks only
                        temp_ax[1,key+2].imshow(temp_pred)
                        temp_ax[1,key+2].tick_params(axis='both', labelsize=0, length = 0)
                        
                        #Computed weighted IOU (account for class imbalance)
                        _, _, avg_jacc, avg_dice, avg_mAP = eval_metrics(true_masks[img].unsqueeze(0),
                                                                         preds,num_classes)
                        temp_ax[1,key+2].set_title('IOU: {:.2f}, \n F1 Score: {:.2f}, \n mAP: {:.2f}'.format(avg_jacc, avg_dice, avg_mAP))
                    del model
                    torch.cuda.empty_cache()

                    #Compute estimated fat
                    if show_fat:
                        temp_fat[key+1] = preds[0].count_nonzero().item() * (temp_org_size/temp_ds_size) * (temp_org_rate)**2
                 
                #Plot estimated fat area and highlight model closest to actual value
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
                print('Finished image {} of {} for {} dataset'.format(img_count,len(dataloaders[phase].sampler),phase))
        
    
    
    
    
    
    
    