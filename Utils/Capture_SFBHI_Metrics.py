# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 14:46:58 2021
Get metrics based on indiviudal images, conditions, and weeks
@author: jpeeples
"""
## Python standard libraries
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import time
import pdb

## PyTorch dependencies
import torch
import torch.nn as nn

## Local external libraries
from Demo_Parameters import Parameters
from Utils.initialize_model import initialize_model
from Utils.functional import *
from Utils.eval import Average_Metric, dice_coeff
from Utils.create_dataloaders import Get_Dataloaders
from Utils.Capture_Metrics import add_to_excel

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

def Compute_SFBHI_Metrics(indices,mask_type,metrics,seg_models,device,folds,
                    num_classes,labels_df,args,temp_params=None):

    model_names = []
    if temp_params is None:
        temp_params = Parameters(args)
    
    #Set names of models and metrics
    model_names = []
    metric_names = []
    for key in seg_models:
        model_names.append(seg_models[key])
    for key in metrics:
        metric_names.append(metrics[key])
        
    #Initialize tables
    train_fold_tables = []
    val_fold_tables = []

    #Generate master table of results
    for fold in range(0,folds):
        
        dataloaders, pos_wt = Get_Dataloaders(fold,indices,temp_params,
                                              temp_params['batch_size'])
        val_table = np.zeros((len(dataloaders['val'].sampler),4+len(metrics),len(seg_models)),dtype='object')
        test_table = np.zeros((len(dataloaders['test'].sampler),4+len(metrics),len(seg_models)),dtype='object')
        for phase in ['val']:
        
            img_count = 0
            for batch in dataloaders[phase]:
               
                imgs, true_masks, idx = (batch['image'], batch['mask'],
                                                  batch['index'])
                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=mask_type)
               
                for img in range(0,imgs.size(0)):
                    #Get conversion rate from pixels to fat
                    #Remove white space
                    labels_df.columns = labels_df.columns.str.rstrip()
                    labels_df = labels_df.apply(lambda x: x.str.rstrip() if x.dtype == "object" else x)
                    try:
                        img_week = labels_df.loc[labels_df['Image Name']==idx[img]]['Week'].iloc[-1]
                        img_condition_letter = labels_df.loc[labels_df['Image Name']==idx[img]]['Condition - Letter'].iloc[-1]
                        img_condition_name = labels_df.loc[labels_df['Image Name']==idx[img]]['Condition - Name'].iloc[-1]
                    except: #Image is not there, set img labels to None
                        img_week = 'N/A'
                        img_condition_letter = 'N/A'
                        img_condition_name = 'N/A'
                     
                    # Initialize the models and compute metrics
                    for key in seg_models:
                        
                        setattr(args, 'model', seg_models[key])
                        temp_params = Parameters(args)
                        model = initialize_model(seg_models[key], num_classes,temp_params)
                        
                        #Get location of best weights
                        sub_dir, _ = Generate_Dir_Name(fold, temp_params)
                        
                        # If parallelized, need to set model
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
                        sub_dir, fig_dir = Generate_Dir_Name(fold, temp_params)
                        
                        #Get output and plot
                        model.eval()
                        
                        with torch.no_grad():
                            temp_start_time = time.time()
                            output = model(imgs[img].unsqueeze(0))
                            inf_time = (time.time() - temp_start_time)
                        
                        #Compute metrics
                        loss = Average_Metric(output,true_masks,pos_wt=torch.tensor(pos_wt).to(device),metric_name='BCE')
                        pred = (output > 0.5).float()
                        prec = Average_Metric(pred, true_masks,metric_name='Precision')
                        rec = Average_Metric(pred, true_masks,metric_name='Recall')
                        f1_score = Average_Metric(pred, true_masks,metric_name='F1')
                        haus_dist, _ = Average_Metric(pred, true_masks,metric_name='Hausdorff')
                        jacc_score = Average_Metric(pred, true_masks,metric_name='Jaccard')
                        tot = dice_coeff(pred, true_masks).item()
                        adj_rand = Average_Metric(pred, true_masks,metric_name='Rand')
                        iou_score = Average_Metric(pred, true_masks,metric_name='IOU_All')
                        val_acc = Average_Metric(pred, true_masks,metric_name='Acc')
                        spec = Average_Metric(pred, true_masks,metric_name='Spec')
                        
                        metric_dict = {'dice': tot,'pos_IOU': jacc_score,
                        'loss': loss,'inf_time': inf_time,'overall_IOU': iou_score,
                        'pixel_acc': val_acc,'haus_dist': haus_dist,
                        'adj_rand': adj_rand,'precision': prec,'recall': rec,
                        'f1_score': f1_score, 'specificity': spec}
                        
                        del model
                        torch.cuda.empty_cache()
                        
                        #Convert metrics to list based on order provided
                        metric_vals = np.zeros(len(metrics))
    
                        metric_count = 0
                        
                        for metric in metric_dict.keys():
                            metric_vals[metric_count] = metric_dict[metric]
                            metric_count += 1
                        
                        #Fill in table with metrics
                        img_info = np.array([idx[img],img_week,img_condition_letter,img_condition_name],dtype='object')
                        img_info = np.concatenate((img_info,metric_vals))
                        if phase == 'val':
                            val_table[img_count,:,key] = img_info
                        elif phase == 'test':
                            test_table[img_count,:,key] = img_info
                     
                    img_count += 1
                    print('Finished image {} of {} for {} dataset for Fold {}'.format(img_count,len(dataloaders[phase].sampler),phase,fold))
        
        # pdb.set_trace()
        val_fold_tables.append(val_table)
        # test_fold_tables.append(test_table)
        
    #Separate results by labels (conditions and weeks, TBD) 
    
    #Get weeks and conditions, Combine later
    weeks = labels_df['Week'].unique()
    conditions_letters = labels_df['Condition - Letter'].unique()
    conditions_names = labels_df['Condition - Name'].unique()
    for week in weeks:
        val_table = np.zeros((len(seg_models),len(metrics),folds))
        _, fig_dir = Generate_Dir_Name(0, temp_params)
        fig_dir = fig_dir + 'Metrics/Weeks/'
        if not os.path.exists(fig_dir):
                      os.makedirs(fig_dir)
                     
        val_writer = pd.ExcelWriter(fig_dir+'Week_{}_Val_Metrics.xlsx'.format(week), engine='xlsxwriter')
        for fold in range(0,folds):
            #Grab all images that equal label
            temp_table = val_fold_tables[fold]
            mask = (temp_table[:,1,:] == week)
            mask = np.repeat(mask[:,np.newaxis,:],temp_table.shape[1],axis=1)
            temp_img_table = temp_table[mask]
            
            #Average metrics across images in fold
            if not temp_img_table.any():
                avg_metrics = np.nan*np.average(temp_table[:,4:,:],axis=0)
            else:
                #Use  true values of images to get values
                temp_img_table = temp_table[mask[:,0,0]]
                avg_metrics = np.average(temp_img_table[:,4:,:],axis=0)
            val_table[:,:,fold] = avg_metrics.T
            
            #Save to excel file in metrics folder
            add_to_excel(val_table[:,:,fold],val_writer,model_names,metric_names,
                          fold=fold)
            
            #Compute average and std across folds
        add_to_excel(val_table,val_writer,model_names,metric_names,overall=True)
        # add_to_excel(test_table,test_writer,model_names,metric_names,overall=True)    
    
        #Save spreadsheets
        val_writer.save()
        # test_writer.save()
        
    for condition in conditions_letters:
        val_table = np.zeros((len(seg_models),len(metrics),folds))
        _, fig_dir = Generate_Dir_Name(0, temp_params)
        fig_dir = fig_dir + 'Metrics/Conditions/'
        if not os.path.exists(fig_dir):
                     os.makedirs(fig_dir)
                     
        val_writer = pd.ExcelWriter(fig_dir+'Condition_{}_Val_Metrics.xlsx'.format(condition), engine='xlsxwriter')
        for fold in range(0,folds):
            #Grab all images that equal label
            temp_table = val_fold_tables[fold]
            mask = (temp_table[:,2,:] == condition)
            mask = np.repeat(mask[:,np.newaxis,:],temp_table.shape[1],axis=1)
            temp_img_table = temp_table[mask]
            
            #Average metrics across images in fold
            if not temp_img_table.any():
                avg_metrics = np.nan*np.average(temp_table[:,4:,:],axis=0)
            else:
                #Use  true values of images to get values
                temp_img_table = temp_table[mask[:,0,0]]
                avg_metrics = np.average(temp_img_table[:,4:,:],axis=0)
            val_table[:,:,fold] = avg_metrics.T
            
            #Save to excel file in metrics folder
            add_to_excel(val_table[:,:,fold],val_writer,model_names,metric_names,
                         fold=fold)
            
            #Compute average and std across folds
        add_to_excel(val_table,val_writer,model_names,metric_names,overall=True)
        # add_to_excel(test_table,test_writer,model_names,metric_names,overall=True)    
    
        #Save spreadsheets
        val_writer.save()
        # test_writer.save()
        

        
    
    
    
    
    
    