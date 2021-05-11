# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 16:08:12 2021
Compute validation/test metrics for each model
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
import pickle
import pdb

## PyTorch dependencies
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from View_Results_Parameters import Parameters

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

def load_metrics(sub_dir, metrics, phase = 'val'):
    
    # #Load metrics
    # if test: #Loss variable is tensor, causes issue if no GPU
    #     temp_file = open(sub_dir+'test_metrics.pkl', 'rb')
    # else:
    #     temp_file = open(sub_dir+'val_metrics.pkl', 'rb')
    
    temp_file = open(sub_dir + 'new_{}_metrics.pkl'.format(phase), 'rb')
    # temp_file = open(sub_dir + '{}_metrics.pkl'.format(phase), 'rb')
    temp_metrics = pickle.load(temp_file)
    temp_file.close()
    
    #Return max value for each metric (unless loss or inference time)
    metric_vals = np.zeros(len(metrics))
    
    #Get argmax of dice coefficient and save metrics corresponding to this value
    # max_index = np.argmax(temp_metrics['dice'])
    count = 0
    for metric in metrics.keys():
        metric_vals[count] = temp_metrics[metric]
        count += 1
    
    return metric_vals
 
def add_to_excel(table,writer,model_names,metrics_names,fold=1,overall=False):
    if overall:
        table_avg = np.mean(table,axis=-1)
        table_std = np.std(table,axis=-1)
        DF_avg = pd.DataFrame(table_avg,index=model_names,columns=metrics_names)
        DF_std = pd.DataFrame(table_std,index=model_names,columns=metrics_names)
        DF_avg.to_excel(writer,sheet_name='Overall Avg')
        DF_std.to_excel(writer,sheet_name='Overall Std')
    else:
        DF = pd.DataFrame(table,index=model_names,columns=metrics_names)
        DF.colummns = metrics_names
        DF.index = model_names
        DF.to_excel(writer,sheet_name='Fold_{}'.format(fold+1))
           
#Compute desired metrics and save to excel spreadsheet
def Get_Metrics(metrics,seg_models,model_selection,hist_skips,hist_pools,attention,folds=5):

    #Set names of models
    # pdb.set_trace()
    # print(seg_models.values())
    
    model_names = []
    metric_names = []
    for key in seg_models:
        model_names.append(seg_models[key])
    for key in metrics:
        metric_names.append(metrics[key])
        
    #Intialize validation and test arrays
    train_table = np.zeros((len(seg_models),len(metrics),folds))
    val_table = np.zeros((len(seg_models),len(metrics),folds))
    test_table = np.zeros((len(seg_models),len(metrics),folds))

    _, fig_dir = Generate_Dir_Name(0, Parameters())
    # train_writer = pd.ExcelWriter(fig_dir+'Train_Metrics.xlsx', engine='xlsxwriter')
    val_writer = pd.ExcelWriter(fig_dir+'Val_Metrics.xlsx', engine='xlsxwriter')
    test_writer = pd.ExcelWriter(fig_dir+'Test_Metrics.xlsx', engine='xlsxwriter')
    
    # Initialize the histogram model for this run
    for split in range(0,folds):
        for key in seg_models:
            
            temp_params = Parameters(histogram_skips=hist_skips[key],
                                     histogram_pools=hist_pools[key],
                                     use_attention=attention[key],
                                     model_selection=model_selection[key])
            
            #Get location of best weights
            sub_dir, _ = Generate_Dir_Name(split, temp_params)
            
            #Get metrics for validation and test
            # train_table[key,:,split] = load_metrics(sub_dir,metrics,phase='train')
            val_table[key,:,split] = load_metrics(sub_dir,metrics,phase='val')
            test_table[key,:,split] = load_metrics(sub_dir, metrics,phase = 'test')
            
            print('Fold {}'.format(split+1))
        #Add metrics to spreadsheet
        # add_to_excel(train_table[:,:,split],train_writer,model_names,metric_names,fold=split)
        add_to_excel(val_table[:,:,split],val_writer,model_names,metric_names,fold=split)
        add_to_excel(test_table[:,:,split],test_writer,model_names,metric_names,fold=split)
    
    #Compute average and std across folds
    add_to_excel(val_table,val_writer,model_names,metric_names,overall=True)
    add_to_excel(test_table,test_writer,model_names,metric_names,overall=True)    
    
    #Save spreadsheets
    val_writer.save()
    test_writer.save()
    
    










    
    