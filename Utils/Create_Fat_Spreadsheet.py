# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 21:04:33 2021
Generate spreadsheet of fat
@author: jpeeples
"""
## Python standard libraries
from __future__ import print_function
import numpy as np
import os
import pandas as pd
import pdb

## PyTorch dependencies
import torch
import torch.nn as nn

from View_Results_Parameters import Parameters
from Utils.Initialize_Model import initialize_model
from Utils.functional import *
from Utils.PytorchUNet.create_dataloaders import Get_Dataloaders


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

def add_to_excel(table,writer,model_names,img_names,fold=1):
   
    DF = pd.DataFrame(table,index=img_names,columns=model_names)
    DF.colummns = model_names
    DF.index = img_names
    DF.to_excel(writer,sheet_name='Fold_{}'.format(fold+1))

def Generate_Fat(indices,mask_type,seg_models,device,folds,num_classes,
                 fat_df,folder):

    temp_params = Parameters()
    model_names = []
    model_names.append('Ground Truth')

    #Set names of models
    for key in seg_models:
        model_names.append(seg_models[key])
    del key
    
    for phase in ['val','test']:
        
        writer = pd.ExcelWriter(folder+'{}_Fat_Measures.xlsx'.format(phase.capitalize()),engine='xlsxwriter')
        
        for split in range(0,folds):
            
            #Generate dataloaders and pos wt
            dataloaders = Get_Dataloaders(split,indices,temp_params,temp_params['batch_size'])
        
        #Initialize table for data
            fat_table = []
            img_names = []
            img_count = 0
            for batch in dataloaders[phase]:
               
                imgs, true_masks, idx = (batch['image'], batch['mask'],
                                                  batch['index'])
                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=mask_type)
               
                for img in range(0,imgs.size(0)):
            
                    #Initialize fat array
                    temp_fat = np.zeros(len(seg_models)+1)
                    
                    #Get conversion rate from pixels to fat
                    temp_org_size = fat_df.loc[fat_df['Image Name (.tif)']==idx[img]]['# of Pixels'].iloc[-1]
                    temp_ds_size = fat_df.loc[fat_df['Image Name (.tif)']==idx[img]]['Down sampled # of Pixels'].iloc[-1]
                    temp_org_rate = fat_df.loc[fat_df['Image Name (.tif)']==idx[img]]['Reference Length (um/px)'].iloc[-1]
        
                    #Compute percentage of fat from ground truth
                    temp_fat[0] = true_masks[img][0].count_nonzero().item() * (temp_org_size/temp_ds_size) * (temp_org_rate)**2
                    img_names.append(idx[img])
                    
                    
                    # Initialize the histogram model for this run
                    for key in seg_models:
                        
                        temp_params = Parameters(model=seg_models[key])
                        
                        model_name = temp_params['Model_names'][temp_params['model_selection']]
                
                        model = initialize_model(model_name, num_classes,
                                                        temp_params)
                        
                        # If parallelized, need to set change model
                        model = nn.DataParallel(model)
                
                        # Send the model to GPU if available
                        model = model.to(device)
                        
                        #Get location of best weights
                        sub_dir, fig_dir = Generate_Dir_Name(split, temp_params)
                        
                        #Load weights for model
                        model.load_state_dict(torch.load(sub_dir + 'best_wts.pt', 
                                                    map_location=torch.device(device)))
                        
                        #Get output and plot
                        model.eval()
                        # pdb.set_trace()
                        with torch.no_grad():
                            preds = model(imgs[img].unsqueeze(0))
                            preds = (torch.sigmoid(preds) > .5).float()
                    
                        torch.cuda.empty_cache()
    
                        #Compute estimated fat
                        temp_fat[key+1] = preds[0].count_nonzero().item() * (temp_org_size/temp_ds_size) * (temp_org_rate/1000)**2
                     
                    #Save fat value for models
                    fat_table.append(temp_fat)
                    img_count += 1
                    print('Finished image {} of {}'.format(img_count,len(dataloaders[phase].sampler)))
                       
            #Create excel spreadsheet
            fat_table = np.array(fat_table)
            add_to_excel(fat_table, writer, model_names, img_names,fold=split)
            print('Finished fold {} of {}'.format(split+1,folds))
                
        #Create Training and Validation folder
        if not os.path.exists(folder):
            os.makedirs(folder)
        writer.save()
        writer.close()
        
    
    
    
    
    
    
    