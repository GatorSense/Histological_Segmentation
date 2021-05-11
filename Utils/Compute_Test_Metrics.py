# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 15:09:46 2021
Get test metrics for model (model not properly used on HPG)
@author: jpeeples
"""
## Python standard libraries
from __future__ import print_function
import numpy as np
from sklearn.metrics import jaccard_score as jsc
from sklearn.metrics import adjusted_rand_score as arsc
# from skimage import metrics
import pandas as pd
import pdb
import pickle
import time

## PyTorch dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from View_Results_Parameters import Parameters
from Utils.Initialize_Model import initialize_model
from Utils.PytorchUNet.create_dataloaders import Get_Dataloaders
from Utils.functional import *
import pdb


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

def Get_Test_Metrics(dataset,indices,mask_type,seg_models,device,hist_skips,
                     hist_pools,attention,model_selection,num_classes,folds=5,data_type='val'):

    model_names = []
    
    #Set names of models
    for key in seg_models:
        model_names.append(seg_models[key])
    
    # Initialize the histogram model for this run
    for split in range(0,folds):
        
        #Generate dataloaders and pos wt
        temp_params = Parameters()
        dataloaders, pos_wt = Get_Dataloaders(dataset,split,indices,temp_params,
                                              temp_params['batch_size'])
        for key in seg_models:
            
            temp_params = Parameters(histogram_skips=hist_skips[key],
                                     histogram_pools=hist_pools[key],
                                     use_attention=attention[key],
                                     model_selection=model_selection[key])
            
            #Get location of best weights
            sub_dir, _ = Generate_Dir_Name(split, temp_params)
                
            model_name = temp_params['Model_names'][temp_params['model_selection']]
            model = initialize_model(model_name, num_classes,
                                           temp_params)
            
            # If parallelized, need to set change model
            if temp_params['Parallelize_model']:
                model = nn.DataParallel(model)
            
            # Send the model to GPU if available
            model = model.to(device)
            
            #Get location of best weights
            sub_dir, _ = Generate_Dir_Name(split, temp_params)
            
            #Load weights for model and get test metrics
            best_wts = torch.load(sub_dir + 'best_wts.pt', map_location=torch.device(device))
            if temp_params['Dataset'] == 'Hist_Fat':
                pos_wt = 3
            else:
                pos_wt = 1
                
            test_metrics = eval_net(model, dataloaders[data_type], 
                                                  device,pos_wt=torch.tensor(pos_wt),
                                                  best_wts=best_wts)
            #Save results
            output_test = open(sub_dir + 'new_{}_metrics.pkl'.format(data_type),'wb')
            pickle.dump(test_metrics,output_test)
            output_test.close()
        
        print('Finished fold {} of {}'.format(split+1, folds))
                
                    
def eval_net(net, loader, device,pos_wt=torch.tensor(1),best_wts=None):
    """Evaluation without the densecrf with the dice coefficient"""
    if best_wts is not None:
        net.load_state_dict(best_wts)
    net.eval()
    mask_type = torch.float32 if net.module.n_classes == 1 else torch.long
    # pdb.set_trace()
    n_val = 0
    # n_val = len(loader)  # the number of batches
    tot = 0
    jacc_score = 0
    loss = 0
    inf_time = 0
    iou_score = 0
    val_acc = 0
    haus_dist = 0
    haus_count = 0
    prec = 0
    rec = 0
    spec = 0
    f1_score = 0
    adj_rand = 0
    #Intialize Hausdorff Distance object
    hausdorff_pytorch = HausdorffDistance()

    for batch in loader:
        imgs, true_masks = batch['image'], batch['mask']
        imgs = imgs.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=mask_type)

        with torch.no_grad():
            temp_start_time = time.time()
            mask_pred = net(imgs)
            temp_end_time = (time.time() - temp_start_time)/imgs.size(0)
            inf_time += temp_end_time
            
        if net.module.n_classes > 1:
            tot += F.cross_entropy(mask_pred, true_masks.squeeze(1)).item()
            prec += precision(mask_pred,true_masks).item()
            rec += recall(mask_pred,true_masks).item()
            f1_score += f_score(mask_pred,true_masks).item()
            jacc_score += iou(mask_pred,true_masks)
            temp_preds = torch.max(mask_pred,2).cpu().numpy().reshape(-1)
            temp_masks = torch.max(true_masks,2).cpu().numpy().reshape(-1)
            haus_dist += hausdorff_distance(temp_masks,temp_preds).item()
            adj_rand += arsc(temp_masks,temp_preds)
            n_val += true_masks.size(0)
            
        else:
            pred = torch.sigmoid(mask_pred)
            loss += Average_Metric(pred,true_masks,pos_wt=pos_wt.to(device),metric_name='BCE')
            # loss += Average_Metric(pred,true_masks,metric_name='Dice_Loss')
            # class_probs.append(F.softmax())
            pred = (pred > 0.5).float()
            prec += Average_Metric(pred, true_masks,metric_name='Precision')
            rec += Average_Metric(pred, true_masks,metric_name='Recall')
            f1_score += Average_Metric(pred, true_masks,metric_name='F1') 
            spec += Average_Metric(pred, true_masks,metric_name='Specificity')
            temp_haus, temp_haus_count = Average_Metric(pred, true_masks,metric_name='Hausdorff')
            haus_dist += temp_haus
            haus_count += temp_haus_count
            jacc_score += Average_Metric(pred, true_masks,metric_name='Jaccard')
            tot += dice_coeff(pred, true_masks).item()
            adj_rand += Average_Metric(pred, true_masks,metric_name='Rand')
            iou_score += Average_Metric(pred, true_masks,metric_name='IOU_All')
            val_acc += Average_Metric(pred, true_masks,metric_name='Acc')
            n_val += true_masks.size(0)
            

    return {'dice': tot / n_val,'pos_IOU': jacc_score / n_val,
            'loss': (loss / n_val).cpu().numpy(),'inf_time': inf_time / n_val,
            'overall_IOU': iou_score/ n_val,'pixel_acc': val_acc / n_val,
            'haus_dist': haus_dist / (n_val-haus_count),'adj_rand': adj_rand / n_val,
            'precision': prec / n_val,'recall': rec / n_val,
            'f1_score': f1_score / n_val, 'specificity': spec / n_val}

class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 1e-7
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    # return s / (i + 1)
    return s

def Average_Metric(input,target,pos_wt=None,metric_name='Prec'):
    """Metrics for batches"""
    s = 0
    haus_count = 0
    hausdorff_pytorch = HausdorffDistance()

    for i, c in enumerate(zip(input, target)):
        if metric_name == 'Precision':
            s +=  precision(c[1],c[0]).item()
        elif metric_name == 'Recall':
            s += recall(c[1],c[0]).item()
        elif metric_name == 'F1':
            s += f_score(c[1],c[0]).item()
        elif metric_name == 'Hausdorff':
            temp_haus = hausdorff_pytorch.compute(c[1].unsqueeze(0),c[0].unsqueeze(0)).item()
            if temp_haus == np.inf: #If output does not have positive class, do not include in avg (GT has few positive ROI) 
                haus_count +=1
            else:
                s += temp_haus
        elif metric_name == 'Jaccard':
            s += iou(c[1],c[0]).item()
        elif metric_name == 'Rand':
            s += arsc(c[1].cpu().numpy().reshape(-1).astype(int),
                      c[0].cpu().numpy().reshape(-1).astype(int))
        elif metric_name == 'IOU_All': #Background
            s += jsc(c[1].cpu().numpy().reshape(-1).astype(int),
                      c[0].cpu().numpy().reshape(-1).astype(int),average='macro')
        elif metric_name == 'Acc':
            s += 100*np.sum(c[1].cpu().numpy().reshape(-1).astype(int) == 
                            c[0].cpu().numpy().reshape(-1).astype(int))/(len(c[0].cpu().numpy().reshape(-1).astype(int)))
        elif metric_name == 'BCE':
            s += F.binary_cross_entropy_with_logits(c[0],c[1],
                                                    pos_weight=pos_wt)
        elif metric_name == 'Specificity':
            s += specificity(c[1],c[0]).item()
        elif metric_name == 'Dice_Loss':
            s += 1-f_score(c[1],c[0]).item()
        else:
            raise RuntimeError('Metric is not implemented')
        
    if metric_name == 'Hausdorff':
        return s, haus_count
    else:
        return s



