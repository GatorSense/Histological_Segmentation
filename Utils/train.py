import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt

import time
from torch.utils.tensorboard import SummaryWriter
import pickle
import pdb

from .create_dataloaders import Get_Dataloaders
from .functional import *
from .metrics import eval_metrics
from .eval import *
from barbar import Bar
from .decode_segmentation import decode_segmap
from .pytorchtools import EarlyStopping


#Add save option here
def Generate_Dir_Name(split,Network_parameters):
    
    if Network_parameters['hist_model'] is not None:
        filename = (Network_parameters['folder'] + '/' + Network_parameters['mode'] 
                    + '/' + Network_parameters['Dataset'] + '/' 
                    + Network_parameters['hist_model'] + '/Run_' 
                    + str(split + 1) + '/')
        summaryname = (Network_parameters['folder'] + '/' + Network_parameters['mode'] 
                    + '/' + Network_parameters['Dataset'] + '/' 
                    + Network_parameters['hist_model'] + '/Summary/')
    #Baseline model
    else:
        filename = (Network_parameters['folder'] + '/'+ Network_parameters['mode'] 
                    + '/' + Network_parameters['Dataset'] + '/' +
                    Network_parameters['Model_name'] 
                    + '/Run_' + str(split + 1) + '/')   
        summaryname = (Network_parameters['folder'] + '/'+ Network_parameters['mode'] 
                    + '/' + Network_parameters['Dataset'] + '/' +
                    Network_parameters['Model_name']
                    + '/Summary/')   
        
    #Make save directory
    if not os.path.exists(filename):
        os.makedirs(filename)
        
    if not os.path.exists(summaryname):
        os.makedirs(summaryname)
        
    return filename,summaryname

def train_net(net,device,indices,split,Network_parameters,epochs=5,
              batch_size={'train': 1,'val': 1, 'test': 1},lr=0.001,save_cp=True,
              save_results=True,save_epoch=5,dir_checkpoint='checkpoints/'):
    
    dir_name,sum_name = Generate_Dir_Name(split, Network_parameters)
    
    since = time.time()

    val_dice_track = np.zeros(epochs)
   
    n_val = len(indices['val'][split])
    n_train = len(indices['train'][split])
    n_test = len(indices['test'][split])

    dataloaders, pos_wt = Get_Dataloaders(split,indices,Network_parameters,batch_size)

    writer = SummaryWriter(log_dir=sum_name+ 'Run_' +str(split+1))
    
    global_step = 1

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Training Batch size:      {batch_size['train']}
        Validation Batch size:    {batch_size['val']}
        Test Batch size: {batch_size['test']}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Testing size: {n_test}
        Checkpoints:     {save_cp}
        Device:          {device.type}
    ''')

   
    #Set optimizer
    optimizer = optim.Adam(net.parameters(),lr=lr,weight_decay=1e-8)
    
    #Set Early stopping
    early_stopping = EarlyStopping(patience=10, verbose=True)
    
    if torch.cuda.device_count() > 1:
        n_classes = net.module.n_classes
        n_channels = net.module.n_channels
    else:
        n_classes = net.n_classes
        n_channels = net.n_channels
  
    if n_classes > 1:
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
    else:
        #Setting positve weight to highest ratio in training dataset
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_wt).to(device))
       
    best_dice = -np.inf
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)
        
        net.train()
        val_iter_track = []
        
        epoch_loss = 0
        epoch_acc = 0
        best_loss = np.inf
        
        if n_classes > 1:
            epoch_jacc = 0
            epoch_dice = 0
            epoch_mAP = 0
            epoch_class_acc = 0
        else:
            epoch_IOU_pos = 0
            epoch_IOU = 0
            epoch_dice = 0
            epoch_haus_dist = 0
            epoch_prec = 0
            epoch_rec = 0
            epoch_f1_score = 0
            epoch_adj_rand = 0
            inf_samps = 0 #Invalid samples for hausdorff distance
     
        
        for phase in ['train','val']:
            
            if phase == 'train':
                net.train()
        
                for idx, batch in enumerate(Bar(dataloaders[phase])):
                    # pdb.set_trace()
                    imgs = batch['image']
                    true_masks = batch['mask']
                
                    assert imgs.shape[1] == n_channels, \
                        f'Network has been defined with {n_channels} input channels, ' \
                        f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                        'the images are loaded correctly.'
    
                    imgs = imgs.to(device=device, dtype=torch.float32)
                    mask_type = torch.float32 if n_classes == 1 else torch.long
                    true_masks = true_masks.to(device=device, dtype=mask_type)
                    
                    masks_pred = net(imgs)
                    
                    if n_classes > 1:
                        loss = criterion(masks_pred, true_masks) #Target should be NxHxW
                        pred_out = torch.argmax(masks_pred, dim=1)
                      
                    else:
                        loss = criterion(masks_pred, true_masks) #Target should be NxCxHxW
                        
                    #Aggregate loss for epoch
                    epoch_loss += loss.item() * imgs.size(0)
                    del imgs
                    torch.cuda.empty_cache()
                   
                    if n_classes > 1:
                        overall_acc, class_acc, avg_jacc, avg_dice, avg_mAP = eval_metrics(true_masks,masks_pred,n_classes)
                        epoch_jacc += avg_jacc
                        epoch_dice += avg_dice
                        epoch_mAP += avg_mAP
                        epoch_acc += overall_acc
                        epoch_class_acc += class_acc
                    else:
                        pred_out = (torch.sigmoid(masks_pred) > .5).float()
                        temp_haus, temp_haus_count = Average_Metric(pred_out, 
                                                                    true_masks,
                                                                    metric_name='Hausdorff')
                        epoch_haus_dist += temp_haus
                        epoch_dice += dice_coeff(pred_out, true_masks).item()
                            
                        epoch_IOU_pos += Average_Metric(pred_out, true_masks,metric_name='Jaccard')
                        epoch_IOU += Average_Metric(pred_out, true_masks,metric_name='IOU_All')
                        epoch_acc += Average_Metric(pred_out, true_masks,metric_name='Acc')
                        inf_samps += temp_haus_count
                        epoch_prec += Average_Metric(pred_out, true_masks,metric_name='Precision')
                        epoch_rec += Average_Metric(pred_out, true_masks,metric_name='Recall')
                        epoch_f1_score += Average_Metric(pred_out, true_masks,metric_name='F1')
                        epoch_adj_rand += Average_Metric(pred_out, true_masks,metric_name='Rand')
    
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_value_(net.parameters(), 0.1) 
                    optimizer.step()

                total = len(dataloaders[phase].sampler)
                if n_classes > 1:
                    writer.add_scalar('Dice_F1/train', epoch_dice/total, global_step)
                    writer.add_scalar('Jaccard/train',epoch_jacc/total,global_step)
                    writer.add_scalar('Loss/train', epoch_loss, global_step)
                    writer.add_scalar('Pixel_Acc/train',epoch_acc/total,global_step)
                    writer.add_scalar('mAP/train',epoch_mAP/total,global_step)
                
                    
                    print('{} Loss: {:.4f} mAP: {:.4f} F1: {:.4f}'.format(phase, epoch_loss/total, 
                                                                          epoch_mAP/total,epoch_dice/total))   
                
                else:
                    writer.add_scalar('Dice/train', epoch_dice/total, global_step)
                    writer.add_scalar('IOU_pos/train',epoch_IOU_pos/total,global_step)
                    writer.add_scalar('Loss/train', epoch_loss, global_step)
                    writer.add_scalar('Pixel_Acc/train',epoch_acc/total,global_step)
                    writer.add_scalar('Overall_IOU/train',epoch_IOU/total,global_step)
                    writer.add_scalar('HausdorffDistance/train',epoch_haus_dist/(total-inf_samps+1e-7),global_step)
                    writer.add_scalar('adj_rand/train',epoch_adj_rand/total,global_step)
                    writer.add_scalar('precison/train',epoch_prec/total,global_step)
                    writer.add_scalar('recall/train',epoch_rec/total,global_step)
                    writer.add_scalar('f1_score/train',epoch_f1_score/total)
                
                    print('{} Loss: {:.4f} IOU_pos: {:.4f} Dice Coefficient: {:.4f}'.format(phase, 
                                                                                                  epoch_loss/total, 
                                                                                                  epoch_IOU_pos/total,
                                                                                                  epoch_dice/total))   
            else:
                    net.eval()
                        
                    for tag, value in net.named_parameters():
                        if value.requires_grad:
                            tag = tag.replace('.', '/')
                            writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                            writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                      
          
                    val_dict = eval_net(net, dataloaders['val'], device, pos_wt=torch.tensor(pos_wt))
                  
                    if n_classes > 1:
                        logging.info('Validation cross entropy: {}'.format(val_dict['loss']))
                        writer.add_scalar('Dice_F1/val', val_dict['dice'], global_step)
                        writer.add_scalar('Jaccard/val',val_dict['jacc'],global_step)
                        writer.add_scalar('Loss/val', val_dict['loss'], global_step)
                        writer.add_scalar('Pixel_Acc/val',val_dict['pixel_acc'],global_step)
                        writer.add_scalar('mAP/val',val_dict['mAP'],global_step)
                        writer.add_scalar('Class_Acc/val',val_dict['mAP'],global_step)
                    
                        
                        print('{} Loss: {:.4f} mAP: {:.4f} F1: {:.4f} Avg Inf Time: {:4f}s'.format(phase, val_dict['loss'], 
                                                                              val_dict['mAP'],val_dict['dice'],val_dict['inf_time']))   
                    
                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val_dict['dice']))
                        writer.add_scalar('Dice/val', val_dict['dice'], global_step)
                        writer.add_scalar('IOU_pos/val',val_dict['pos_IOU'],global_step)
                        writer.add_scalar('Loss/val', val_dict['loss'], global_step)
                        writer.add_scalar('Pixel_Acc/val',val_dict['pixel_acc'],global_step)
                        writer.add_scalar('Overall_IOU/val',val_dict['overall_IOU'],global_step)
                        writer.add_scalar('HausdorffDistance/val',val_dict['haus_dist'],global_step)
                        writer.add_scalar('adj_rand/val',val_dict['adj_rand'],global_step)
                        writer.add_scalar('precison/val',val_dict['precision'],global_step)
                        writer.add_scalar('recall/val',val_dict['recall'],global_step)
                        writer.add_scalar('f1_score/val',val_dict['f1_score'])
                        
                    #Save images out every save epoch and last epoch
                    # if (epoch % save_epoch) == 0 or epoch == epochs-1:
                    #     writer.add_images('images/original/train', imgs, global_step)
                    #     writer.add_images('images/labeled/train',batch['mask'],global_step)
                    #     writer.add_images('images/original/val', val_imgs, global_step)
                    #     writer.add_images('images/labeled/val',val_batch['mask'],global_step)
                    #     if n_classes == 1:
                    #         writer.add_images('masks/true/train', true_masks, global_step)
                    #         writer.add_images('masks/pred/train', torch.sigmoid(masks_pred) > 0.5, global_step)
                    #         writer.add_images('masks/true/val', val_true_masks, global_step)
                    #         writer.add_images('masks/pred/val', torch.sigmoid(val_masks_pred) > 0.5, global_step)
                    #     else:
                    #         pdb.set_trace()
                    #         writer.add_images('masks/true/train', true_masks.squeeze(), global_step)
                    #         temp_imgs = torch.argmax(masks_pred.squeeze(), dim=0).detach().cpu().numpy()
                    #         writer.add_images('masks/pred/train', decode_segmap(temp_imgs), global_step)
                    #         writer.add_images('masks/true/val', val_true_masks.squeeze(), global_step)
                    #         temp_imgs = torch.argmax(val_masks_pred.squeeze(), dim=0).detach().cpu().numpy()
                    #         writer.add_images('masks/pred/val', decode_segmap(temp_imgs), global_step)
                    
                        val_iter_track.append(val_dict['dice'])
                        global_step += 1
                        print('{} Loss: {:.4f} IOU_pos: {:.4f} Dice Coefficient: {:.4f} Avg Inf Time: {:4f}s'.format(phase, 
                                                                                                  val_dict['loss'], 
                                                                                                  val_dict['pos_IOU'],
                                                                                                  val_dict['dice'],
                                                                                                  val_dict['inf_time']))
                        val_dice_track[epoch] = sum(val_iter_track)/len(val_iter_track)
                        
                    early_stopping(val_dict['loss'], net)
                
        
        #Check dice coefficient and save best model
        # if val_dice_track[epoch] > best_dice:
        #     best_dice = val_dice_track[epoch]
        #     best_wts = net.state_dict()
        #     best_model = net
        if val_dict['loss'] < best_loss:
            best_loss = val_dict['loss']
            best_wts = net.state_dict()
            best_model = net
            val_metrics = val_dict
            
         #Early stop once loss stops improving
        if early_stopping.early_stop:
            print()
            print("Early stopping")
            break
        
        #Save every save_epoch
        if save_cp and (epoch % save_epoch) == 0:
            try:
                os.mkdir(dir_name+dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_name+dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    #Test model on hold out test set
    test_metrics = eval_net(best_model, dataloaders['test'], device, pos_wt=torch.tensor(pos_wt))
    
    
    time_elapsed = time.time() - since
    text_file = open(dir_name+'Run_Time.txt','w')
    n = text_file.write('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    text_file.close()
    text_file = open(dir_name+'Training_Weight.text','w')
    n = text_file.write('Training Positive Weight: ' + str(pos_wt))
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # print('Best Val Dice: ')
    # print(max(val_dice_track))
    
    # val_metrics = eval_net(best_model, dataloaders['val'], device, pos_wt=torch.tensor(pos_wt))
    
    torch.save(best_wts,dir_name+'best_wts.pt')
    output_val = open(dir_name + 'val_metrics.pkl','wb')
    pickle.dump(val_metrics,output_val)
    output_val.close()
    output_test = open(dir_name + 'test_metrics.pkl','wb')
    pickle.dump(test_metrics,output_test)
    output_val.close()
    writer.close()

