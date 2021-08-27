# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 14:35:49 2021
Generate Dataloaders
@author: jpeeples
"""
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
from .utils import ExpandedRandomSampler
from .dataset import PhotoDataset
import numpy as np
import random

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32 
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def Get_Dataloaders(split,indices,Network_parameters,batch_size):
    
    
    if Network_parameters['Dataset'] == 'GlaS':
        train_loader, val_loader, test_loader = load_glas(Network_parameters['imgs_dir'],
                                                          indices,batch_size,
                                                          Network_parameters['num_workers'],
                                                          split=split,
                                                          augment=Network_parameters['augment'],
                                                          rotate=Network_parameters['rotate'])
        pos_wt = 1
        
    else:
        train_loader, val_loader, test_loader = load_SFBHI(Network_parameters['imgs_dir'],
                                                          indices,batch_size,
                                                          Network_parameters['num_workers'],
                                                          split=split,
                                                          augment=Network_parameters['augment'],
                                                          rotate=Network_parameters['rotate'],
                                                          patch_size=Network_parameters['center_size'],
                                                          label_path=Network_parameters['masks_dir'])
       
        #Get postive weight (for histological fat images only)
        pos_wt = 3
       
    dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    
    return dataloaders, pos_wt
    
def load_glas(data_path,indices, batch_size, num_workers, pin_memory=True,
              split=0, patch_size=416,sampler_mul=8, augment=False, rotate=False):
  
    test_transform = transforms.ToTensor()

    train_loader = DataLoader(
        PhotoDataset(
            data_path= data_path,
            files=indices['train'][split],
            patch_size=patch_size,
            # augment=augment,
            # rotate=rotate,
            transform=transforms.Compose([
                #transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.05),
                transforms.ToTensor(),
            ]),
            mask_transform=transforms.ToTensor(),
            preload=False,
        ),
        batch_size=batch_size['train'],
        num_workers=num_workers,
        #sampler=ExpandedRandomSampler(len(indices['train'][split]), sampler_mul),
        pin_memory=pin_memory,
        drop_last=True,worker_init_fn=seed_worker
    )
    valid_loader = DataLoader(
        PhotoDataset(data_path= data_path, files=indices['val'][split],
                     transform=test_transform, mask_transform=test_transform, preload=False),
        batch_size=1, num_workers=num_workers,
        shuffle=False,
        pin_memory=pin_memory,worker_init_fn=seed_worker
    )
    test_loader = DataLoader(
        PhotoDataset(data_path= data_path, files=indices['test'][split],
                     transform=test_transform, mask_transform=test_transform, preload=False),
        batch_size=1, num_workers=num_workers,
        pin_memory=pin_memory,worker_init_fn=seed_worker
    )
    
    return train_loader, valid_loader, test_loader

def load_SFBHI(data_path,indices, batch_size, num_workers, pin_memory=True,
              split=0, patch_size=416,sampler_mul=8, augment=False, rotate=False,
              label_path=None):
    test_transform = transforms.ToTensor()
    
    train_loader = DataLoader(
        PhotoDataset(
            data_path= data_path,
            files=indices['train'][split],
            patch_size=patch_size,
            # augment=augment,
            # rotate=rotate,
            transform=transforms.Compose([
                # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.05),
                transforms.ToTensor(),
            ]),
            mask_transform=transforms.ToTensor(),
            preload=False,class_label=False,label_path=label_path
        ),
        batch_size=batch_size['train'],
        num_workers=num_workers,
        # sampler=ExpandedRandomSampler(len(indices['train'][split]), sampler_mul),
        pin_memory=pin_memory,
        drop_last=True,worker_init_fn=seed_worker
    )
    valid_loader = DataLoader(
        PhotoDataset(data_path= data_path, files=indices['val'][split],
                     transform=test_transform, mask_transform=test_transform, preload=False,
                     class_label=False,label_path=label_path),
        batch_size=batch_size['val'], num_workers=num_workers,
        shuffle=False,
        pin_memory=pin_memory,worker_init_fn=seed_worker
    )
    test_loader = DataLoader(
        PhotoDataset(data_path= data_path, files=indices['test'][split],
                     transform=test_transform, mask_transform=test_transform, preload=False,
                     class_label=False,label_path=label_path),
        batch_size=batch_size['test'], num_workers=num_workers,
        pin_memory=pin_memory,worker_init_fn=seed_worker
    )
    
    return train_loader, valid_loader, test_loader