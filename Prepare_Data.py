# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 18:07:33 2019
Load datasets for models
@author: jpeeples
"""
## Python standard libraries
from __future__ import print_function
from __future__ import division
import csv
import pdb
from os.path import join

#From WSL repo
def csv_reader(fname):
    with open(fname, 'r') as f:
        out = list(csv.reader(f))
    return out

def get_files(folds_dir, split, fold):
    splits = ['train', 'valid', 'test']
    csv_dir = join(folds_dir, 'split_{}'.format(split), 'fold_{}'.format(fold))
    csv_files = [join(csv_dir,  '{}_s_{}_f_{}.csv'.format(s, split, fold)) for s in splits]
    split_files = [csv_reader(csv) for csv in csv_files]
    return split_files

def decode_classes(files: list,class_label=True) -> list:
    if class_label:
        classes = {'benign': 0, 'malignant': 1}
        files_decoded_classes = []
        for f in files:
            class_name = f[2]
            files_decoded_classes.append((f[0], f[1], classes[class_name]))
    else:
        files_decoded_classes = []
        for f in files:
            files_decoded_classes.append((f[0]+'.jpeg',))
    return files_decoded_classes

def Prepare_DataLoaders(Network_parameters, splits):
    Dataset = Network_parameters['Dataset']
    imgs_dir = Network_parameters['imgs_dir']

    # Load datasets
    #Histologial images
    if (Dataset=='SFBHI'):
       #Get files for each fold
       train_indices = []
       val_indices = []
       test_indices = []
       for fold in range(0,splits):
           files = get_files(imgs_dir+'folds',0,fold)
           temp_train, temp_val, temp_test = [decode_classes(f,class_label=False) for f in files]
           train_indices.append(temp_train)
           val_indices.append(temp_val)
           test_indices.append(temp_test)
           
   #Glas Dataset
    elif Dataset == 'GlaS':
       #Get files for each fold
       train_indices = []
       val_indices = []
       test_indices = []
       for fold in range(0,splits):
           files = get_files(imgs_dir+'folds',0,fold)
           temp_train, temp_val, temp_test = [decode_classes(f) for f in files]
           train_indices.append(temp_train)
           val_indices.append(temp_val)
           test_indices.append(temp_test)
    
    #Generate indices (img files) for training, validation, and test
    indices = {'train': train_indices, 'val': val_indices, 'test': test_indices}
    
    return indices