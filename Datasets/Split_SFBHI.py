# -*- coding: utf-8 -*-
"""
Created on Thurs Aug 18 11:21:47 2021
Code modified from https://palikar.github.io/posts/pytorch_datasplit/
@author: jpeeples
"""

import logging
from functools import lru_cache

import pandas as pd
import glob
import os
from os import path
from sklearn.model_selection import StratifiedKFold

import numpy as np
import pdb


class DataSplitExcel:

    def __init__(self, img_dir, mask_dir, train_split=0.70, val_split=0.15, 
                 random_seed=0,shuffle=False,img_extension='jpg',mask_extension='png',
                 CV=True,folds=5,label_file='Labels.csv',label_selection='time'):
        
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.random_seed = random_seed
        self.shuffle = shuffle
        self.label_selection = label_selection
        
        #Get number of files (make sure img and mask are available)
        img_files = glob.glob(os.path.expanduser(self.img_dir + "/*.{}".format(img_extension)))
        mask_files = glob.glob(os.path.expanduser(self.mask_dir + "/*.{}".format(mask_extension)))
        
        #Get files that have both a mask and image
        img_file_names = [os.path.splitext(x)[0].split('\\')[-1] for x in img_files]
        mask_names = [os.path.splitext(x)[0].split('\\')[-1] for x in mask_files]
        img_files_set = set(img_file_names)
        mask_img_inter = img_files_set.intersection(mask_names)
        mask_img_inter = list(mask_img_inter)
        
        #Check label file for images to that have condition
        basepath = path.dirname(__file__)
        filepath = path.abspath(path.join(basepath, "..", label_file))
        df = pd.read_excel(filepath)
        
        #Remove white spaces
        df.columns = df.columns.str.rstrip()
        df = df.apply(lambda x: x.str.rstrip() if x.dtype == "object" else x)
        img_list = df['Image Name'].tolist()
        
        #Do not need this anymore, verified image names match excel sheet given
        
        # #Remove 'scale' to be consistent with names of images from excel file and image
        img_list_new = [os.path.splitext(x)[0].replace(" ", "").split('-scale')[0] for x in img_list]
        img_names_new = [os.path.splitext(x)[0].replace(" ", "").split('-scale')[0] for x in mask_img_inter]
        
        #Remove images that we do not want to consider
        #Keep track of indices from total images
        ind_dict = dict((k,i) for i,k in enumerate(img_names_new))
        inter_names = set(ind_dict).intersection(img_list_new)
        indices = [ ind_dict[x] for x in inter_names ]
        
        #Check which images are not included (no condition or week provided)
        # diff_imgs = list(set(img_list_new) - (inter_names))
        
        #Map valid image names back to existing files
        img_names = [mask_img_inter[x] for x in indices]
        self.img_names = img_names
        
        img_week = []
        img_condition = []
        
        for x in self.img_names:
            try:
                img_week.append(df[df['Image Name'] == x.split('-scale')[0]]['Week'].tolist()[0])
                img_condition.append(df[df['Image Name'] == x.split('-scale')[0]]['Condition - Letter'].tolist()[0])
            except:
                img_week.append(df[df['Image Name'] == x]['Week'].tolist()[0])
                img_condition.append(df[df['Image Name'] == x]['Condition - Letter'].tolist()[0])
                
        # self.img_names = img_names
        dataset_size = len(self.img_names)
        
        self.indices = list(range(dataset_size))

        #Shuffle data if desired
        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(self.indices)
            
        #Split data based on weeks
        if label_selection == 'time':
            self.img_labels = img_week
        elif label_selection == 'condition':
            self.img_labels = img_condition
        else:
            self.label_selection = 'random'
                
        self.img_week = img_week
        self.img_condition = img_condition

        #Perform strastified k-fold CV or train/val/test split for dataset
        if CV:
            #Get labels and sort based on image/mask files
            skf = StratifiedKFold(n_splits=folds,shuffle=self.shuffle, 
                                  random_state=self.random_seed)
            self.train_idx = []
            self.val_idx = []
            for train_index, val_index in skf.split(self.img_names, self.img_labels):
                self.train_idx.append(train_index)
                self.val_idx.append(val_index)
            self.test_idx = self.val_idx
        
        else:
            self.train_idx, self.val_idx, self.test_idx = np.split(self.indices, 
                                                               [int(len(self.indices)*train_split), 
                                                                int(len(self.indices)*(1-val_split))])
       
       
    @lru_cache(maxsize=4)
    def save_excel(self,folder='folds',fold=0):
        logging.debug('Save image filenames in excel file')
        
        #Get training, validation, and test images
        data = np.array([self.img_names,self.img_week,self.img_condition]).T
        train_imgs = data[self.train_idx[fold].astype(int)]
        val_imgs = data[self.val_idx[fold].astype(int)]
        test_imgs = data[self.test_idx[fold].astype(int)]
        
        img_splits = {'train': train_imgs, 'valid': val_imgs, 'test': test_imgs}
        
        #Create folder for spread sheeets
        excel_location = self.img_dir + '/' + folder + '/' + '{}_split_{}'.format(self.label_selection,self.random_seed) + '/' + 'fold_{}'.format(fold) + '/'
        
        if not os.path.exists(excel_location):
            os.makedirs(excel_location)
            
        #Save image names in train, validation, and test spreadsheets
        for phase in ['train', 'valid', 'test']:
            imgs = pd.DataFrame(img_splits[phase])
            imgs.to_csv(excel_location+'{}_s_{}_f_{}.csv'.format(phase,self.random_seed,fold),
                        header=False,index=False)

    
if __name__ == '__main__':
    img_path = 'SFBHI/Images/'
    mask_path = 'SFBHI/Labels/'
    label_path = 'Image Name, Week, and Condition.xlsx'
    k_fold = 5
    random_seed = 1
  
    # split = DataSplitExcel(img_path, mask_path, train_split=0.70, val_split=0.15,
    #              random_seed=random_seed,shuffle=True,img_extension='jpeg',mask_extension='jpeg',
    #              CV=True,folds=k_fold,label_file=label_path,label_selection='time')
    
    split = DataSplitExcel(img_path, mask_path, train_split=0.70, val_split=0.15,
                  random_seed=random_seed,shuffle=True,img_extension='jpeg',mask_extension='jpeg',
                  CV=True,folds=k_fold,label_file=label_path,label_selection='condition')
    
    for folds in range(0,k_fold):
        split.save_excel(fold=folds)
    