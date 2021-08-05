# -*- coding: utf-8 -*-
"""
Created on Thurs July 15 11:21:47 2021
Code modified from https://palikar.github.io/posts/pytorch_datasplit/
@author: jpeeples
"""

import logging
from functools import lru_cache

import pandas as pd
import random
import glob
import os

import numpy as np
import pdb


class DataSplitExcel:

    def __init__(self, img_dir, mask_dir, train_split=0.70, val_split=0.15, 
                 random_seed=0,shuffle=False,img_extension='jpg',mask_extension='png'):
        
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.random_seed = random_seed
        
        
        #Get number of files (make sure img and mask are available)
        img_files = glob.glob(os.path.expanduser(self.img_dir + "/*.{}".format(img_extension)))
        mask_files = glob.glob(os.path.expanduser(self.mask_dir + "/*.{}".format(mask_extension)))
        
        #Get files that have both a mask and image
        img_names = [os.path.splitext(x)[0].split('\\')[-1] for x in img_files]
        mask_names = [os.path.splitext(x)[0].split('\\')[-1] for x in mask_files]
        img_files_set = set(img_names)
        intersection = img_files_set.intersection(mask_names)
        intersection = list(intersection)
        self.img_names = intersection
        
        dataset_size = len(intersection)
        
        self.indices = list(range(dataset_size))

        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(self.indices)

     
        self.train_idx, self.val_idx, self.test_idx = np.split(self.indices, 
                                                               [int(len(self.indices)*train_split), 
                                                                int(len(self.indices)*(1-val_split))])
       
       
    @lru_cache(maxsize=4)
    def save_excel(self,folder='folds',fold=0):
        logging.debug('Save image filenames in excel file')
        
        #Get training, validation, and test images
        train_imgs = np.array(self.img_names)[self.train_idx.astype(int)]
        val_imgs = np.array(self.img_names)[self.val_idx.astype(int)]
        test_imgs = np.array(self.img_names)[self.test_idx.astype(int)]
        
        img_splits = {'train': train_imgs, 'valid': val_imgs, 'test': test_imgs}
        
        #Create folder for spread sheeets
        excel_location = self.img_dir + '/' + folder + '/' + 'split_{}'.format(self.random_seed) + '/' + 'fold_{}'.format(fold) + '/'
        
        if not os.path.exists(excel_location):
            os.makedirs(excel_location)
            
        #Save image names in train, validation, and test spreadsheets
        for phase in ['train', 'valid', 'test']:
            imgs = pd.DataFrame(img_splits[phase])
            imgs.to_csv(excel_location+'{}_s_{}_f_{}.csv'.format(phase,self.random_seed,fold),
                        header=False,index=False)

    
if __name__ == '__main__':
    img_path = 'CSAS/raw/'
    mask_path = 'CSAS/Labels/'
  
    split = DataSplitExcel(img_path, mask_path, train_split=0.70, val_split=0.15,
                 random_seed=1,shuffle=True,img_extension='jpg',mask_extension='png')
    
    for folds in range(0,3):
        split.save_excel(fold=folds)
    