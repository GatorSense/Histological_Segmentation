# -*- coding: utf-8 -*-
"""
This pytorch custom dataset was modified from code in this repository:
https://github.com/jeromerony/survey_wsl_histology. Please cite their work:
    
@article{rony2019weak-loc-histo-survey,
  title={Deep weakly-supervised learning methods for classification and localization in histology images: a survey},
  author={Rony, J. and Belharbi, S. and Dolz, J. and Ben Ayed, I. and McCaffrey, L. and Granger, E.},
  journal={coRR},
  volume={abs/1909.03354},
  year={2019}
}
@author: jpeeples 
"""

from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from .utils import check_files
from torchvision.transforms import functional as F
from random import random, randint
from os.path import splitext
from os import listdir
import os
import pdb
from glob import glob
import itertools
import numpy as np
import torch

#Update load_data function
def load_data(samples, resize=None, min_resize=None):
    images = {}
    for image_path in samples:
        image = Image.open(image_path)
        if resize is not None:
            image = image.resize(resize, resample=Image.LANCZOS)
        elif min_resize:
            image = F.resize(image, min_resize, interpolation=Image.LANCZOS)
        images[image_path] = image.copy()
    return images

class PhotoDataset(Dataset):
    def __init__(self, data_path, files, transform, mask_transform, augment=False, 
                 patch_size=None, rotate=False,
                 preload=False, resize=None, min_resize=None,class_label=True,
                 label_path=None,CSAS=False,img_ext='.jpg',mask_ext='.png'):
        self.transform = transform
        self.mask_transform = mask_transform
        self.resize = resize
        self.min_resize = min_resize
        self.augment = augment
        self.patch_size = patch_size
        self.rotate = rotate
        self.masks_dir = label_path
        self.class_label = class_label
        self.CSAS = CSAS
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        
        #Get rgb mapping for CSAS data
        self.rgb = [(86, 115, 181), (132, 167, 77), (77, 77, 77), (141, 202, 207),
                    (211, 150, 202), (209, 205, 75), (255, 196, 52), (51, 69, 83),
                    (145, 58, 219), (58, 69, 219), (80, 139, 48), (164, 38, 41)]
        self.mapping = {tuple(c): t for c, t in zip(self.rgb, range(len(self.rgb)))}

       #Updated from previous loader
        if class_label:
            self.samples = [(os.path.join(data_path,file[0]), os.path.join(data_path,file[1]), 
                         file[2]) for file in files]
        else:
            self.samples = [(os.path.join(data_path,file[0].replace(" ", "")), os.path.join(label_path,file[0].replace(" ", "")), 
                         file[0].replace(" ", "")) for file in files]
            
            self.ids = [splitext(file)[0] for file in listdir(data_path)
                        if (not file.startswith('.') and os.path.isfile(os.path.join(data_path,file)))]
                   
        self.n = len(self.samples)
            

        self.preloaded = False
        if preload:
            self.images = load_data([image_path for image_path, _, _ in self.samples],
                                    resize=resize, min_resize=min_resize)
            self.masks = load_data([mask_path for _, mask_path, _ in self.samples if mask_path != ''],
                                   resize=resize, min_resize=min_resize)
            self.preloaded = True
            print(self.__class__.__name__ + ' loaded with {} images'.format(len(self.images.keys())))

    def __len__(self):
        return len(self.samples)
    

    def __getitem__(self, index):
        image_path, mask_path, label = self.samples[index]
        img_name = image_path.rsplit('/',1)[-1].rsplit('.',1)[0]
        if self.preloaded:
                image = self.images[image_path].convert('RGB')
        else:
            try:
                image = Image.open(image_path).convert('RGB')
            except:
                image = Image.open(image_path+self.img_ext).convert('RGB')
            
            if self.resize is not None:
                image = image.resize(self.resize, resample=Image.LANCZOS)
            elif self.min_resize is not None:
                image = F.resize(image, self.min_resize, interpolation=Image.LANCZOS)
        image_size = image.size # to generate the mask if there is no file

        if mask_path == '':
            mask = Image.new('L', image_size)
        else:
            if self.preloaded:
                mask = self.masks[mask_path].convert('L')
            else:
                try:
                    mask = Image.open(mask_path).convert('L')
                except:
                    mask = Image.open(mask_path+self.mask_ext).convert('RGB')
                    
                if self.resize is not None:
                    mask = mask.resize(self.resize, resample=Image.LANCZOS)
                elif self.min_resize is not None:
                    mask = F.resize(mask, self.min_resize, interpolation=Image.LANCZOS)

        if self.augment:

            # extract patch
            if self.patch_size is not None:
                left = randint(0, image_size[0] - self.patch_size)
                up = randint(0, image_size[1] - self.patch_size)
                image = image.crop(box=(left, up, left + self.patch_size, up + self.patch_size))
                mask = mask.crop(box=(left, up, left + self.patch_size, up + self.patch_size))

            # rotate
            if self.rotate:
                if self.CSAS:
                    angle = randint(0, 71) * 5
                    image = image.rotate(angle)
                    mask = mask.rotate(angle)
                else:
                    angle = randint(0, 3) * 90
                    image = image.rotate(angle)
                    mask = mask.rotate(angle)
                    

            # flip
            if random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        if self.transform is not None:
            image = self.transform(image)
            if self.CSAS:
                h, w = mask.size[0], mask.size[1]
                #Fill rotate image with negative ones if area is result of data augmentation
                class_mask = -torch.ones((h,w),dtype=torch.long)
                temp_mask = torch.from_numpy(np.array(mask)).permute(2, 0, 1).contiguous()
                for k in self.mapping:
                    # Get all indices for current class
                    idx = (temp_mask==torch.tensor(k, dtype=torch.uint8).unsqueeze(1).unsqueeze(2))
                    validx = (idx.sum(0) == 3)
                    class_mask[validx] = torch.tensor(self.mapping[k], dtype=torch.long)
                    
                # plt.figure();plt.imshow(class_mask)
                # plt.figure();plt.imshow(mask)
                # plt.figure();plt.imshow(image.permute(1, 2, 0))
                # pdb.set_trace()
                mask = class_mask
            else:
                mask = self.mask_transform(mask)
                if self.class_label:
                    #Preprocess to be binary
                    mask = (mask != 0).long()
                else:
                    #Clean up masks, software leaves two labels (1,2) for fat
                    mask[mask>=.5] = 1
                    mask[mask<.5] = 0

        return {'image':image,'mask': mask, 'index': img_name, 'label': label}