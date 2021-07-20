# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 11:20:44 2019
Parameters for histogram layer experiments
Only change parameters in this file before running
demo.py
@author: jpeeples 
"""
import numpy as np

def Parameters(args):
    ######## ONLY CHANGE PARAMETERS BELOW ########
    #Flag for if results are to be saved out
    #Set to True to save results out and False to not save results
    save_results = args.save_results
    
    #Select model, options include:
    # 'UNET'
    # 'Attention UNET'
    # 'UNET+'
    # 'JOSHUA'
    # 'JOSHUA+'
    model = args.model
    
    seg_models = {'UNET': 0,'UNET+': 1, 'Attention UNET': 2, 'JOSHUA': 3, 'JOSHUA+': 4}
    model_selection = {0: 1, 1: 1, 2: 4, 3: 1, 4: 1}
    hist_skips = {0: False, 1: False, 2: False, 3: True, 4:True}
    attention = {0: False, 1: True, 2: True, 3: False, 4: True}
    
    #Flag for to save out model at certain checkpoints (default: every 5 epochs)
    #Set to True to save results out and False to not save results
    save_cp = args.save_cp
    save_epoch = args.save_epoch
    
    #Location to store trained models
    #Always add slash (/) after folder name
    folder = args.folder
    
    #Flag to use histogram model(s) or baseline UNET model
    # Set either to True to use histogram layer(s) and both to False to use baseline model 
    #Use histogram(s) as attention mechanism, set to True
    histogram_skips = hist_skips[seg_models[model]]
    histogram_pools = False
    use_attention = attention[seg_models[model]]
    
    #Location at which to apply histogram layer(s) for skip connections and/or pooling
    #Will need to set to True to add histogram layer and False to not add histogram layer
    #(default:  all levels, up to 4 different locations)
    skip_locations = [True,True,True,True]
    pool_locations = [True,True,True,True] 
    
    #Select dataset. Set to number of desired segmentation dataset
    data_selection = args.data_selection
    Dataset_names = { 1: 'SFBHI', 2: 'GlaS'}
    
    #If SFBHI, generate images with adipose tissue graphs
    if data_selection == 1:
        show_fat = True
    else:
        show_fat = False
    
    #Number of input channels for each dataset (for now, all are 3 channels-RGB)
    channels = args.channels
    
    Model_name = model
    
    #For upsampling feature maps, set to True to use bilinear interpolation
    #Set to False to learn transpose convolution (consume more memory)
    bilinear = args.bilinear
    
    #Data augmentation (default True)
    #Set to true, training data will be rotated, random flip (p=.5), random patch extraction
    augment = args.augment
    rotate = args.rotate
    
    #Resize the image before center crop. Recommended values for resize is 256 (used in paper)
    #Center crop size is recommended to be 256 (No center crops). 
    resize_size = args.resize_size
    center_size = args.center_size
    
    #Number of folds for K fold CV
    fold_datasets = {1: 4, 2: 5}
    folds = fold_datasets[args.data_selection]
    
    #Set random state for K fold CV for repeatability of data/model initialization
    random_state = args.random_state
    
    #Number of bins for histogram layer. Recommended values are 4, 8 and 16.
    #Set number of bins to powers of 2 (e.g., 2, 4, 8, etc.)
    #Note: a 1x1xK convolution is used to 
    #downsample feature maps before binning process. If the bin values are set
    #higher th, than an error will occur due to attempting to reduce the number of 
    #features maps to values less than one
    numBins = args.numBins
    
    #Flag for feature extraction (fix backbone/encoder). False, train whole model.
    #Flag to add BN to convolutional features (default: False)
    feature_extraction = args.feature_extraction
    add_bn = args.add_bn
    
    #Set initial learning rate for model
    #Recommended values are .001 or .01
    lr = args.lr
    
    #Parameters of Histogram Layer
    #For no padding, set 0. If padding is desired,
    #enter amount of zero padding to add to each side of image 
    #(did not use padding in paper, recommended value is 0 for padding)
    padding = args.padding
    
    
    #Set whether to use sum (unnormalized count) or average pooling (normalized count)
    # (default: average pooling)
    #Set whether to enforce sum to one constraint across bins (default: True)
    normalize_count = args.normalize_count
    normalize_bins = args.normalize_bins
    
    #Batch size for training and epochs. If running experiments on single GPU (e.g., 2080ti),
    #training batch size is recommended to be 4 for SFBHI and 2 for GlaS. If using at least two GPUs, 
    #the recommended training batch size is 8 for SFBHI and 4 for GlaS (as done in paper)
    #May need to reduce batch size if CUDA out of memory issue occurs
    batch_size = {'train': args.train_batch_size, 'val': args.val_batch_size, 
                  'test': args.test_batch_size}
    num_epochs = args.num_epochs
    
    #Pin memory for dataloader (set to True for experiments)
    pin_memory = True
    
    #Set number of workers, i.e., how many subprocesses to use for data loading.
    #Usually set to 0 or 1. Can set to more if multiple machines are used.
    #Number of workers for experiments for two GPUs was three
    num_workers = 0

    #Visualization parameters for figures
    fig_size = 12
    font_size = 16
    
    #Run on multiple GPUs
    parallelize_model = args.parallelize_model
    
    
    ######## ONLY CHANGE PARAMETERS ABOVE ########
    if feature_extraction:
        mode = 'Feature_Extraction'
    else:
        mode = 'Fine_Tuning'
    
    #Location of segmentation datasets (images and masks)
    img_dirs = {'SFBHI': './Datasets/SFBHI/Images/', 
                'GlaS': './Datasets/GlaS/'}
    
    
    #Light directory
    mask_dirs = {'SFBHI': './Datasets/SFBHI/Labels/', 
                 'GlaS': './Datasets/GlaS/'}
        
    #Number of classes in each dataset
    num_classes = {'SFBHI': 1, 
                 'GlaS': 1}
    
    #Number of runs and/or splits for each dataset (5 fold)
    Splits = {'SFBHI': 4, 
              'GlaS': 5}
    
    Dataset = Dataset_names[data_selection]
    imgs_dir = img_dirs[Dataset]
    masks_dir = mask_dirs[Dataset]
    
    if histogram_skips and not(histogram_pools): #Only skip connections
        Hist_model_name = (model + '_' + str(numBins) + 'Bins_Skip_' + 
                           str(np.where(skip_locations)[0]+1))
    elif not(histogram_skips) and histogram_pools: #Only pooling layers
        Hist_model_name = (model + '_' + str(numBins) + 'Bins_Pool_' + 
                           str(np.where(pool_locations)[0]+1))
    elif histogram_skips and histogram_pools: #Both skip and pooling
        Hist_model_name = (model + '_' + str(numBins) + 'Bins_Skip_' +  str(np.where(skip_locations)[0]+1)
                            + '_Pool_' +str(np.where(pool_locations)[0]+1))
    else: #Base UNET model
        Hist_model_name = None
        
    #Return dictionary of parameters
    Network_parameters = {'save_results': save_results,'folder': folder, 
                          'Dataset': Dataset, 'imgs_dir': imgs_dir,
                          'masks_dir': masks_dir,'num_workers': num_workers, 
                          'mode': mode,'lr_rate': lr,
                          'batch_size' : batch_size, 
                          'num_epochs': num_epochs, 'resize_size': resize_size, 
                          'center_size': center_size, 'padding': padding, 
                          'normalize_count': normalize_count, 
                          'normalize_bins': normalize_bins,
                          'numBins': numBins,'Model_name': Model_name,
                          'num_classes': num_classes, 'Splits': Splits, 
                          'feature_extraction': feature_extraction,
                          'hist_model': Hist_model_name,
                          'add_bn': add_bn, 'pin_memory': pin_memory,
                          'folds': folds,'fig_size': fig_size, 'font_size': font_size, 
                          'Parallelize_model': parallelize_model,
                          'histogram_skips': histogram_skips,
                          'histogram_pools': histogram_pools,
                          'skip_locations': skip_locations, 'channels': channels,
                          'pool_locations': pool_locations, 'bilinear': bilinear,
                          'random_state': random_state, 'save_cp': save_cp,
                          'save_epoch': save_epoch, 'use_attention': use_attention,
                          'augment': augment, 'rotate': rotate, 'show_fat': show_fat}
    return Network_parameters
