# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 11:20:44 2019
Parameters for histogram layer experiments
Only change parameters in this file before running
demo.py
@author: jpeeples 
"""
import numpy as np

def Parameters(histogram_skips=True,histogram_pools=True,use_attention=True,
               model_selection=1):
    ######## ONLY CHANGE PARAMETERS BELOW ########
    #Flag for if results are to be saved out
    #Set to True to save results out and False to not save results
    save_results = True
    
    #Flag for to save out model at certain checkpoints (default: every 5 epochs)
    #Set to True to save results out and False to not save results
    save_cp = True
    save_epoch = 5
    
    #Location to store trained models
    folder = 'HPG_Results/Journal_Draft_V3_Results/BCE_Results/'
    
    
    #Flag to use histogram model(s) or baseline UNET model
    # Set either to True to use histogram layer(s) and both to False to use baseline model 
    #Use histogram(s) as attention mechanism, set to True
    histogram_skips = histogram_skips
    histogram_pools = histogram_pools
    use_attention = use_attention
    
    #Location at which to apply histogram layer(s) for skip connections and/or pooling
    #Will need to set to True to add histogram layer and False to not add histogram layer
    #(default:  all levels, up to 4 different locations)
    skip_locations = [True,True,True,True]
    pool_locations = [True,True,True,True] 
    
    #Select dataset. Set to number of desired segmentation dataset
    data_selection = 1
    Dataset_names = { 1: 'Hist_Fat', 2: 'Hist_BFS', 3: 'Stanford_background', 
                      4: 'SIFTFlow_Semantic', 5:'SIFTFlow_Geometric', 6: 'GlaS'}
    
    #Number of input channels for each dataset (for now, all are 3 channels-RGB)
    channels = 3
    
    #Segmentation model to use
    #Histogram layer only implemented for UNET model
    model_selection = model_selection
    if use_attention:
        Model_names = {1: 'UNET_Attention', 
                       2: 'FCN_Attention',
                       3: 'DeepLabv3_Attention',
                       4: 'Attention_UNET'}
    else:
        Model_names = {1: 'UNET', 
                       2: 'FCN',
                       3: 'DeepLabv3'}
    
    #For upsampling feature maps, set to True to use bilinear interpolation
    #Set to False to learn transpose convolution (consume more memory)
    bilinear = True
    
    #Data augmentation (default False)
    #Set to true, training data will be rotated, random flip (p=.5), random patch extraction
    augment = False
    rotate = False
    
    #Resize the image before center crop. Recommended values for resize is 256 (used in paper), 384,
    #and 512 
    #Center crop size is recommended to be 256. Patch extracted from training data
    #For seg
    resize_size = 256
    center_size = 256
    
    #Scale to resize images
    #Scale should be 1 for all datasets except histological dataset (imgs really 
    # large)
    #Scale to resize images
    img_scale = {'Hist_Fat': 1, 'Hist_BFS': .2, 'Stanford_background': 1, 
                      'SIFTFlow_Semantic': 1, 'SIFTFlow_Geometric': 1, 'GlaS': 1}

    
    #Number of folds for K fold CV
    folds = 5
    
    #Set random state for K fold CV for repeatability of data
    random_state = 1
    
    #Number of bins for histogram layer. Recommended values are 4, 8 and 16.
    #Set number of bins to powers of 2 (e.g., 2, 4, 8, etc.)
    #For HistRes_B models using ResNet18 and ResNet50, do not set number of bins
    #higher than 128 and 512 respectively. Note: a 1x1xK convolution is used to 
    #downsample feature maps before binning process. If the bin values are set
    #higher than 128 or 512 for HistRes_B models using ResNet18 or ResNet50 
    #respectively, than an error will occur due to attempting to reduce the number of 
    #features maps to values less than one
    numBins = 16
    
    #Flag for feature extraction (fix backbone/encoder). False, train whole model.
    # True, only update histogram layers and decoder (default: False)
    #Flag to use pretrained model from ImageNet or train from scratch (default: True)
    #Flag to add BN to convolutional features (default:True)
    feature_extraction = False
    use_pretrained = False #Need to fix for UNET/UNET Hist, will not work for FCN/DeepLabV3
    add_bn = False
    
    #Set initial learning rate for model
    #Recommended values are .001 or .01
    lr = .01
    
    #Set momentum for SGD optimizer. 
    #Recommended value is .9 (used in paper)
    alpha = .9
    
    #Parameters of Histogram Layer
    #For no padding, set 0. If padding is desired,
    #enter amount of zero padding to add to each side of image 
    #(did not use padding in paper, recommended value is 0 for padding)
    padding = 0
    
    #Apply rotation to test set (did not use in paper)
    #Set rotation to True to add rotation, False if no rotation (used in paper)
    #Recommend values are between 0 and 25 degrees
    #Can use to test robustness of model to rotation transformation
    rotation = False
    degrees = 25
    
    #Set whether to have the histogram layer inline or parallel (default: parallel)
    #Set whether to use sum (unnormalized count) or average pooling (normalized count)
    # (default: average pooling)
    #Set whether to enforce sum to one constraint across bins (default: True)
    # parallel = False
    normalize_count = True
    normalize_bins = False
    
    #Set step_size and decay rate for scheduler
    #In paper, learning rate was decayed factor of .1 every ten epochs (recommended)
    step_size = 10
    gamma = .1
    
    #Batch size for training and epochs. If running experiments on single GPU (e.g., 2080ti),
    #training batch size is recommended to be 64. If using at least two GPUs, 
    #the recommended training batch size is 128 (as done in paper)
    #May need to reduce batch size if CUDA out of memory issue occurs
    batch_size = {'train': 1, 'val': 1, 'test': 1}
    num_epochs = 2
    
    
    #Pin memory for dataloader (set to True for experiments)
    pin_memory = True
    
    #Set number of workers, i.e., how many subprocesses to use for data loading.
    #Usually set to 0 or 1. Can set to more if multiple machines are used.
    #Number of workers for experiments for two GPUs was three
    num_workers = 0
    
    #Output feature map size after histogram layer
    feat_map_size = 4
    
    #Visualization parameters for figures
    fig_size = 12
    font_size = 16
    
    #Run on multiple GPUs
    Parallelize_model = True
    
    
    ######## ONLY CHANGE PARAMETERS ABOVE ########
    if feature_extraction:
        mode = 'Feature_Extraction'
    else:
        mode = 'Fine_Tuning'
    
    slurmTmpDir = '.'
    #No norm
    img_dirs = {'Hist_Fat': slurmTmpDir + '/Datasets/HistologicalDataset/New_Labels_V2/No_Norm/H&E stiched_resized_{}/'.format(resize_size), 
                'Hist_BFS':'/Datasets/HistologicalDataset/H&E stiched/',
                'Stanford_background': slurmTmpDir + '/Datasets/iccv09Data/images/',
                'SIFTFlow_Semantic': slurmTmpDir + '/Datasets/SiftFlowDataset/Images/'+
                'spatial_envelope_256x256_static_8outdoorcategories/',
                'SIFTFlow_Geometric': slurmTmpDir + '/Datasets/SiftFlowDataset/Images/'+
                'spatial_envelope_256x256_static_8outdoorcategories/',
                'GlaS': slurmTmpDir + '/Datasets/GlaS/'}
    
  
    # '/Datasets/HistologicalDataset/Hand-Labeled Images_Nisha/'
    #'/Datasets/HistologicalDataset/New_Labels_V2/No_Norm/New_GT_Resized_{}/'.format(resize_size)
    mask_dirs = {'Hist_Fat': slurmTmpDir + '/Datasets/HistologicalDataset/New_Labels_V2/No_Norm/New_GT_Resized_{}/'.format(resize_size), 
             'Hist_BFS': '/Datasets/HistologicalDataset/BFS_masks/',
             'Stanford_background': '/Datasets/iccv09Data/labels/',
             'SIFTFlow_Semantic': '/Datasets/SiftFlowDataset/SemanticLabels/'+
             'spatial_envelope_256x256_static_8outdoorcategories/',
             'SIFTFlow_Geometric': '/Datasets/SiftFlowDataset/SemanticLabels/'+
             'spatial_envelope_256x256_static_8outdoorcategories/',
             'GlaS': slurmTmpDir + '/Datasets/GlaS/'}
        
    #light directory
    # label_dirs = {'Hist_Fat': '/Datasets/HistologicalDataset/New_Labels_V2/Match_Histogram_Dark/New_GT_Resized_256/'}
    label_dirs = {'Hist_Fat': slurmTmpDir + '/Datasets/HistologicalDataset/New_Labels_V2/No_Norm/New_GT_Resized_{}/'.format(resize_size)}

    
    
    #Number of classes in each dataset
    num_classes = {'Hist_Fat': 1, 
                 'Hist_BFS': 3,
                 'Stanford_background': 8,
                 'SIFTFlow_Semantic': 33,
                 'SIFTFlow_Geometric': 3,
                 'GlaS': 1
                 }
    
    
    #Number of runs and/or splits for each dataset (5 fold)
    Splits = {'Hist_Fat': 5, 
              'Hist_BFS': 5,
              'Stanford_background': 5,
              'SIFTFlow_Semantic': 5,
              'SIFTFlow_Geometric': 5,
              'GlaS': 5
                 }
    
    Dataset = Dataset_names[data_selection]
    imgs_dir = img_dirs[Dataset]
    masks_dir = mask_dirs[Dataset]
    
    if histogram_skips and not(histogram_pools): #Only skip connections
        Hist_model_name = ('HistUNET_' + str(numBins) + 'Bins_Skip_' + 
                           str(np.where(skip_locations)[0]+1))
    elif not(histogram_skips) and histogram_pools: #Only pooling layers
        Hist_model_name = ('HistUNET_' + str(numBins) + 'Bins_Pool_' + 
                           str(np.where(pool_locations)[0]+1))
    elif histogram_skips and histogram_pools: #Both skip and pooling
        Hist_model_name = ('HistUNET_' + str(numBins) + 'Bins_Skip_' +  str(np.where(skip_locations)[0]+1)
                            + '_Pool_' +str(np.where(pool_locations)[0]+1))
    else: #Base UNET model
        Hist_model_name = None
        
    if use_attention and Hist_model_name is not None:
        Hist_model_name = Hist_model_name + '_Attention'
    
    #Return dictionary of parameters
    Network_parameters = {'save_results': save_results,'folder': folder, 
                          'Dataset': Dataset, 'imgs_dir': imgs_dir,'label_dirs': label_dirs,
                          'masks_dir': masks_dir,'num_workers': num_workers, 
                          'mode': mode,'lr_rate': lr,
                          'momentum': alpha, 'step_size': step_size,
                          'gamma': gamma, 'batch_size' : batch_size, 
                          'num_epochs': num_epochs, 'resize_size': resize_size, 
                          'center_size': center_size, 'padding': padding, 
                          'normalize_count': normalize_count, 
                          'normalize_bins': normalize_bins,
                          'numBins': numBins,'feat_map_size': feat_map_size,
                          'Model_names': Model_names, 'model_selection': model_selection,
                          'num_classes': num_classes, 'Splits': Splits, 
                          'feature_extraction': feature_extraction,
                          'hist_model': Hist_model_name, 'use_pretrained': use_pretrained,
                          'add_bn': add_bn, 'pin_memory': pin_memory,
                          'degrees': degrees, 'rotation': rotation, 
                          'img_scale': img_scale, 'folds': folds,
                          'fig_size': fig_size, 'font_size': font_size, 
                          'Parallelize_model': Parallelize_model,
                          'histogram_skips': histogram_skips,
                          'histogram_pools': histogram_pools,
                          'skip_locations': skip_locations, 'channels': channels,
                          'pool_locations': pool_locations, 'bilinear': bilinear,
                          'random_state': random_state, 'save_cp': save_cp,
                          'save_epoch': save_epoch, 'use_attention': use_attention,
                          'augment': augment, 'rotate': rotate}
    return Network_parameters