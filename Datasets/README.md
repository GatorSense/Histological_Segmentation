# Downloading datasets:

Note: Due to the size of the datasets, the images were not 
upload to the repository. Please follow these instructions
to ensure the code works. If any of these datasets are used,
please cite the appropiate sources (papers, repositories, etc.) as mentioned
on the webpages and provided here.

##  Silk Fibroin Biomaterial Histology Images (SFBHI) [[`BibTeX`](#CitingSFBHI)]

Please download the [`SFBHI_dataset`](https://drive.google.com/drive/folders/1Csqh6W_i-7vbHiS6_9shZj-4a0M-qVsZ?usp=sharing) 
and follow these instructions:

1. Download and unzip the folder `SFBHI` (Note: this folder contains the 
256x256 images, the full sized images are available in the drive as well.)
2. The structure of the `SFBHI` folder is as follows:
```
└── root dir
    ├── Images   // Contains images and training/validation folds.
        ├── folds
            ├── split_0
                ├── fold_0
                ├── fold_1
                ├── fold_2
                ├── fold_3
                ├── fold_4
    ├── Labels  // Contains labels for each training/validation fold.   
```
## Generate new data splits:
If new data splits are desired for the SFBHI dataset, please use `Split_SFBHI.py`
to generate new paritions of the dataset.

## <a name="CitingSFBHI"></a>Citing SFBHI

If you use the SFBHI dataset, please cite the following reference using the following entry.

**Plain Text:**

Peeples, J. K., Jameson, J. F., Kotta, N. M., Grasman, J. M., Stoppel, W. L., & Zare, A. Jointly 
Optimized Spatial Histogram UNET Architecture (JOSHUA) for Adipose Tissue 
Segmentation. bioRxiv, 2021. doi: 10.1101/2021.11.22.469463.  

**BibTex:**
```
@article {Peeples2021.11.22.469463,
	author = {Peeples, Joshua K and Jameson, Julie F and Kotta, Nisha M and Grasman, Jonathan M and Stoppel, Whitney L and Zare, Alina},
	title = {Jointly Optimized Spatial Histogram UNET Architecture (JOSHUA) for Adipose Tissue Segmentation},
	elocation-id = {2021.11.22.469463},
	year = {2021},
	doi = {10.1101/2021.11.22.469463},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2021/11/23/2021.11.22.469463},
	eprint = {https://www.biorxiv.org/content/early/2021/11/23/2021.11.22.469463.full.pdf},
	journal = {bioRxiv}
}
```
## Gland Segmentation in Colon Histology Images (GlaS) [[`BibTeX`](#CitingGlaS)]

Please download the 
[`GlaS dataset`](https://warwick.ac.uk/fac/cross_fac/tia/data/glascontest/download/) 
and follow these instructions:

1. Download and unzip the file
2. Name the folder `GlaS`
3. The structure of the `GlaS` folder is as follows:
```
└── root dir
    ├── folds   // Contains folders of training and test images.
        ├── split_0
            ├── fold_0
            ├── fold_1
            ├── fold_2
            ├── fold_3
            ├── fold_4
    ├── grade.csv // Contains information for each image.  
```
## <a name="CitingGlaS"></a>Citing GlaS

If you use the GlaS dataset, please cite the following references using the following entry.

**Plain Text:**

Sirinukunwattana, K., Pluim, J. P., Chen, H., Qi, X., Heng, P. A., Guo, 
Y. B., ... & Rajpoot, N. M. (2017). Gland segmentation in colon histology 
images: The glas challenge contest. Medical image analysis, 35, 489-502.

Sirinukunwattana, K., Snead, D. R., & Rajpoot, N. M. (2015). A stochastic 
polygons model for glandular structures in colon histology images. 
IEEE transactions on medical imaging, 34(11), 2366-2378

**BibTex:**
```
@article{sirinukunwattana2017gland,
  title={Gland segmentation in colon histology images: The glas challenge contest},
  author={Sirinukunwattana, Korsuk and Pluim, Josien PW and Chen, Hao and Qi, Xiaojuan and Heng, Pheng-Ann and Guo, Yun Bo and Wang, Li Yang and Matuszewski, Bogdan J and Bruni, Elia and Sanchez, Urko and others},
  journal={Medical image analysis},
  volume={35},
  pages={489--502},
  year={2017},
  publisher={Elsevier}
}

@article{sirinukunwattana2015stochastic,
  title={A stochastic polygons model for glandular structures in colon histology images},
  author={Sirinukunwattana, Korsuk and Snead, David RJ and Rajpoot, Nasir M},
  journal={IEEE transactions on medical imaging},
  volume={34},
  number={11},
  pages={2366--2378},
  year={2015},
  publisher={IEEE}
}

```
