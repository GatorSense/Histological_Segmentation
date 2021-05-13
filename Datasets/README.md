# Downloading datasets:

Note: Due to the size of the datasets, the images were not 
upload to the repository. Please follow these instructions
to ensure the code works. If any of these datasets are used,
please cite the appropiate sources (papers, repositories, etc.) as mentioned
on the webpages and provided here.

##  Silk Fibroin Biomaterial Histology Images (SFBHI) [[`BibTeX`](#CitingSFBHI)]

Please download the [`SFBHI_dataset`](https://drive.google.com/drive/folders/1ZYGR7HxrJFIk5V9UIUYWWjcJuAyYkMxA?usp=sharing) 
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
## <a name="CitingSFBHI"></a>Citing SFBHI

If you use the SFBHI dataset, please cite the following reference using the following entry.

**Plain Text:**

Peeples, J., Jameson, J., Kotta, N., Stoppel, W., & Zare, A. (2021). Jointly Optimized Spatial Histogram U-Net Architecture (JOSHUA) for 
adipose tissue identification in histological images of lyophilized silk 
sponge implants. arXiv preprint TBD.

**BibTex:**
```
@article{peeples2021jointly,
  title={Jointly Optimized Spatial Histogram U-Net Architecture (JOSHUA) for 
adipose tissue identification in histological images of lyophilized silk 
sponge implants},
  author={Peeples, Joshua and Jameson, Julie and Kotta, Nisha, and Stoppel, Whitney, and Zare, Alina},
  journal={arXiv preprint TBD},
  year={2021}
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