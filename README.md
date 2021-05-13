# Jointly Optimized Spatial Histogram U-Net Architecture (JOSHUA):
**Jointly Optimized Spatial Histogram U-Net Architecture (JOSHUA) for 
adipose tissue identification in histological images of lyophilized silk 
sponge implants**

![abstract](Figures/Graphical_Abstract_Background.PNG)

_Joshua Peeples, Julie Jameson, Nisha Kotta, Whitney Stoppel, and Alina Zare_

Note: If this code is used, cite it: Joshua Peeples, Julie Jameson, Nisha Kotta, Whitney Stoppel, and Alina Zare. 
(2021, May 11). GatorSense/Histological_Segmentation: Initial Release (Version v1.0). 
Zendo. https://doi.org/10.5281/zenodo.3731417
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3731417.svg)](https://doi.org/10.5281/zenodo.3731417)

[[`arXiv`](https://arxiv.org/abs/2001.00215)]

[[`BibTeX`](#CitingHist)]


In this repository, we provide the paper and code for histological segmentation models from "Jointly Optimized Spatial Histogram U-Net Architecture (JOSHUA) for 
adipose tissue identification in histological images of lyophilized silk 
sponge implants."

## Installation Prerequisites

This code uses python, pytorch, numpy, Pillow, tensorboard, and future. 
Please use [[`Pytorch's website`](https://pytorch.org/get-started/locally/)] to download necessary packages.

## Demo

Run `demo.py` in Python IDE (e.g., Spyder) or command line. To evaluate performance,
run `View_Results.py` (if results are saved out).

## Main Functions

The segmentation models (U-Net, Attention U-Net, U-Net+, JOSHUA, and JOSHUA+)
runs using the following functions. 

1. Prepare dataset(s) for model 

 ```indices = Prepare_Dataloaders(**Parameters)```

2. Intialize model

 ```model = intialize_model(**Parameters)```

3. Train model 

```train_dict = train_net(**Parameters)```

4. Validate and/or test model

```test_dict = eval_net(**Parameters)```


## Parameters
The parameters can be set in the following script:

```Demo_Parameters.py```

## Inventory

```
https://github.com/GatorSense/Histological_Segmetation

└── root dir
    ├── demo.py   //Run this. Main demo file.
    ├── Demo_Parameters.py // Parameters file for demo.
    ├── Prepare_Data.py  // Load data for demo file.
    ├── View_Results.py // Run this after demo to view saved results.
    └── Utils  //utility functions
        ├── capture_metrics.py  // Record evaluation metrics in excel spreadsheet.
        ├── create_dataloaders.py  // Generate Pytorch dataloaders for each dataset.
        ├── create_individual_figures.py  // Create figures to display segmentation results.
        ├── dataset.py  // Load training, validation, and test splits.
        ├── eval.py  // Evaluate models on validation and test data.
        ├── functional.py  // Contains functions to compute evaluation metrics.
        ├── initialize_model.py  // Initialize segmentation model(s).
        ├── RBFHistogramPooling.py  // Create histogram layer. 
        ├── save_results.py  // Save results from demo script.
        ├── train.py  // Train and evaluate model.
        ├── utils.py  // Functions for data augmentation.
        └── models  // individual model parts
            ├── attention_unet_model.py  // Attention U-Net model.
            ├── Histogram_Model.py  // JOSHUA/JOSHUA+ model.
            ├── histunet_parts.py  // Individual components for JOSHUA/JOSHUA+ models.
            ├── unet_model.py  // U-Net model.
            ├── unet_parts.py  // Individual components for U-Net model.
```

## License

This source code is licensed under the license found in the [`LICENSE`](LICENSE) 
file in the root directory of this source tree.

This product is Copyright (c) 2021 J. Peeples, J. Jameson, N. Kotta, W. Stoppel, 
and A. Zare. All rights reserved.

## <a name="CitingHist"></a>Citing Histological Segmentation

If you use the histological segmentation code, please cite the following 
reference using the following entry.

**Plain Text:**

Peeples, J., Jameson, J., Kotta, N., Stoppel, W., & Zare, A. (2021). Jointly 
Optimized Spatial Histogram U-Net Architecture (JOSHUA) for 
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

