# GFS Bias Correction using ML

This project implements a deep learning model in PyTorch for bias correction of GFS (Global Forecast System) temperature forecasts using ERA5 data (ground truth) as a reference. The goal is to reduce forecast biases in GFS 2m temperature forecasts by training a neural network to predict unbiased GFS data based on ERA5 observations.

## Table of Contents

- [Background](#background)
- [Getting Started](#getting-started)
  - [Environment](#environment)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Training](#training)
  - [Bias Correction](#bias-correction)
- [Model Architecture](#model-architecture)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

## Background

The GFS is a widely used weather forecasting system, but it may contain biases in temperature predictions. Bias correction is essential to improve the accuracy of these forecasts. This project presents a PyTorch-based bias correction model that learns to predict unbiased GFS temperature forecasts by comparing them with ERA5 data, which serves as the ground truth.

## Getting Started

### Environment

Set up the environment with Conda:

    conda create -- name ml4bc

### Prerequisites

Before using this project, make sure you have a workspace with the following libraries installed:

- Python 3.x
- PyTorch
- NumPy
- Xarray
- Other required libraries (e.g., Matplotlib, Pandas)

### Installation

In order to get the code, either clone the project, or download a zip from GitHub:

    git clone https://github.com/NOAA-EMC/ML4BC.git

Active the Conda environment:

    conda activate ml4bc

Install all python requirements:

    pip install [requirement]

## Usage

### Data Preparation

- Domain: the contiguous United States (CONUS) subdomain with the longitude expanding from 60°W to 130°W, and the latitude expanding from 25°N to 50°N
- Spatial resolution: 0.25x0.25 degree
- Temporal resolution: 6-hours
- Data period:
    - Training: 23 March 2021 to 31 December 2023
    - Testing: 1 January to 31 May 2024
- GFS: GFS(v16) with 6 hour time intervals for forecast hours 6 to 240 were downloaded from aws s3 bucket using script: script/gen_training_0.25d.py
- ERA5: ERA5 reanalysis dataset with 6 hour time intervals were downloaded from Copernicus Climate Data Store using script: script/gen_training_0.25d.py
- Observations: NCEP ADP Global Surface Weather Observations (BUFR format) were downloaded from the NSF National Center for Atmospheric Research (NCAR) Research Data Archive. Scripts for data processing and pairing with model data are given in scripts/ADPsfc/
- GFS-ERA5 Pairing: GFS forecasts were aligned with ERA5 data over time using script: script/gen_training_0.25d.py

GFS and ERA5 downloading and pairing, submit the jobcard with python:

    python /scripts/jobcard_submission.py

### Training

- Mean squared error (MSE) is used as the loss function
- The maximum epoch was set to 50, but an early stopping criterion was used to stop the training process if the validation loss did not improve in the last 30 epochs
- A learning rate scheduler was applied to reduce the learning rate by a factor of 0.1 when the validation loss did not increase for four epochs. The initial learning rate was set to 0.001
- The Adam optimizer (Kingma & Ba, 2014) is used with default values
- For the training dataset, 90% of the dataset was randomly selected for model training and the remaining 10% of the dataset for validation
- The training was done on NOAA Hera HPC with a single Nvidia Tesla P100

To train the model on Hera with GPU, change the path for the "data_dir" in train_unet.py and set up a batch script to run the train_unet.py

An example batch script named train_unet.sh for executing on NOAA Hera utilizing 8 GPUs:

    #!/bin/bash
    
    process='unet_training'
    
    #SBATCH -A $ACCNT
    #SBATCH -p fge
    #SBATCH -q gpuwf
    #SBATCH -J {process}
    #SBATCH -N 1
    #SBATCH -n 1
    #SBATCH --ntasks-per-node=8
    #SBATCH --time=168:00:00
    #SBATCH -o {process}_output.out
    #SBATCH -e {process}_error.out
    
    # Activate the Conda environment
    conda activate ml4bc
    
    # Execute the script
    python3 train_unet.py

Submit the job:

    sbatch train_unet.sh

![Capture1](https://github.com/user-attachments/assets/bfa78129-34e5-4d62-9732-ee7f5d828760)

### Bias Correction

After training the model, the final model weights were applied to correct the GFS T2m forecasts for lead times ranging from 6 to 240 hours

To do bias correction with the final model weights on Hera, change the path for the "data_dir" in predict_unet.py and set up a batch script to run the predict_unet.py

An example batch script named predict_unet.sh for executing on NOAA Hera utilizing a single CPU:

    #!/bin/bash
    
    process='unet_predict'
    
    #SBATCH -A $ACCNT
    #SBATCH -p hera
    #SBATCH -J {process}
    #SBATCH -N 1
    #SBATCH -n 1
    #SBATCH --cpus-per-task=1
    #SBATCH --time=8:00:00
    #SBATCH -o {process}_output.out
    #SBATCH -e {process}_error.out
    
    # Activate the Conda environment
    conda activate ml4bc
    
    # Execute the script
    python3 predict_unet.py

Submit the job:

    sbatch predict_unet.sh

## Model Architecture

The U-Net is a type of CNN initially developed for biomedical image segmentation (Ronneberger et al., 2015). U-Net performs well on tasks where precise localization and segmentation of features are required. Its u-shaped structure, which includes an encoding (downsampling) path and decoding (upsampling) path, allows it to capture both the context and the detailed features of the input. The encoding path consists of repeated application of convolutions followed by max-pooling operations, which reduce the spatial resolution of the data while increasing the feature sizes. The decoding path, on the other hand, involves upsampling operations that restore the spatial resolution and combine them with the corresponding feature maps from the encoding path through skip connections. This structure allows U-Net to utilize both coarse and fine features for accurate prediction. We kept the number of the downsampling and upsampling blocks the same as the original U-Net. The size of convolution kernels is 3-by-3 with batch normalization (Ioffe and Szegedy, 2015) and leaky rectified linear unit (LeakyReLU, Maas et al., 2013). 

We added the Convolutional Block Attention Module (CBAM, Woo et al., 2018) to the original U-Net. The attention mechanism can identify important features across channels and spatial regions (Trebing et al., 2021). The CBAMs are performed after each double convolution. However, the convoluted image is downsampled along the encoding path, which can keep the original feature. The image with the attention mechanism is reused in the corresponding upsampling part through the skip-connections. We refer to the bias-correction model used in this study as BC-Unet.

![Capture](https://github.com/user-attachments/assets/4bdbcf8a-ea69-4696-965c-d867850da60d)

The architecture of BC-Unet. Arrows represent the operation pass. Blue boxes are two 3-by-3 convolutional layers. Numbers above blue boxes are the number of convolutional channels. Green and red arrows represent max-pooling and up-sampling, respectively. Yellow boxes are refined images after applying CBAM. The black arrows represent layer concatenations.

## Contributing

Users are encouraged to create an issue from [here](https://github.com/NOAA-EMC/ML4BC/issues) for their questions and to create a pull request if they plan to make code updates. The code managers will review and test the code before committing them to the repository.

## License

https://creativecommons.org/publicdomain/zero/1.0/

## Citation 

Cui L., Wang J., Tabas S. Carley J.: A Machine Learning-Based Bias Correction Method for Global Forecast System Products, NOAA Office Note, https://doi.org/10.25923/6266-e822, 2025
