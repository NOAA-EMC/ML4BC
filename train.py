'''
Description: Main code for calling the ML4BC Machine Learning model, an Autoencoder Conv3D model, which is developed and designed for 
GFS 2m temperature bias correction. The model training inputs were prepared using preprocessing.py from GFS (biased) and ERA5 (unbiased) 
data from 20210321 to 20231018 in every 6 hours. The data has 0.25-degree spatial resolution and 50 hourly timesteps (e.g., [721,1440,50]).

In summary, this model provides functionalities, including:
(i) preprocessing.py: a utility for preparing ML4BC model inputs from 0.25-degree resolution GFS and ERA5 data. This Python utility has
two modules one for GFS info and the second one for era5 info. 
(ii) autoencoder_model.py: Main model structure.
(iii) netcdf_dataset.py: Provide functionalities for data processing, normalizing, rescaling and making pytorch dataloader for both GFS and ERA5.
(iv) check_missing_files: A function for checking missing files.
(v) calculate_mean_and_std: A function to calculate mean and standard deviation of training dataset which provides values for normalization and rescaling modules.
(vi) ml4bc.py: Model Initiation, Training Loop, Module for Saving Model State
(vii) ml4bc.ipynb: An example of the ML4BC modeling process.
(iix) postprocessing.ipynb: A notebook for postprocessing including data sanity check, plotting, and data analysis.
    
Author: Sadegh Sadeghi Tabas (sadegh.tabas@noaa.gov)
Revision history:
    -20231029: Sadegh Tabas, initial code
'''

import os
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from tqdm import tqdm
from autoencoder_model import get_autoencoder
from netcdf_dataset import NetCDFDataset, check_missing_files, calculate_mean_and_std
from datetime import date

def train_one_epoch(epoch_index, X, y):
    running_loss = 0.
    last_loss = 0.

    # Create a custom progress bar for the epoch
    progress_bar = tqdm(enumerate(zip(X, y)), total=len(X), desc=f'Epoch [{epoch+1}/{EPOCHS}]', dynamic_ncols=True)
    for i, (gfs_data, era5_data) in progress_bar:
    #for i, (gfs_data, era5_data) in enumerate(zip(X, y)):

        # Zero gradients for every batch
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(gfs_data.to(device))

        #Compute the loss and its gradients
        loss = loss_fn(outputs, era5_data.to(device))
        loss.backward()

        #Adjust learning weights
        optimizer.step()

        running_loss += loss.item()

    progress_bar.close()  # Close the custom progress bar

    return running_loss / (i + 1)

def get_data(startdate, enddate, gfs_path, era5_path, batch_size, num_workers, seed, train=True, shuffle=True):

    print(f'batch_size: {batch_size}, workers: {num_workers}, seed: {seed}')
    check_missing_files(startdate, enddate, gfs_path, era5_path)

    # Create GFS and ERA5 datasets
    gfs_dataset = NetCDFDataset(gfs_path, startdate, enddate)
    era5_dataset = NetCDFDataset(era5_path, startdate, enddate)

    if train:
        # Create the shuffled indices for both datasets
        shuffled_indices = torch.randperm(len(gfs_dataset))

        # Apply shuffled indices to both datasets
        gfs_dataset.file_list = [gfs_dataset.file_list[i] for i in shuffled_indices]
        era5_dataset.file_list = [era5_dataset.file_list[i] for i in shuffled_indices]


    torch.manual_seed(seed)

    gfs_data_loader = DataLoader(gfs_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    era5_data_loader = DataLoader(era5_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return gfs_data_loader, era5_data_loader


if __name__ == '__main__':
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    seed = 42

    batch_size = 8
    num_workers = 0

    # Get training data
    gfs_root_dir = '/scratch1/NCEPDEV/nems/Linlin.Cui/Tests/ML4BC/GFS'
    era5_root_dir = '/scratch1/NCEPDEV/nems/Linlin.Cui/Tests/ML4BC/ERA5'
    start_date = date(2021, 3, 23)  
    end_date = date(2023, 3, 31) 
    X_train, y_train = get_data(start_date, end_date, gfs_root_dir, era5_root_dir, batch_size, num_workers, seed)

    # Get validation data
    gfs_root_dir = '/scratch1/NCEPDEV/nems/Linlin.Cui/Tests/ML4BC/GFS/val_data'
    era5_root_dir = '/scratch1/NCEPDEV/nems/Linlin.Cui/Tests/ML4BC/ERA5/val_data'
    start_date = date(2023, 4, 1)  
    end_date = date(2023, 12, 21) 
    X_val, y_val = get_data(start_date, end_date, gfs_root_dir, era5_root_dir, batch_size, num_workers, seed, train=False, shuffle=False)

    # Define the loss function and optimizer
    loss_fn = nn.MSELoss()
    model = get_autoencoder(device)  # Accessing the autoencoder model
    lr = 1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    writer = SummaryWriter('runs/gfs_trainer_{}'.format(timestamp))
    epoch_number = 0

    EPOCHS = 50

    for epoch in range(EPOCHS):
        model.train()
        print(f'Epoch {epoch}')
        avg_loss = train_one_epoch(epoch_number, X_train, y_train)

        running_vloss = 0.
        model.eval()

        with torch.no_grad():
            for i, (vinputs, vlabels) in enumerate(zip(X_val, y_val)):
                voutputs = model(vinputs.to(device))
                vloss = loss_fn(voutputs, vlabels.to(device))
                running_vloss += vloss
    
        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        writer.add_scalars('Training vs. Validation Loss',
                           { 'Training' : avg_loss, 'Validation': avg_vloss},
                           epoch_number + 1)
        writer.flush()

        #Save the model's state
        model_path = f'checkpoints/model_{timestamp}_{epoch_number}.pth'
        torch.save(model.state_dict(), model_path)

        epoch_number += 1

