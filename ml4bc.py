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

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
from autoencoder_model import get_autoencoder
from netcdf_dataset import NetCDFDataset, check_missing_files, calculate_mean_and_std
from datetime import date

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps")

# Define your data directories
gfs_root_dir = '/scratch1/NCEPDEV/nems/Linlin.Cui/Tests/ML4BC/GFS_6h'
era5_root_dir = '/scratch1/NCEPDEV/nems/Linlin.Cui/Tests/ML4BC/ERA5_6h'

# Define the start and end date for the dataset
start_date = date(2021, 3, 23)  # Adjust the start date
end_date = date(2023, 3, 23)    # Adjust the end date

check_missing_files(start_date, end_date, gfs_root_dir, era5_root_dir)

# Create GFS and ERA5 datasets
gfs_dataset = NetCDFDataset(gfs_root_dir, start_date, end_date)
era5_dataset = NetCDFDataset(era5_root_dir, start_date, end_date)

# Create the shuffled indices for both datasets
shuffled_indices = torch.randperm(len(gfs_dataset))

# Apply shuffled indices to both datasets
gfs_dataset.file_list = [gfs_dataset.file_list[i] for i in shuffled_indices]
era5_dataset.file_list = [era5_dataset.file_list[i] for i in shuffled_indices]

batch_size = 8
shuffle = False
num_workers = 0
seed = 42
torch.manual_seed(seed)

# Define the loss function and optimizer
loss_fn = nn.MSELoss()
autoencoder = get_autoencoder(device)  # Accessing the autoencoder model
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.0001)

gfs_data_loader = DataLoader(gfs_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
era5_data_loader = DataLoader(era5_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

# Training loop with a custom progress bar
num_epochs = 20
for epoch in range(num_epochs):
    autoencoder.train()
    total_loss = 0.0

    # Create a custom progress bar for the epoch
    progress_bar = tqdm(enumerate(zip(gfs_data_loader, era5_data_loader)), total=len(gfs_data_loader), desc=f'Epoch [{epoch+1}/{num_epochs}]', dynamic_ncols=True)
    for batch_idx, (gfs_data, era5_data) in progress_bar:
        optimizer.zero_grad()
        outputs = autoencoder(gfs_data.to(device))
        loss = loss_fn(outputs, era5_data.to(device))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    progress_bar.close()  # Close the custom progress bar

    # Calculate and print the average loss for the epoch
    avg_loss = total_loss / len(gfs_data_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

    # Save the trained model
    torch.save(autoencoder.module.state_dict() if isinstance(autoencoder, nn.DataParallel) else autoencoder.state_dict(), f'autoencoder_model_epoch_{epoch+1}.pth')

