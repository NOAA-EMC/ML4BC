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

def train_one_epoch(epoch_index, tb_writer, X, y):
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
        #print(f'i = {i}')
        # if i % 1000 == 999:
        #     last_loss = running_loss / 1000
        #     print('  batch {} loss: {}'.format(i + 1, last_loss))
        #     print('  batch {} loss: {}'.format(i + 1, last_loss))
        #     tb_x = epoch_index * len(X) + i + 1
        #     tb_writer.add_scalar('Loss/train', last_loss, tb_x)
        #     running_loss = 0.

    progress_bar.close()  # Close the custom progress bar
    print('LOSS train {}'.format(running_loss / (i + 1)))

    return running_loss / (i + 1)

def get_data(startdate, enddate, gfs_path, era5_path, train=True, shuffle=True):
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

    print(gfs_dataset.file_list[-1])
    print(era5_dataset.file_list[-1])

    batch_size = 8
    num_workers = 0
    seed = 42
    torch.manual_seed(seed)

    gfs_data_loader = DataLoader(gfs_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    era5_data_loader = DataLoader(era5_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return gfs_data_loader, era5_data_loader


if __name__ == '__main__':
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)


    # Get training data
    gfs_root_dir = '/scratch1/NCEPDEV/nems/Linlin.Cui/Tests/ML4BC/GFS_6h'
    era5_root_dir = '/scratch1/NCEPDEV/nems/Linlin.Cui/Tests/ML4BC/ERA5_6h'
    start_date = date(2021, 3, 23)  
    end_date = date(2023, 3, 23) 
    X_train, y_train = get_data(start_date, end_date, gfs_root_dir, era5_root_dir)

    # Get validation data
    gfs_root_dir = '/scratch1/NCEPDEV/nems/Linlin.Cui/Tests/ML4BC/GFS_6h_val'
    era5_root_dir = '/scratch1/NCEPDEV/nems/Linlin.Cui/Tests/ML4BC/ERA5_6h_val'
    start_date = date(2023, 4, 1)  
    end_date = date(2023, 7, 14) 
    X_val, y_val = get_data(start_date, end_date, gfs_root_dir, era5_root_dir, train=False, shuffle=False)

    # Define the loss function and optimizer
    loss_fn = nn.MSELoss()
    model = get_autoencoder(device)  # Accessing the autoencoder model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    writer = SummaryWriter('runs/gfs_trainer_{}'.format(timestamp))
    epoch_number = 0

    EPOCHS = 50

    best_vloss = 1_000_000.

    loss_file = open('loss.txt', "w+")
    loss_file.write(f'Loss: Train, validation \n')
    for epoch in range(EPOCHS):
        model.train()
        avg_loss = train_one_epoch(epoch_number, writer, X_train, y_train)

        running_vloss = 0.
        model.eval()

        with torch.no_grad():
            for i, (vinputs, vlabels) in enumerate(zip(X_val, y_val)):
                voutputs = model(vinputs.to(device))
                vloss = loss_fn(voutputs, vlabels.to(device))
                running_vloss += vloss
    
        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        loss_file.write(f'{avg_loss:.4f} {avg_vloss:.4f} \n')
        writer.add_scalars('Training vs. Validation Loss',
                           { 'Training' : avg_loss, 'Validation': avg_vloss},
                           epoch_number + 1)
        writer.flush()

        #Save the model's state
        model_path = f'checkpoints/model_{timestamp}_{epoch_number}.pth'
        torch.save(model.state_dict(), model_path)

        epoch_number += 1

    loss_file.close()