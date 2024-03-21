'''
Description: This script provides utilities, including:
(i) NetCDFDataset class and  Provide functionalities for data processing, normalizing, rescaling, and making pytorch dataloader for both GFS and ERA5.
(ii) check_missing_files: A function for checking missing files.
(iii) calculate_mean_and_std: A function to calculate the mean and standard deviation of the training dataset which provides values for normalization and rescaling modules.
    
Author: Sadegh Sadeghi Tabas (sadegh.tabas@noaa.gov)
Revision history:
    -20231029: Sadegh Tabas, initial code
'''

import os

import netCDF4 as nc
import numpy as np
import torch
from torch.utils.data import Dataset
from datetime import timedelta, date

class NetCDFDataset(Dataset):
    def __init__(self, root_dir, start_date, end_date, transform=True):
        self.root_dir = root_dir
        self.file_list = self.create_file_list(root_dir, start_date, end_date)
        self.mean = 278.83097 #279.9124
        self.std = 56.02780 #107.1107
        self.transform = transform

    @staticmethod
    def create_file_list(root_dir, start_date, end_date):
        file_list = []
        time_step = timedelta(days=1)  # 6 hours interval
        current_date = start_date

        while current_date <= end_date:
            for hh in ['00', '06', '12', '18']:
                filename = f'{os.path.basename(root_dir).split('_')[0]}.t2m.{current_date.strftime("%Y%m%d")}{hh}.nc'
                file_list.append(filename)
            current_date += time_step
        
        return file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, self.file_list[idx])
        
        # Load NetCDF data
        dataset = nc.Dataset(file_path)
        data = dataset.variables['t2m'][:].astype(np.float32)  # Adjust 'data' to the variable name in your file
        dataset.close()
        #print(file_path) 
        # Reshape the data to (1, 50, 721, 1440)
        data = data.reshape(1, 1, 721, 1440)

        if self.transform:
            data = self.normalize_data(data)  # Normalize the data if transform is True

        return torch.tensor(data)

    def normalize_data(self, data):
        data = (data - self.mean) / self.std
        return data

    def rescale_data(self, data):
        data = (data * self.std) + self.mean
        return data

def check_missing_files(start_date, end_date, gfs_directory, era5_directory):
    time_step = timedelta(days=1)
    current_date = start_date
    total_missing_files = 0

    while current_date <= end_date:
        date_str = current_date.strftime("%Y%m%d")
        for hour_str in ['00', '06', '12', '18']:
            gfs_file_name = f"GFS.t2m.{date_str}{hour_str}.nc"
            gfs_file_path = os.path.join(gfs_directory, gfs_file_name)

            era5_file_name = f"ERA5.t2m.{date_str}{hour_str}.nc"
            era5_file_path = os.path.join(era5_directory, era5_file_name)

            if not os.path.exists(gfs_file_path):
                print(f"Missing file in GFS directory: {gfs_file_name}")
                total_missing_files += 1

            if not os.path.exists(era5_file_path):
                print(f"Missing file in ERA5 directory: {era5_file_name}")
                total_missing_files += 1

        current_date += time_step

    print(f"Total number of missing files: {total_missing_files}")
    
def calculate_mean_and_std(root_dir, start_date, end_date):
    time_step = timedelta(days=1)
    current_date = start_date
    total_count = 0
    total_mean = 0.0
    total_var = 0.0

    while current_date <= end_date:
        for hour in ['00', '06', '12', '18']:
            filename = f"GFS.t2m.{current_date.strftime('%Y%m%d')}{hour}.nc"
            file_path = os.path.join(root_dir, filename)

            if os.path.exists(file_path):
                dataset = nc.Dataset(file_path)
                data = dataset.variables['t2m'][:]  # Adjust this to your variable name
                dataset.close()

                current_mean = np.mean(data)
                total_mean = (total_count * total_mean + len(data) * current_mean) / (total_count + len(data))
                total_var = (total_count * total_var + np.sum((data - current_mean) ** 2)) / (total_count + len(data))
                total_count += len(data)

        current_date += time_step

    total_std = np.sqrt(total_var / total_count)
    return total_mean, total_std
