'''
Description: This script provides utilities, including:
(i) NetCDFDataset class and  Provide functionalities for data processing, normalizing, rescaling, and making pytorch dataloader for both GFS and ERA5.
(ii) check_missing_files: A function for checking missing files.
(iii) calculate_mean_and_std: A function to calculate the mean and standard deviation of the training dataset which provides values for normalization and rescaling modules.
    
Author: Sadegh Sadeghi Tabas (sadegh.tabas@noaa.gov)
Revision history:
    -20231029: Sadegh Tabas, initial code
    -20231129: Sadegh Tabas, update the class to account for multiple vars as well as time idx
'''
import netCDF4 as nc
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from datetime import timedelta, date
import xarray as xr


class NetCDFDataset(Dataset):
    def __init__(self, root_dir, process, start_date, end_date, transform=False):
        self.root_dir = root_dir
        self.process = process
        self.file_list = self.create_file_list(root_dir, start_date, end_date)
        self.mean = 278.83097
        self.std = 56.02780
        self.transform = transform
        
        # Define latitudes and longitudes
        self.lat_lons = []
        for lat in range(-90, 91, 1):
            for lon in range(0, 360, 1):
                self.lat_lons.append([lat, lon])

    @staticmethod
    def create_file_list(root_dir, start_date, end_date):
        file_list = []
        time_step = timedelta(days=1)
        current_date = start_date

        while current_date <= end_date:
            for hh in ['00', '06', '12', '18']:
                filename = f'{os.path.basename(root_dir)}.{current_date.strftime("%Y%m%d")}{hh}.nc'
                file_list.append(filename)
            current_date += time_step
        
        return file_list

    def __len__(self):
        return len(self.file_list) * 129  # Each file has 129 time dimensions

    def __getitem__(self, idx):
        file_idx = idx // 129  # Calculate the file index
        time_idx = idx % 129  # Calculate the time index within the file
        
        file_path = os.path.join(self.root_dir, self.file_list[file_idx])
        dataset = xr.open_dataset(file_path)
        variable_names = dataset.data_vars

        all_data = []
        
        for var_name in variable_names:
            data = dataset.variables[var_name][time_idx].astype(np.float32)
            data = data.fillna(-999)
            all_data.append(data)

        dataset.close()

        all_data = np.stack(all_data)  # Stack the data along the variable dimension
        all_data = all_data.reshape(all_data.shape[:-2] + (-1,))
        all_data = np.transpose(all_data, (1,0))
        
        if self.process=='gfs':
            time_variable = np.full((181*360, 1), np.array(time_idx, dtype=np.float32))  # Create an array for the time index
            sin_lat_lons = np.sin(np.array(self.lat_lons) * np.pi / 180.0).astype(np.float32)
            cos_lat_lons = np.cos(np.array(self.lat_lons) * np.pi / 180.0).astype(np.float32)
            
            # Add time, sin and cos index as a new variables dimension
            all_data = np.concatenate((all_data, time_variable, sin_lat_lons, cos_lat_lons), axis=1)
        
        if self.transform:
            pass

        return torch.tensor(all_data)

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
            gfs_file_name = f"GFS.{date_str}{hour_str}.nc"
            gfs_file_path = os.path.join(gfs_directory, gfs_file_name)

            era5_file_name = f"ERA5.{date_str}{hour_str}.nc"
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
