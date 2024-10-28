'''
Description: This script provides utilities, including:
(i) NetCDFDataset class and  Provide functionalities for data processing, normalizing, rescaling, and making pytorch dataloader for both GFS and ERA5.
(ii) check_missing_files: A function for checking missing files.
(iii) calculate_mean_and_std: A function to calculate the mean and standard deviation of the training dataset which provides values for normalization and rescaling modules.
    
Author: Sadegh Sadeghi Tabas (sadegh.tabas@noaa.gov)
Revision history:
    -20231029: Sadegh Tabas, initial code
    -20240401: Linlin Cui, updated to find pair of GFS and ERA5 data
'''

import os
from datetime import timedelta, datetime

import xarray as xr
import numpy as np
import torch
from torch.utils.data import Dataset

from scripts.get_stats import calculate_mean_and_std

class NetCDFDataset(Dataset):
    def __init__(
        self, 
        data_dir: str, 
        start_date, 
        end_date, 
        bbox=None, 
        transform=True
    ):
        self.data_dir = data_dir
        self.file_list = self.create_file_list(data_dir, start_date, end_date)
        self.bbox = bbox
        self.transform = transform

        if self.transform:
            self.gfs_means, self.gfs_stds = calculate_mean_and_std(data_dir, start_date, end_date, 'gfs', bbox)
            self.era5_means, self.era5_stds = calculate_mean_and_std(data_dir, start_date, end_date, 'era5', bbox)
            #print(f'gfs mean: {self.gfs_means}, stds: {self.gfs_stds}')
            #print(f'era5 mean: {self.era5_means}, stds: {self.era5_stds}')

    @staticmethod
    def create_file_list(root_dir, start_date, end_date):
        file_list = []

        datevector = np.arange(start_date, end_date, np.timedelta64(6, 'h')).astype(datetime)
        for date in datevector:
            gfs_fname = os.path.join(root_dir, f'GFS.{date.strftime("%Y%m%d%H")}.nc')
            era_fname = os.path.join(root_dir, f'ERA5.{date.strftime("%Y%m%d%H")}.nc')

            if os.path.isfile(gfs_fname) and os.path.isfile(era_fname):
                file_list.append([gfs_fname, era_fname])
            else:
                print(f'No matching GFS/ERA5 for time {date.strftime("%Y%m%d%H")}')
        
        return file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        f1, f2 = self.file_list[idx]

        gfs_file_path = os.path.join(self.data_dir, f1)
        gfs_tensor = self.get_gfs_data(gfs_file_path)

        era5_file_path = os.path.join(self.data_dir, f2)
        era5_tensor = self.get_era_data(era5_file_path)

        return gfs_tensor, era5_tensor

    def get_gfs_data(self, filename: str):

        # Load NetCDF data
        ds = xr.open_dataset(filename)
        

        if self.bbox is not None:
            xmin, xmax, ymin, ymax = self.bbox
            values = ds.t2m.sel(latitude=slice(ymax, ymin), longitude=slice(xmin, xmax)).values.astype(np.float32)  #subset the domain
        else:
            values = ds.t2m.values.astype(np.float32)

        if self.transform:
            data = self.normalize_data(values, self.gfs_means, self.gfs_stds)  # Normalize the data if transform is True

        data_tensor = torch.from_numpy(data) if not isinstance(data, torch.Tensor) else data
        #print(data_tensor.size())

        # Reshape the data

        ds.close()

        return data_tensor

    def get_era_data(self, filename: str):

        #T2m min: 232.6013375680839, max: 322.3867275283499: mean: 288.54418174392987, std: 10.753967912307285
        # Load NetCDF data
        ds = xr.open_dataset(filename)

        if self.bbox is not None:
            xmin, xmax, ymin, ymax = self.bbox
            values = ds.t2m.sel(latitude=slice(ymax, ymin), longitude=slice(xmin, xmax)).values.astype(np.float32)  #subset the domain
        else:
            values = ds.t2m.values.astype(np.float32)
        #assert data.shape == (105, 281), f"{filename}: shape of data should be (105, 281)"

        if self.transform:
            data = self.normalize_data(values, self.era5_means, self.era5_stds)  # Normalize the data if transform is True

        data_tensor = torch.from_numpy(data) if not isinstance(data, torch.Tensor) else data

        # Reshape the data
        data_reshaped = data_tensor.unsqueeze(0)

        ds.close()

        return data_reshaped

    def normalize_data(self, data, mean, std):
        data = (data - mean) / std
        return data

    def rescale_data(self, data):
        data = (data * self.std) + self.mean
        return dat
