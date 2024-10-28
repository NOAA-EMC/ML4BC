'''
Description: a utility for preparing ML4BC model inputs from 1.0-degree resolution GFS and ERA5 data. This Python utility has
two modules one for GFS info and the second one for era5 info. 
At this stage, we are going to train the ML4BC model for 2m temperature forecasts with spatial and temporal resolutions of 1.0 degrees
and 3-hourly, respectively. In summary, this utility provides functionalities including:
(i)   Downloading ECMWF's ERA5 and NOAA GFS hourly forecasts,
(ii)  Extract relevant variables from grib2 data
(iii) Reshape, rename, and transform extracted data to the desired format
(iv)  Merge the data through time with the desired number of forecast time steps and save it in NetCDF format
(v)   Produce ERA5 and GFS outputs for model training

Author: Sadegh Sadeghi Tabas (sadegh.tabas@noaa.gov)
Revision history:
    -20231106: Sadegh Tabas, provided utility to process ERA5 and GFS 1.0 degree spatial, 3-hourly temporal and 16-day lead time info.
'''
import os
import boto3
import xarray as xr
import subprocess
import numpy as np
from datetime import datetime, timedelta, date, time
from botocore.config import Config
from botocore import UNSIGNED
import argparse
import fnmatch
import pygrib
import cdsapi

class DataProcessor:
    def __init__(self, start_date, end_date, gfs_variables_with_levels, output_directory=None, download_directory=None, keep_downloaded_data=False):
        self.start_date = start_date
        self.end_date = end_date
        self.output_directory = output_directory
        self.download_directory = download_directory
        self.keep_downloaded_data = keep_downloaded_data
        self.output_gfs = None
        self.output_era5 = None
        self.gfs_vars = gfs_variables_with_levels
        # Specify the output directory where you want to save the files
        if self.output_directory is None:
            self.output_directory= os.getcwd()
        
        # Initialize the S3 client
        self.s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

        # Specify the S3 bucket name and root directory for GFS forecasts
        self.paginator = self.s3.get_paginator('list_objects_v2')
        self.bucket_name = 'noaa-gfs-bdp-pds'
        self.root_directory = 'gfs'

        # Initialize the cdsapi client
        self.cds = cdsapi.Client()


    def gfs_process(self):
        # Specify the local directory where you want to save the GFS files
        if self.download_directory is None:
            self.local_base_directory = os.path.join(os.getcwd(), 'noaa-gfs-bdp-pds-data')  # Use the current directory if not specified
        else:
            self.local_base_directory = os.path.join(self.download_directory, 'noaa-gfs-bdp-pds-data')
        os.makedirs(self.local_base_directory, exist_ok=True)
        
        self.output_gfs = os.path.join(self.output_directory, 'GFS')
        # Check if 'output_directory' exists and create it if it doesn't
        os.makedirs(self.output_gfs, exist_ok=True)
        
        # Loop through the 6-hour intervals
        current_datetime = datetime.combine(self.start_date, datetime.min.time())
        end_datetime = datetime.combine(self.end_date, time(18, 0, 0))
        while current_datetime <= end_datetime:
            date_str = current_datetime.strftime("%Y%m%d")
            time_str = current_datetime.strftime("%H")

            # Construct the S3 prefix for the directory
            s3_prefix = f"{self.root_directory}.{date_str}/{time_str}/{'atmos'}/{'gfs'}"

            # List objects in the S3 directory
            s3_objects = self.paginator.paginate(Bucket=self.bucket_name, Prefix=s3_prefix)
            
            # Filter objects based on the desired formats
            mergeDSs = []
            for page in s3_objects:
                for obj in page['Contents']:
                    obj_key = obj['Key']
                    if fnmatch.fnmatch(obj_key, f'*.pgrb2.1p00.f[0-9][0-9][0-9]'):
                        # Define the local directory path where the file will be saved
                        local_directory = os.path.join(self.local_base_directory, date_str, time_str)
    
                        # Create the local directory if it doesn't exist
                        os.makedirs(local_directory, exist_ok=True)
    
                        # Define the local file path
                        local_file_path = os.path.join(local_directory, os.path.basename(obj_key))
    
                        # Check if the file exited on the disk
                        if os.path.exists(local_file_path):
                            print(f"The file at {local_file_path} exists.")
                        else:
                            # Download the file from S3 to the local path
                            self.s3.download_file(self.bucket_name, obj_key, local_file_path)
                            print(f"Downloaded {obj_key} to {local_file_path}")
    
                        grbs = pygrib.open(local_file_path)
                        
                        # Specify the variable and level you want to extract
                        # variable_name = '2 metre temperature'
                        mergeDAs = []
                        for variable_name in self.gfs_vars:
                            for level_type_info in self.gfs_vars[variable_name]:
                                levelType = level_type_info['typeOfLevel']
                                desired_level = level_type_info['level']
                                
                                # Find the matching grib message
                                variable_message = grbs.select(name=variable_name, typeOfLevel=levelType, level=desired_level)[0]
                                
                                # create a netcdf dataset using the matching grib message
                                lats, lons = variable_message.latlons()
                                lats = lats[:,0]
                                lons = lons[0,:]
                                data = variable_message.values
                                steps = variable_message.validDate
                                varName = f'{variable_message.shortName}_{levelType}_{desired_level}'
                                da = xr.Dataset(
                                    data_vars={
                                        varName: (['latitude', 'longitude'], data)
                                    },
                                    coords={
                                        'longitude': lons,
                                        'latitude': lats,
                                        'time': steps,  
                                    }
                                )
                                da[varName] = da[varName].astype('float32')
                                mergeDAs.append(da)
                            
                        ds = xr.merge(mergeDAs)
                        ds['latitude'] = ds['latitude'].astype('float32')
                        ds['longitude'] = ds['longitude'].astype('float32')
                        
                        mergeDSs.append(ds)
                    
            # final_dataset = xr.merge(mergeDSs)
            final_dataset = xr.concat(mergeDSs, dim='time')
            
            # Define the output NetCDF file name
            output_file_name = f'GFS.{current_datetime.strftime("%Y%m%d%H")}.nc'
            output_file_path = os.path.join(self.output_gfs, output_file_name)
            final_dataset.to_netcdf(output_file_path)
            print(f"Saved the dataset to {output_file_path}")
            
            if not self.keep_downloaded_data:
                # Remove downloaded data from the specified directory
                print("Removing downloaded data...")
                try:
                    os.system(f"rm -rf {local_directory}")
                    print("Downloaded data removed.")
                except Exception as e:
                    print(f"Error removing downloaded data: {str(e)}")               

            # Move to the next 6-hour interval
            current_datetime += timedelta(hours=6)

        print("GFS Data Processing Completed.")
        
    def era5_process(self):
        # Specify the local directory where you want to save the GFS files
        if self.download_directory is None:
            self.local_base_directory = os.path.join(os.getcwd(), 'ecmwf-era5-t2m-data')  # Use the current directory if not specified
        else:
            self.local_base_directory = os.path.join(self.download_directory, 'ecmwf-era5-t2m-data')
        os.makedirs(self.local_base_directory, exist_ok=True)

        self.output_era5 = os.path.join(self.output_directory, 'ERA5')
        # Check if 'output_directory' exists and create it if it doesn't
        os.makedirs(self.output_era5, exist_ok=True) 
        
        # Loop through the 6-hour intervals
        current_datetime = datetime.combine(self.start_date, datetime.min.time())
        end_datetime = datetime.combine(self.end_date, time(18, 0, 0))
        while current_datetime <= end_datetime:
            local_directory = self.local_base_directory
            
            current_date = current_datetime.date()
            current_end = current_date + timedelta(days=17)
            era5_filename = 'ERA5_t2m_'+str(current_date)+'_to_'+str(current_end)+'.nc'

            # Define the local file path
            local_file_path = os.path.join(local_directory, era5_filename)
            
            # Check if the file exited on the disk
            if os.path.exists(local_file_path):
                print(f"The file at {local_file_path} exists.")
            else:
                # Download the file from S3 to the local path
                print ('Start Downloading ERA5 t2m Data from', str(current_date), 'to', str(current_end))
            
                self.cds.retrieve(
                    'reanalysis-era5-single-levels',
                    {
                        'product_type': 'reanalysis',
                        'variable': '2m_temperature',
                        'grid': '1.0/1.0',
                        'date': f'{current_date}/{current_end}',
                        'time': [
                            '00:00', #'01:00', '02:00',
                            '03:00', #'04:00', '05:00',
                            '06:00', #'07:00', '08:00',
                            '09:00', #'10:00', '11:00',
                            '12:00', #'13:00', '14:00',
                            '15:00', #'16:00', '17:00',
                            '18:00', #'19:00', '20:00',
                            '21:00', #'22:00', '23:00',
                        ],
                        'format': 'netcdf',
                    },
                    f'{local_file_path}')
                
                print ('ERA5 t2m Data Downloading Completed')
            
            for hour in [0, 6, 12, 18]:
                # Combine the date and time to create a new datetime object
                frame_start = datetime.combine(current_date, time(hour,0,0))
                frame_end = frame_start + timedelta(hours=384)
                
                current_ds = xr.open_dataset(local_file_path)
                
                # Slice the dataset based on the time range
                sliced_ds = current_ds.sel(time=slice(frame_start, frame_end))

                # Define the output NetCDF file name
                output_file_name = f'ERA5.{frame_start.strftime("%Y%m%d%H")}.nc'
                output_file_path = os.path.join(self.output_era5, output_file_name)
                sliced_ds.to_netcdf(output_file_path)
                print(f"Saved the dataset to {output_file_path}")

            if not self.keep_downloaded_data:
                # Remove downloaded data from the specified directory
                print("Removing downloaded data...")
                try:
                    os.system(f"rm -rf {local_file_path}")
                    print("Downloaded data removed.")
                except Exception as e:
                    print(f"Error removing downloaded data: {str(e)}")
                    
            current_datetime += timedelta(days=1)
            
        print("ERA5 Data Processing Completed.")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and process GFS and ERA5 data")
    parser.add_argument("start_date", help="Start date in the format 'YYYYMMDD'")
    parser.add_argument("end_date", help="End date in the format 'YYYYMMDD'")
    parser.add_argument("-o", "--output", help="Output directory for processed data")
    parser.add_argument("-d", "--download", help="Download directory for raw data")
    parser.add_argument("-k", "--keep", help="Keep downloaded data (optional)", action="store_true", default=False)
    parser.add_argument("-p", "--process", nargs="*", choices=["gfs", "era5"], help="Specify which process to run")
    args = parser.parse_args()

    start_date = datetime.strptime(args.start_date, "%Y%m%d")
    end_date = datetime.strptime(args.end_date, "%Y%m%d")
    output_directory = args.output
    download_directory = args.download
    keep_downloaded_data = args.keep
    
    gfs_variables_with_levels = {
        '2 metre temperature': [{'typeOfLevel': 'heightAboveGround', 'level': 2}],
        'Temperature': [{'typeOfLevel': 'isobaricInhPa', 'level': 1000},{'typeOfLevel': 'surface', 'level': 0}],
        'Relative humidity': [{'typeOfLevel': 'isobaricInhPa', 'level': 1000}],
        'Pressure reduced to MSL': [{'typeOfLevel': 'meanSea', 'level': 0}],
        '10 metre U wind component': [{'typeOfLevel': 'heightAboveGround', 'level': 10}],
        '10 metre V wind component': [{'typeOfLevel': 'heightAboveGround', 'level': 10}],
        'Soil Temperature': [{'typeOfLevel': 'depthBelowLandLayer', 'level': 0}],
        'Land-sea mask': [{'typeOfLevel': 'surface', 'level': 0}],
        'Liquid volumetric soil moisture (non-frozen)': [{'typeOfLevel': 'depthBelowLandLayer', 'level': 0}],
    }
    
    data_processor = DataProcessor(start_date, end_date, gfs_variables_with_levels, output_directory, download_directory, keep_downloaded_data)      
    if not args.process or "era5" in args.process:
        data_processor.era5_process()
    if not args.process or "gfs" in args.process:
        data_processor.gfs_process()
