import os
from datetime import timedelta, datetime

import xarray as xr
import numpy as np

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
    
def calculate_mean_and_std(root_dir, start_date, end_date, product='gfs', bbox=None):
    time_step = timedelta(days=1)
    current_date = start_date
    total_count = 0
    total_mean = 0.0
    total_var = 0.0

    while current_date <= end_date:
        print(current_date)
        for hour in ['00', '06', '12', '18']:
            if product == 'gfs':
                filename = f"GFS.{current_date.strftime('%Y%m%d')}{hour}.nc"
            elif product == 'era5':
                filename = f"ERA5.{current_date.strftime('%Y%m%d')}{hour}.nc"
            else:
                raise ValueError(f'Product {product} is not supported, choose gfs or era5!')

            file_path = os.path.join(root_dir, filename)

            if os.path.exists(file_path):
                ds = xr.open_dataset(file_path)

                if bbox is not None:
                    xmin, xmax, ymin, ymax = bbox
                    ds = ds.sel(latitude=slice(ymax, ymin), longitude=slice(xmin, xmax))

                data = ds.t2m.values.astype(np.float64)  # Adjust this to your variable name
                ds.close()

                total_mean += np.mean(data)
                total_var += np.var(data)
                total_count += 1

        current_date += time_step

    total_mean = total_mean / total_count
    total_std = np.sqrt(total_var / total_count)

    return total_mean, total_std

if __name__ == "__main__":

    data_dir = "/scratch1/NCEPDEV/nems/Linlin.Cui/Tests/ML4BC/run27/data"
    start_date = datetime(2021, 3, 23)
    end_date = datetime(2021, 3, 31) #included

    #bbox = [230, 300, 25, 50]
    #global_means, global_stds = calculate_mean_and_std(data_dir, start_date, end_date, bbox)

    global_means, global_stds = calculate_mean_and_std(data_dir, start_date, end_date, 'gfs')

    print(global_means)
    print(global_stds)
