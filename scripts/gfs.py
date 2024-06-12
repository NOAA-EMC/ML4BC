import os
import pathlib
import glob
import boto3
import xarray as xr
import numpy as np
from botocore.config import Config
from botocore import UNSIGNED
import pygrib

gfs_vars = {
    '2t': [{'typeOfLevel': 'heightAboveGround', 'level': 2, 'name': 't2m'}],
    #'t': [{'typeOfLevel': 'isobaricInhPa', 'level': 1000},{'typeOfLevel': 'surface', 'level': 0}],
    #'r': [{'typeOfLevel': 'isobaricInhPa', 'level': 1000}],
    #'prmsl': [{'typeOfLevel': 'meanSea', 'level': 0}],
    #'10u': [{'typeOfLevel': 'heightAboveGround', 'level': 10}],
    #'10v': [{'typeOfLevel': 'heightAboveGround', 'level': 10}],
    #'st': [{'typeOfLevel': 'depthBelowLandLayer', 'level': 0}],
    #'lsm': [{'typeOfLevel': 'surface', 'level': 0}],
    #'soill': [{'typeOfLevel': 'depthBelowLandLayer', 'level': 0}],
}

s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
bucket_name = 'noaa-gfs-bdp-pds'

def get_gfs_data_by_date(date, outdir, download_dir):


    # Define the output NetCDF file name
    output_file_name = f'GFS.{date.strftime("%Y%m%d%H")}.nc'
    output_file_path = os.path.join(outdir, output_file_name)

    if os.path.isfile(output_file_path):
        print(f'File {output_file_path} exists!')
        return 0

    mergeDSs = []
    for i in np.arange(6, 241, 6):
        key = f"gfs.{date.strftime('%Y%m%d')}/{date.hour:02d}/atmos/gfs.t{date.hour:02d}z.pgrb2.0p25.f{i:03d}"
        filename = pathlib.Path(download_dir) / key
        if filename.is_file():
            continue
        filename.parent.mkdir(parents=True, exist_ok=True)

        with open(filename, 'wb') as f:
                try:
                    s3.download_fileobj(bucket_name, key, f)
                    print(f"Downloading file {key}: success!") 
                except:
                    print(f'file {key} is not available!')
                    continue
    
    files = glob.glob(f"{download_dir}/gfs.{date.strftime('%Y%m%d')}/{date.hour:02d}/atmos/*")
    files.sort()
    for fname in files:
        print(fname)
        grbs = pygrib.open(fname)
        mergeDAs = []
        for variable_name in gfs_vars:
            for level_type_info in gfs_vars[variable_name]:
                levelType = level_type_info['typeOfLevel']
                desired_level = level_type_info['level']
                varName = level_type_info['name']
                
                # Find the matching grib message
                variable_message = grbs.select(shortName=variable_name, typeOfLevel=levelType, level=desired_level)[0]
                
                # create a netcdf dataset using the matching grib message
                lats, lons = variable_message.latlons()
                lats = lats[:,0]
                lons = lons[0,:]
                data = variable_message.values
                steps = variable_message.validDate
                #varName = f'{variable_message.shortName}_{levelType}_{desired_level}'
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
    
    final_dataset.to_netcdf(output_file_path)

    final_dataset.close()

    print(f"Saved the dataset to {output_file_path}")

