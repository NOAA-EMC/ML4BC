from datetime import datetime, timedelta

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import geopandas as gpd
import scipy as sp

def get_dataarray(values, lon, lat):
   da = xr.DataArray(
       data=values,
       dims=("latitude", "longitude"),
       coords={"latitude": lat, "longitude": lon},
   )

   return da

#ftime = '192'
date1 = datetime(2024, 1, 1)
date2 = datetime(2024, 6, 1)
datevector = np.arange(date1, date2, np.timedelta64(6, 'h')).astype(datetime)
ds = xr.open_dataset(f'../../run27/data/ERA5.{date1.strftime("%Y%m%d%H")}.nc')
ds_slice = ds.t2m.sel(latitude=slice(50, 25), longitude=slice(230, 300))
lon = ds_slice.longitude.values
lon = lon - 360
lat = ds_slice.latitude.values
ds.close()
print(lon.shape)
print(lat.shape)

gdf = gpd.read_file('/scratch1/NCEPDEV/nems/Linlin.Cui/Tests/ML4BC/ADPSFC/adpsfc_stations_conus.shp')

station_lon, station_lat = gdf.geometry.x.values, gdf.geometry.y.values
bxy = np.c_[station_lat, station_lon]

for date in datevector:
    print(date)

    t2m_gfs, t2m_corrected, t2m_era5 = [], [], []
    for it in np.arange(6, 241, 6):

        ftime = f'f{it:03d}'
    
        for product in ['unet', 'gfs', 'era5']:
            if product == 'gfs':
                data = np.load(f'../../run34/outputs_{ftime}/t2m_gfs_{date.strftime("%Y%m%d%H")}.npy')
            elif product == 'unet':
                data = np.load(f'../../run34/outputs_{ftime}/t2m_corrected_{date.strftime("%Y%m%d%H")}.npy')
                data = np.squeeze(data)
            elif product == 'era5':
                data = np.load(f'../../run34/outputs_{ftime}/t2m_era5_{date.strftime("%Y%m%d%H")}.npy')
            else:
                raise ValueError(f'product {product} is not implemented!')

            data = data - 273.15 #kelvin to celsius
            val_fd = sp.interpolate.RegularGridInterpolator((lat[::-1], lon),np.squeeze(data[::-1,:]),'nearest', bounds_error=False, fill_value = float('nan'))
            val_int = val_fd(bxy)

            if product == 'gfs':
                t2m_gfs.append(val_int)
            elif product == 'unet':
                t2m_corrected.append(val_int)
            elif product == 'era5':
                t2m_era5.append(val_int)
            else:
                raise ValueError(f'product {product} is not implemented!')
    
    
    t2m_gfs = np.array(t2m_gfs)
    t2m_corrected = np.array(t2m_corrected)
    t2m_era5 = np.array(t2m_era5)

    np.save(f'grid2obs/gfs2obs_{date.strftime("%Y%m%d%H")}.npy', t2m_gfs)
    np.save(f'grid2obs/unet2obs_{date.strftime("%Y%m%d%H")}.npy', t2m_corrected)
    np.save(f'grid2obs/era52obs_{date.strftime("%Y%m%d%H")}.npy', t2m_era5)
    
