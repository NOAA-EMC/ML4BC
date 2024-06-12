from datetime import datetime, timedelta
import pathlib

import numpy as np
import xarray as xr

if __name__ == '__main__':

    startdate = datetime(2024, 1, 1)
    enddate = datetime(2024, 2, 1) #excluded
    datevector = np.arange(startdate, enddate, np.timedelta64(6, 'h')).astype(datetime)

    outdir = pathlib.Path('/scratch1/NCEPDEV/nems/Linlin.Cui/Tests/ML4BC/data_test')
    outdir.mkdir(parents=True, exist_ok=True)

    #select f072 
    dt = 3

    #read era5
    ds = xr.open_dataset('era5_surface_t2m_20240101-20240203.nc')

    for date in datevector:
        print(date)

        #timevectors = np.arange(
        #    date + timedelta(hours=6),
        #    date + timedelta(days=10.04),
        #    np.timedelta64(6, 'h')
        #)

        date2 = date + timedelta(days=dt)

        ds_slice = ds.sel(time=date2)
        ds_slice.to_netcdf(f'{outdir}/ERA5.{date.strftime("%Y%m%d%H")}.nc')
        ds_slice.close()
