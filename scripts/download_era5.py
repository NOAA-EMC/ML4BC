from datetime import datetime
import cdsapi

startdate = datetime(2024, 1, 1)
enddate = datetime(2024, 2, 3)
c = cdsapi.Client()

r = c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'format': 'grib',
        'variable': ['2m_temperature'],
        'date': f"{startdate.strftime('%Y-%m-%d')}/{enddate.strftime('%Y-%m-%d')}",
        'time': [
            '00:00',
            '06:00',
            '12:00',
            '18:00',
        ],
        'format': 'netcdf'
    },
    )

r.download(f'era5_surface_t2m_{startdate.strftime("%Y%m%d")}-{enddate.strftime("%Y%m%d")}.nc')

