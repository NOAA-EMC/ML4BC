'''
Purpose: This script gets sorted timeseries for given time period (startdate, enddate),
         subsets stations within a polygon, and save it to pickle file.
8/23/2024, Linlin Cui (linlin.cui@noaa.gov)
'''
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt

def resample_station(df):
    df.sort_index(inplace=True)
    df.reset_index(inplace=True)
    #resampled_df =  df.resample('6h').bfill()
    return df


startdate = datetime(2024, 1, 1)
enddate = datetime(2024, 8, 21, 6)
#date2 = datetime(2024, 1, 10, 6)
datevector = np.arange(startdate, enddate, np.timedelta64(6, 'h')).astype(datetime)

#define polygon [230, 300], [25, 50]
xmin, xmax, ymin, ymax = -130, -60, 25, 50
coords = ((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin))
polygon = Polygon(coords)

data = []
for date in datevector:
    fname = f'2024/ascii/gdas.adpsfc.t{date.hour:02d}z.{date.strftime("%Y%m%d")}.txt'
    print(fname)
    fid = open(fname, 'r')
    lines=fid.readlines()
    fid.close()
    
    
    for i, line in enumerate(lines[3:]):
        #print(i+3)
    
        values = re.split(r'\s+', line)
        reporttime = datetime.strptime(values[3], '%Y%m%d%H%M')

    
        data.append({
            'reporttime': datetime.strptime(values[3], '%Y%m%d%H%M'), 
            #'station_type': values[2],
            'station': values[4], 
            #'elev': float(values[7]), 
            't2m': float(values[11]), 
            'geometry': Point(values[6], values[5])
        })
        #else:
        #    continue

gdf = gpd.GeoDataFrame(data, crs='epsg:4326')
gdf['t2m'] = gdf['t2m'].replace(-9999.9, np.nan)
#gdf.dropna(inplace=True)

gdf2 = gdf[gdf.within(polygon)]

gdf2.set_index('reporttime', inplace=True)
df = pd.DataFrame(gdf2.drop(columns='geometry'))
df_sorted = df.groupby('station').apply(resample_station, include_groups=False)
df_sorted.to_pickle(f"./2024/gdas.adpsfc.{startdate.strftime('%Y%m%d%H')}-{enddate.strftime('%Y%m%d%H')}_sorted.pkl")

##https://stackoverflow.com/questions/18835077/selecting-from-multi-index-pandas
##df_obs = df_resampled.iloc[df_resampled.index.get_level_values('station') == 'KLIT']  #
#df_obs = df_sorted.iloc[df_sorted.index.get_level_values('station') == 'KLIT'] 
#
#def closet_timestamp(target_date):
#    closet_index = (df_obs['reporttime'] - target_date).abs().idxmin()
#    return df_obs.loc[closet_index]
#
#
#date_range = pd.date_range(start='1/1/2024', end='1/3/2024', freq='6h', inclusive='right')
#breakpoint()
#df_closest = date_range.to_series().apply(closet_timestamp)
