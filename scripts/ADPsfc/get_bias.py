'''
Description: This script calcuates the bias with 'model minus observation' for each station
             for the period defined as variable 'datevector'.

             Inputs:
               - station data is 2024/gdas.adpsfc.2024010100-2024082106_sorted.pkl, which is the
                 output from get_station_timeseries.py
               - A shapefile for station information: adpsfc_stations_conus.shp
               - model outputs from model2obs.py
8/23/2024, Linlin Cui (linlin.cui@noaa.gov)
'''
import re
from datetime import datetime, timedelta
from time import time

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

cycle = 0
startdate = datetime(2024, 1, 1, cycle)
enddate = datetime(2024, 6, 1)
datevector = np.arange(startdate, enddate, np.timedelta64(1, 'D')).astype(datetime)

df = pd.read_pickle('2024/gdas.adpsfc.2024010100-2024082106_sorted.pkl')

gfs_st = gpd.read_file('/scratch1/NCEPDEV/nems/Linlin.Cui/Tests/ML4BC/ADPSFC/adpsfc_stations_conus.shp')

bias_all_00 = []
bias_all_06 = []
bias_all_12 = []
bias_all_18 = []
bias_all = []

t0 = time()
for i, st in enumerate(gfs_st.station):

    #if i == 10:
    #    break

    print(f'{i+1} of {len(gfs_st)} stations')

    if not st.startswith('K'):   #station name starts with 'K' is in USA
        continue

    print(st)
    df_obs = df.iloc[df.index.get_level_values('station') == st]  #

    bias = []
    t00 = time()
    for date in datevector:
    
        timeindex = pd.date_range(start=date, end=date + timedelta(days=10), freq='6h', inclusive='right')
    
        
        t2m_gfs = np.load(f'/scratch1/NCEPDEV/nems/Linlin.Cui/Tests/ML4BC/Milestone/0/grid2obs/gfs2obs_{date.strftime("%Y%m%d%H")}.npy')
        df_gfs = pd.DataFrame(t2m_gfs, index=timeindex, columns=gfs_st.station)
        df_gfs2 = df_gfs[st].values
        
        t2m_unet = np.load(f'/scratch1/NCEPDEV/nems/Linlin.Cui/Tests/ML4BC/Milestone/0/grid2obs/unet2obs_{date.strftime("%Y%m%d%H")}.npy')
        df_unet = pd.DataFrame(t2m_unet, index=timeindex, columns=gfs_st.station)
        df_unet2 = df_unet[st].values 
        
        t2m_era5 = np.load(f'/scratch1/NCEPDEV/nems/Linlin.Cui/Tests/ML4BC/Milestone/0/grid2obs/era52obs_{date.strftime("%Y%m%d%H")}.npy')
        df_era5 = pd.DataFrame(t2m_era5, index=timeindex, columns=gfs_st.station)
        df_era52 = df_era5[st].values 
    
        #mask = (df_obs.reporttime > date) & (df_obs.reporttime <= date + timedelta(days=10))
    

        def closet_timestamp(target_date):
            closet_index = (df_obs['reporttime'] - target_date).abs().idxmin()
            return df_obs.loc[closet_index]

        df_closest = timeindex.to_series().apply(closet_timestamp)
        if df_closest.t2m.isna().sum()> 0:
        #breakpoint()
            continue
        obs = df_closest.t2m.values
    
        bias.append([df_gfs2-obs, df_unet2-obs, df_era52-obs])

    if not bias:
        print(f'No observation data for station {st}')
        continue
    
    bias = np.array(bias)

    #bias_all_00 should be hour 06,  bias_all_06-> hour 12, bias_all_12 -> hour18, bias_all_18 -> hour 00
    bias_all_00.append({'station': st, 'gfs-obs': bias[:,0,0].mean(), 'unet-obs': bias[:,1,0].mean(), 
        'era5-obs': bias[:,2,0].mean(), 'geometry': gfs_st[gfs_st.station == st].geometry.values[0]})
    bias_all_06.append({'station': st, 'gfs-obs': bias[:,0,1].mean(), 'unet-obs': bias[:,1,1].mean(), 
        'era5-obs': bias[:,2,1].mean(), 'geometry': gfs_st[gfs_st.station == st].geometry.values[0]})
    bias_all_12.append({'station': st, 'gfs-obs': bias[:,0,2].mean(), 'unet-obs': bias[:,1,2].mean(), 
        'era5-obs': bias[:,2,2].mean(), 'geometry': gfs_st[gfs_st.station == st].geometry.values[0]})
    bias_all_18.append({'station': st, 'gfs-obs': bias[:,0,3].mean(), 'unet-obs': bias[:,1,3].mean(), 
        'era5-obs': bias[:,2,3].mean(), 'geometry': gfs_st[gfs_st.station == st].geometry.values[0]})

    bias_all.append({'station': st, 'gfs-obs': bias[:,0,:], 'unet-obs': bias[:,1,:], 
        'era5-obs': bias[:,2,:], 'geometry': gfs_st[gfs_st.station == st].geometry.values[0]})
    print(f'Total time: {time() - t00} seconds for one station')

gdf_00 = gpd.GeoDataFrame(bias_all_00)
gdf_00.set_crs('epsg:4326', inplace=True)
gdf_00.to_file(f'2024/bias_cycle{cycle:02d}_hour06_{startdate.strftime("%Y%m%d%H")}-{(enddate-timedelta(hours=6)).strftime("%Y%m%d%H")}.shp')

gdf_06 = gpd.GeoDataFrame(bias_all_06)
gdf_06.set_crs('epsg:4326', inplace=True)
gdf_06.to_file(f'2024/bias_cycle{cycle:02d}_hour12_{startdate.strftime("%Y%m%d%H")}-{(enddate-timedelta(hours=6)).strftime("%Y%m%d%H")}.shp')

gdf_12 = gpd.GeoDataFrame(bias_all_12)
gdf_12.set_crs('epsg:4326', inplace=True)
gdf_12.to_file(f'2024/bias_cycle{cycle:02d}_hour18_{startdate.strftime("%Y%m%d%H")}-{(enddate-timedelta(hours=6)).strftime("%Y%m%d%H")}.shp')

gdf_18 = gpd.GeoDataFrame(bias_all_18)
gdf_18.set_crs('epsg:4326', inplace=True)
gdf_18.to_file(f'2024/bias_cycle{cycle:02d}_hour24_{startdate.strftime("%Y%m%d%H")}-{(enddate-timedelta(hours=6)).strftime("%Y%m%d%H")}.shp')

gdf_all = gpd.GeoDataFrame(bias_all)
gdf_all.set_crs('epsg:4326', inplace=True)
gdf_all.to_pickle(f'2024/bias_cycle{cycle:02d}_{startdate.strftime("%Y%m%d%H")}-{(enddate-timedelta(hours=6)).strftime("%Y%m%d%H")}.pkl')

print(f'Total time: {(time() - t0)/60} mins')
