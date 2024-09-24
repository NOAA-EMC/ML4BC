'''
Description: This script is to use model weights correcting bias.
8/23/2024, Linlin Cui(linlin.cui@noaa.gov)
'''

from datetime import datetime, timedelta
import pathlib

import torch
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from models.SmaAt_UNet import SmaAt_UNet

def normalise(data, reverse=False):
    mean = 288.34174
    std = 10.80314

    if reverse:
        data =  data * std + mean
    else:
        data = (data - mean) / std

    return data


def load_model(checkpoint_file, device):
    model = SmaAt_UNet(n_channels=1, n_classes=1)
    model.zero_grad()

    checkpoint = torch.load(checkpoint_file, map_location=device)

    try:
        new_state_dict = dict()
        for k, v in checkpoint['state_dict'].items():
            #name = k[7:]
            new_state_dict[k[7:]] = v
        model.load_state_dict(new_state_dict)
    except Exception:
        model.load_state_dict(checkpoint['state_dict'])

    return model

def get_data(ds):

    data = []
    #ds1 - GFS f072 t2m
    ds_slice = ds.sel(latitude=slice(50, 25), longitude=slice(230, 300))  #slice for NA region
    values = np.squeeze(ds_slice.t2m.values.astype(np.float32))  #select f072
    data = normalise(values, False)  # Normalize the data if transform is True
    ds_slice.close()


    data_tensor = torch.from_numpy(data) if not isinstance(data, torch.Tensor) else data
    data_tensor = data_tensor.unsqueeze(0).unsqueeze(0)

    return data_tensor


if __name__ == '__main__':

    #device = 'cuda:0'
    device = 'cpu'

    cpt_file = 'checkpoints/best_model_SmaAt_UNet.pt'

    model = load_model(cpt_file, device)
    model.eval()

    date1 = datetime(2024, 1, 1)
    date2 = datetime(2024, 4, 21)
    xmin, xmax, ymin, ymax = 230, 300, 25, 50
    datevector = np.arange(date1, date2, np.timedelta64(6, 'h')).astype(datetime)

    run = 'run27'
    ftime=72

    y_pred, y_input, y_label = [], [], []
    rmse1 = []
    rmse2 = []
    count = 0

    data_dir = '/scratch1/NCEPDEV/nems/Linlin.Cui/Tests/ML4BC/run27/data'

    outdir = pathlib.Path('./outputs')
    outdir.mkdir(parents=True, exist_ok=True)

    for date in datevector:

        fname = f'{data_dir}/GFS.{date.strftime("%Y%m%d%H")}.nc'

        ds = xr.open_dataset(fname) #GFS

        t2m_gfs = np.squeeze(ds.t2m.sel(latitude=slice(ymax, ymin), longitude=slice(xmin, xmax)).values)
        np.save(f'{outdir}/t2m_gfs_{date.strftime("%Y%m%d%H")}.npy', t2m_gfs)
    
        input_tensor = get_data(ds)


        with torch.no_grad(): 
            output = model(input_tensor)

        output_numpy = output.cpu().numpy()

        t2m_corrected = normalise(output_numpy, reverse=True)
        np.save(f'{outdir}/t2m_corrected_{date.strftime("%Y%m%d%H")}.npy', t2m_corrected)

        ds = xr.open_dataset(f'{data_dir}/ERA5.{date.strftime("%Y%m%d%H")}.nc')
        ds_slice = ds.t2m.sel(latitude=slice(ymax, ymin), longitude=slice(xmin, xmax))
        lon = ds_slice.longitude.values
        lat = ds_slice.latitude.values
        t2m_era5 = ds_slice.values.astype(np.float32)
        np.save(f'{outdir}/t2m_era5_{date.strftime("%Y%m%d%H")}.npy', t2m_era5)
        ds.close()

        y_pred.append(t2m_corrected.mean(axis=(2, 3)).item())
        y_input.append(t2m_gfs.mean())
        y_label.append(t2m_era5.mean())
        tmp1 = np.sqrt(np.mean((np.squeeze(t2m_corrected[0,0,:,:]) - t2m_era5)**2))
        rmse1.append(tmp1)
        tmp2 = np.sqrt(np.mean((t2m_gfs - t2m_era5)**2))
        rmse2.append(tmp2)
        if tmp1 < tmp2: count += 1

    y_pred = np.array(y_pred)
    y_input = np.array(y_input)
    y_label = np.array(y_label)

    rmse1 = np.array(rmse1)
    rmse2 = np.array(rmse2)

    x = np.arange(len(y_pred))
    print(f'Improved percentage (%) for: {count/len(y_pred)*100}, rmse1: {rmse1.mean()}, rmse2: {rmse2.mean()}')
    plt.plot(x, rmse1, label='gfs-corrected vs era5')
    plt.plot(x, rmse2, label='gfs vs era5')
    #plt.plot(x, y_pred, label='gfs-corrected')
    #plt.plot(x, y_input, label='gfs')
    #plt.plot(x, y_label, label='era5')
    plt.xlabel('Number of forecast cycles')
    #plt.ylabel('T2m (K)')
    plt.ylabel('T2m RMSE (K)')
    plt.legend()
    plt.savefig(f'comparison_unet_{run}_f{ftime:03d}_rmse_{date1.strftime("%Y%m%d%H")}-{date2.strftime("%Y%m%d%H")}.png')
    plt.close()
