import os, sys
import socket
from datetime import datetime, timedelta
import time
import multiprocessing as mp
import pathlib

import numpy as np

from gfs import get_gfs_data_by_date

if __name__ == '__main__':

    startdate = datetime(2021, 3, 23)
    enddate = datetime(2041, 1, 1) #excluded
    datevector = np.arange(startdate, enddate, np.timedelta64(6, 'h')).astype(datetime)
    
    outdir = pathlib.Path('/scratch1/NCEPDEV/nems/Linlin.Cui/Tests/ML4BC/data')
    outdir.mkdir(parents=True, exist_ok=True)

    download_dir = 'noaa-gfs-bdp-pds-data'
    
    if socket.gethostname().startswith('hfe'):
        npool = len(datevector) if len(datevector) < mp.cpu_count()/4 else mp.cpu_count()/4
    else:
        npool = len(datevector) if len(datevector) < mp.cpu_count()/2 else mp.cpu_count()/2
    print(f'Using {npool} processors!')

    pool = mp.Pool(int(npool))

    t0 = time.time()
    pool.starmap(get_gfs_data_by_date, [(date, outdir, download_dir) for date in datevector])
    pool.close()
    print(f'Total time: {time.time() - t0} s')

    #remove downloaded files
    try:
        os.system(f"rm -rf {download_dir}")
        print("Downloaded data removed.")
    except Exception as e:
        print(f"Error removing downloaded data: {str(e)}")
