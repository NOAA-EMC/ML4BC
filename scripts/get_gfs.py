import os, sys
import socket
from datetime import datetime, timedelta
import time
import multiprocessing as mp

import numpy as np

from gfs import get_gfs_data_by_date

if __name__ == '__main__':

    startdate = datetime(2021, 3, 23)
    enddate = datetime(2024, 1, 1)
    datevector = np.arange(startdate, enddate, np.timedelta64(6, 'h')).astype(datetime)
    
    outdir = 'GFS'
    
    if socket.gethostname().startswith('hfe'):
        npool = len(datevector) if len(datevector) < mp.cpu_count()/4 else mp.cpu_count()/4
    else:
        npool = len(datevector) if len(datevector) < mp.cpu_count()/2 else mp.cpu_count()/2
    print(f'Using {npool} processors!')

    pool = mp.Pool(int(npool))

    t0 = time.time()
    pool.starmap(get_gfs_data_by_date, [(date, outdir) for date in datevector])
    pool.close()
    print(f'Total time: {time.time() - t0} s')
