'''
Description: a utility for creating and submitting job cards to run gen_training_X.Xd.py scripts in parallel.
Author: Sadegh Sadeghi Tabas (sadegh.tabas@noaa.gov)
Revision history:
    -20231106: Sadegh Tabas, initial code
'''
import os
from datetime import datetime, timedelta

# Define the start and end dates
start_date = datetime(2021, 3, 23)
end_date = datetime(2023, 10, 31)
delta = timedelta(days=5)  # Interval of 10 days
process = 'gfs'
resolution = '1.0'

# Template of the job card
job_card_template = """#!/bin/bash
#SBATCH -J {process}_process
#SBATCH -o {process}_output_{start}_{end}.out
#SBATCH -e {process}_err_{start}_{end}.err
#SBATCH -N 1
#SBATCH -t 30-00:00:00  # 1 month (30 days)

# Load Miniconda module
source ~/.bashrc

# Activate the Conda environment
conda activate ml4bc

# Navigate to the working directory
cd /lustre/ML4BC

# Run the Python script
python3 gen_training_{resolution}d.py {start} {end} -p {process} -o /contrib/$USER/ML4BC/
"""

# Generate and submit job cards
while start_date <= end_date:
    next_date = start_date + delta
    job_card = job_card_template.format(process=process, start=start_date.strftime("%Y%m%d"), end=next_date.strftime("%Y%m%d"), resolution=resolution)
    
    # Write the job card to a file
    job_filename = f"{process}_job_{start_date.strftime('%Y%m%d')}_{next_date.strftime('%Y%m%d')}.sh"
    with open(job_filename, 'w') as job_file:
        job_file.write(job_card)
    
    # Submit the job
    os.system(f"sbatch {job_filename}")
    
    start_date = next_date + timedelta(days=1) 
