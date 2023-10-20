#!/bin/bash
#SBATCH -J gfs_process
#SBATCH -o /contrib/Sadegh.Tabas/ML4BC/era5_output.out
#SBATCH -e /contrib/Sadegh.Tabas/ML4BC/era5_err.err
#SBATCH -N 1
#SBATCH -t 30-00:00:00  # 1 month (30 days)

# Load Miniconda module
source ~/.bashrc

# Activate the Conda environment
conda activate ml4bc

# Navigate to the working directory
cd /lustre/ML4BC

# Run the Python script
python3 preprocessing.py 20210323 20220101 -p era5 -o /contrib/Sadegh.Tabas/ML4BC/