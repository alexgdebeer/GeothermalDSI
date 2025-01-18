#!/bin/bash

#SBATCH --job-name=DSI
#SBATCH --time=00:20:00
#SBATCH --mem-per-cpu=2GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=1000-1999		# Model indices
#SBATCH --hint nomultithread
#SBATCH --partition nesi_research
#SBATCH --mem-per-cpu=2G
#SBATCH --output=./job_output/jobSample.stdout
#SBATCH --error=./job_output/jobSample.stderr

#SBATCH --mail-user=adeb970@aucklanduni.ac.nz
#SBATCH --mail-type=ALL

module load cray-python

cd /nesi/nobackup/uoa00463/Geothermal_projects/Model3D

srun /nesi/project/uoa00463/bin/waiwera-1.5.0 models/FL8788_${SLURM_ARRAY_TASK_ID}_NS.json 
srun /nesi/project/uoa00463/bin/waiwera-1.5.0 models/FL8788_${SLURM_ARRAY_TASK_ID}_PR.json 