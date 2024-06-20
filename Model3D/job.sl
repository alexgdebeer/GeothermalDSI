#!/bin/bash

#SBATCH --job-name=FaultDSI
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=2GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-995		# Model indices

#SBATCH --mail-user=adeb970@aucklanduni.ac.nz
#SBATCH --mail-type=ALL

cd /nesi/nobackup/uoa00463/Geothermal_projects/SyntheticAlex/Geothermal/

srun /nesi/project/uoa00463/bin/waiwera-1.4.0 models/FL8788_${SLURM_ARRAY_TASK_ID}_NS.json 
srun /nesi/project/uoa00463/bin/waiwera-1.4.0 models/FL8788_${SLURM_ARRAY_TASK_ID}_PR.json 