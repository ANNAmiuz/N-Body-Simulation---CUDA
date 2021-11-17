#!/bin/bash
#SBATCH --account=csc4005
#SBATCH --partition=debug
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH -t 10

echo "mainmode: " && /bin/hostname
srun testCUDA 20 0
srun testCUDA 200 0