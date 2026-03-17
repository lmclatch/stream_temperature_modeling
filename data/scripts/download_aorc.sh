#!/bin/bash

#SBATCH --account=dtn

#SBATCH --partition=notchpeak-dtn

#SBATCH --qos=notchpeak-dtn

#SBATCH --time=06:00:00

#SBATCH --ntasks=1

#SBATCH --mem=8G

#SBATCH -o slurmjob-%j.out-%N

#SBATCH -e slurmjob-%j.err-%N

#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --mail-user=u1566670@utah.edu



module load miniconda3/25.9.1

conda activate alaska



echo "Job started: $(date)"

echo "Running on node: $(hostname)"



python /uufs/chpc.utah.edu/common/home/u1566670/hydroinformatics/stream_temperature_modeling/data/scripts/aorc_download.py



echo "Job finished: $(date)"
