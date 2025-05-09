#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -p gpu
#SBATCH --gpus=a100:1
#SBATCH --mem=16000
#SBATCH -A cmsc828-class
#SBATCH -t 00:40:00
#SBATCH -o train_output_%j.log    # STDOUT (log file)
#SBATCH -e train_error_%j.log     # STDERR (error log)

# Log job start time
echo "Job started at: $(date)"

module load gcc/11.3.0
# Activate virtual environment
source /home/kalyan/.venv/bin/activate

# Time the execution and log it
SECONDS=0
python FCNVMB_train.py
duration=$SECONDS

echo "Job finished at: $(date)"
echo "Total execution time: $(($duration / 3600))h $((($duration / 60) % 60))m $(($duration % 60))s"
