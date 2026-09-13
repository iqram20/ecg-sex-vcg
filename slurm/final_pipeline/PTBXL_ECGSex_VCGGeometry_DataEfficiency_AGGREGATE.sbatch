#!/bin/bash
#SBATCH --job-name=PTBVCGAgg
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/athena/madelab/scratch/iqh4001/ECG/ECG-gender/logs/%x-%j.out
#SBATCH --error=/athena/madelab/scratch/iqh4001/ECG/ECG-gender/logs/%x-%j.err

set -eo pipefail
set +u

source ~/.bashrc
conda activate fed_env
set -u

export PTB_VCG_MODE=AGGREGATE
export PTB_VCG_N_BOOT=2000
export PTB_VCG_NUM_WORKERS=0

/home/iqh4001/.conda/envs/fed_env/bin/python -u \
/athena/madelab/scratch/iqh4001/ECG/ECG-gender/Code/PTBXL_ECGSex_VCGGeometry_DataEfficiency.py
