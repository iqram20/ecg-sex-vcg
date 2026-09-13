#!/bin/bash
#SBATCH --job-name=VCGExtFinal
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/athena/madelab/scratch/iqh4001/ECG/ECG-gender/logs/%x-%j.out
#SBATCH --error=/athena/madelab/scratch/iqh4001/ECG/ECG-gender/logs/%x-%j.err

set -eo pipefail
set +u

mkdir -p /athena/madelab/scratch/iqh4001/ECG/ECG-gender/logs

source ~/.bashrc
conda activate fed_env
set -u

export VCGEXT_MODE=FINAL
export VCGEXT_NUM_WORKERS=0
export VCGEXT_N_BOOT=2000

/home/iqh4001/.conda/envs/fed_env/bin/python -u \
/athena/madelab/scratch/iqh4001/ECG/ECG-gender/Code/WCMC_ECGSex_VCGGeometry_ResNet_ExternalConfirmation.py
