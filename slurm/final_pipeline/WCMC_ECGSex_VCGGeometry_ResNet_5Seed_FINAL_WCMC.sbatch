#!/bin/bash
#SBATCH --job-name=VCGFinalWCMC
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

export VCGFINAL_MODE=FINAL
export VCGFINAL_NUM_WORKERS=0
export VCGFINAL_N_BOOT=2000

/home/iqh4001/.conda/envs/fed_env/bin/python -u \
/athena/madelab/scratch/iqh4001/ECG/ECG-gender/Code/WCMC_ECGSex_VCGGeometry_ResNet_5Seed_LockedTest.py
