#!/bin/bash
#SBATCH --job-name=WCMCOnly2x2Final
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/athena/madelab/scratch/iqh4001/ECG/ECG-gender/logs/%x-%j.out
#SBATCH --error=/athena/madelab/scratch/iqh4001/ECG/ECG-gender/logs/%x-%j.err

set -eo pipefail
set +u

mkdir -p /athena/madelab/scratch/iqh4001/ECG/ECG-gender/logs

source ~/.bashrc
conda activate fed_env
set -u

PYTHON=/home/iqh4001/.conda/envs/fed_env/bin/python
SCRIPT=/athena/madelab/scratch/iqh4001/ECG/ECG-gender/Code/09_WCMCOnly_VCG_vs_ResNet_5Seed_2x2.py

export WCMCONLY_MODE=FINAL
export WCMCONLY_NUM_WORKERS=0
export WCMCONLY_N_BOOT=2000

"${PYTHON}" -m py_compile "${SCRIPT}"
"${PYTHON}" -u "${SCRIPT}"
