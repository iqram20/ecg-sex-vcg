#!/bin/bash
#SBATCH --job-name=AgeRegionalLead
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/athena/madelab/scratch/iqh4001/ECG/ECG-gender/logs/%x-%j.out
#SBATCH --error=/athena/madelab/scratch/iqh4001/ECG/ECG-gender/logs/%x-%j.err

set -eo pipefail
set +u

mkdir -p /athena/madelab/scratch/iqh4001/ECG/ECG-gender/logs

source ~/.bashrc
conda activate fed_env
set -u

PYTHON=/home/iqh4001/.conda/envs/fed_env/bin/python
SCRIPT=/athena/madelab/scratch/iqh4001/ECG/ECG-gender/Code/10_WCMC_AgeStratified_RegionalLeadAblation.py

"${PYTHON}" -m py_compile "${SCRIPT}"
"${PYTHON}" -u "${SCRIPT}"
