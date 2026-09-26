#!/bin/bash
#SBATCH --job-name=BreastLeadSens
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --exclude=g34
#SBATCH --output=/athena/madelab/scratch/iqh4001/ECG/ECG-gender/logs/%x-%j.out
#SBATCH --error=/athena/madelab/scratch/iqh4001/ECG/ECG-gender/logs/%x-%j.err

set -eo pipefail
set +u

LOGDIR=/athena/madelab/scratch/iqh4001/ECG/ECG-gender/logs
mkdir -p "${LOGDIR}"

source ~/.bashrc
conda activate fed_env
set -u

PYTHON=/home/iqh4001/.conda/envs/fed_env/bin/python
SCRIPT=/athena/madelab/scratch/iqh4001/ECG/ECG-gender/Code/ECGSex_BreastPlacement_GroupedLeadSensitivity_WCMC.py

export BREASTABL_N_BOOT=2000
export BREASTABL_BATCH_SIZE=96
export BREASTABL_NUM_WORKERS=8
export BREASTABL_FORCE=0

echo "===================================================================================================="
echo "ECG-SEX — BREAST / PRECORDIAL-PLACEMENT GROUPED LEAD SENSITIVITY"
echo "===================================================================================================="
echo "JOB ID              : ${SLURM_JOB_ID}"
echo "NODE                : $(hostname)"
echo "DATE                : $(date)"
echo "PYTHON              : ${PYTHON}"
echo "SCRIPT              : ${SCRIPT}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-UNSET}"
echo "N_BOOT              : ${BREASTABL_N_BOOT}"
echo "BATCH_SIZE          : ${BREASTABL_BATCH_SIZE}"
echo "NUM_WORKERS         : ${BREASTABL_NUM_WORKERS}"
echo "LOGDIR              : ${LOGDIR}"
echo "===================================================================================================="

if [ ! -f "${SCRIPT}" ]; then
    echo "FATAL: missing Python script: ${SCRIPT}"
    exit 1
fi

nvidia-smi

"${PYTHON}" - <<'PY'
import os
import sys
import torch

print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    print("FATAL: CUDA unavailable.")
    sys.exit(1)

print("GPU:", torch.cuda.get_device_name(0))
PY

echo
echo "Starting grouped-lead sensitivity analysis..."
echo

"${PYTHON}" -u "${SCRIPT}"

echo
echo "===================================================================================================="
echo "BREAST / PRECORDIAL-PLACEMENT SENSITIVITY COMPLETE"
echo "DATE: $(date)"
echo "===================================================================================================="