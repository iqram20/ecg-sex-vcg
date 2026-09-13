#!/bin/bash
#SBATCH --job-name=VCGExt
#SBATCH --partition=scu-gpu
#SBATCH --array=0-1%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --exclude=g34
#SBATCH --output=/athena/madelab/scratch/iqh4001/ECG/ECG-gender/logs/%x-%A_%a.out
#SBATCH --error=/athena/madelab/scratch/iqh4001/ECG/ECG-gender/logs/%x-%A_%a.err

set -eo pipefail
set +u

mkdir -p /athena/madelab/scratch/iqh4001/ECG/ECG-gender/logs

source ~/.bashrc
conda activate fed_env
set -u

PYTHON=/home/iqh4001/.conda/envs/fed_env/bin/python
SCRIPT=/athena/madelab/scratch/iqh4001/ECG/ECG-gender/Code/WCMC_ECGSex_VCGGeometry_ResNet_ExternalConfirmation.py

MODES=(PTBXL CHAPMAN)

export VCGEXT_MODE="${MODES[$SLURM_ARRAY_TASK_ID]}"
export VCGEXT_BATCH_SIZE=128
export VCGEXT_NUM_WORKERS=8
export VCGEXT_N_BOOT=2000

echo "===================================================================================================="
echo "ECG-SEX VCG-GEOMETRY RESNET — ZERO-SHOT EXTERNAL CONFIRMATION"
echo "TASK=$SLURM_ARRAY_TASK_ID MODE=$VCGEXT_MODE"
echo "NODE=$SLURMD_NODENAME DATE=$(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-UNSET}"
echo "===================================================================================================="

if [ ! -f "$SCRIPT" ]; then
    echo "FATAL: missing script: $SCRIPT"
    exit 1
fi

nvidia-smi

$PYTHON - <<'PY'
import os
import sys
import torch

print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    print("FATAL: CUDA unavailable.")
    sys.exit(1)

print("GPU:", torch.cuda.get_device_name(0))
PY

$PYTHON -u "$SCRIPT"
