#!/bin/bash
#SBATCH --job-name=VCGIGFix
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --exclude=g34
#SBATCH --output=/athena/madelab/scratch/iqh4001/ECG/ECG-gender/logs/%x-%j.out
#SBATCH --error=/athena/madelab/scratch/iqh4001/ECG/ECG-gender/logs/%x-%j.err

set -eo pipefail
set +u

mkdir -p /athena/madelab/scratch/iqh4001/ECG/ECG-gender/logs

source ~/.bashrc
conda activate fed_env
set -u

PYTHON=/home/iqh4001/.conda/envs/fed_env/bin/python
SCRIPT=/athena/madelab/scratch/iqh4001/ECG/ECG-gender/Code/WCMC_ECGSex_VCGGeometry_ResNet_CorrectedIG.py

export VCGIG_N=256
export VCGIG_BATCH_SIZE=8
export VCGIG_QUAD_NODES=48
export VCGIG_NUM_WORKERS=0
export VCGIG_FORCE=0

# Strict numerical QC.
export VCGIG_MAX_MEDIAN_ABS_ERROR=0.05
export VCGIG_MAX_P95_ABS_ERROR=0.25

echo "===================================================================================================="
echo "ECG-SEX — FINAL VCG-GEOMETRY RESNET CORRECTED IG"
echo "NODE=$(hostname)"
echo "DATE=$(date)"
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
print("GPU count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    print("FATAL: CUDA unavailable.")
    sys.exit(1)

print("GPU:", torch.cuda.get_device_name(0))
PY

$PYTHON -u "$SCRIPT"

echo
echo "===================================================================================================="
echo "CORRECTED IG COMPLETE"
echo "DATE=$(date)"
echo "===================================================================================================="
