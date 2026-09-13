#!/bin/bash
#SBATCH --job-name=VCGFinalSeed
#SBATCH --partition=scu-gpu
#SBATCH --array=0-4%5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --exclude=g34
#SBATCH --output=/athena/madelab/scratch/iqh4001/ECG/ECG-gender/logs/%x-%A_%a.out
#SBATCH --error=/athena/madelab/scratch/iqh4001/ECG/ECG-gender/logs/%x-%A_%a.err

set -eo pipefail
set +u

source ~/.bashrc
conda activate fed_env
set -u

PYTHON=/home/iqh4001/.conda/envs/fed_env/bin/python
SCRIPT=/athena/madelab/scratch/iqh4001/ECG/ECG-gender/Code/WCMC_ECGSex_VCGGeometry_ResNet_5Seed_LockedTest.py

SEEDS=(11 22 33 42 55)

export VCGFINAL_MODE=RUN
export VCGFINAL_SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

export VCGFINAL_BATCH_SIZE=64
export VCGFINAL_NUM_WORKERS=8
export VCGFINAL_MAX_EPOCHS=25
export VCGFINAL_PATIENCE=5
export VCGFINAL_LR_BASE=1e-5
export VCGFINAL_LR_NEW=1e-4
export VCGFINAL_MAX_DOWER_DELTA=0.03
export VCGFINAL_DOWER_REG=1e-3
export VCGFINAL_N_BOOT=2000

echo "===================================================================================================="
echo "ECG-SEX VCG-GEOMETRY RESNET — FINAL 5-SEED CONFIRMATION"
echo "TASK=$SLURM_ARRAY_TASK_ID SEED=$VCGFINAL_SEED"
echo "NODE=$SLURMD_NODENAME DATE=$(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-UNSET}"
echo "===================================================================================================="

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
