#!/bin/bash
#SBATCH --job-name=PTBVCGEff
#SBATCH --partition=scu-gpu
#SBATCH --array=0-11%6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --exclude=g34
#SBATCH --output=/athena/madelab/scratch/iqh4001/ECG/ECG-gender/logs/%x-%A_%a.out
#SBATCH --error=/athena/madelab/scratch/iqh4001/ECG/ECG-gender/logs/%x-%A_%a.err

set -eo pipefail
set +u

source ~/.bashrc
conda activate fed_env
set -u

PYTHON=/home/iqh4001/.conda/envs/fed_env/bin/python
SCRIPT=/athena/madelab/scratch/iqh4001/ECG/ECG-gender/Code/PTBXL_ECGSex_VCGGeometry_DataEfficiency.py

FRACTIONS=(0.10 0.25 0.50 1.00)
SEEDS=(11 22 42)

FRAC_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % 3))

export PTB_VCG_MODE=RUN
export PTB_VCG_FRACTION="${FRACTIONS[$FRAC_INDEX]}"
export PTB_VCG_SEED="${SEEDS[$SEED_INDEX]}"

export PTB_VCG_BATCH_SIZE=96
export PTB_VCG_NUM_WORKERS=8
export PTB_VCG_MAX_EPOCHS=30
export PTB_VCG_PATIENCE=5
export PTB_VCG_LR_LOCAL=3e-4
export PTB_VCG_LR_TRANSFER=1e-5
export PTB_VCG_N_BOOT=2000

echo "===================================================================================================="
echo "PTB-XL FINAL VCG DATA EFFICIENCY"
echo "TASK=$SLURM_ARRAY_TASK_ID FRACTION=$PTB_VCG_FRACTION SEED=$PTB_VCG_SEED"
echo "NODE=$(hostname) DATE=$(date)"
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
