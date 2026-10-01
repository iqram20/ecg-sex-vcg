#!/bin/bash
#SBATCH --job-name=WCMCOnly2x2
#SBATCH --partition=scu-gpu
#SBATCH --array=0-4%5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=18:00:00
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
SCRIPT=/athena/madelab/scratch/iqh4001/ECG/ECG-gender/Code/09_WCMCOnly_VCG_vs_ResNet_5Seed_2x2.py

SEEDS=(11 22 33 42 55)

export WCMCONLY_MODE=RUN
export WCMCONLY_SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

export WCMCONLY_BATCH_SIZE=64
export WCMCONLY_NUM_WORKERS=8
export WCMCONLY_MAX_EPOCHS=30
export WCMCONLY_PATIENCE=5
export WCMCONLY_LR=3e-4
export WCMCONLY_WEIGHT_DECAY=1e-4
export WCMCONLY_GRAD_CLIP=5.0
export WCMCONLY_MAX_DOWER_DELTA=0.03
export WCMCONLY_DOWER_REG=1e-3
export WCMCONLY_N_BOOT=2000

echo "===================================================================================================="
echo "WCMC-ONLY VCG vs RESNET — 5-SEED REVIEWER ANALYSIS"
echo "TASK=$SLURM_ARRAY_TASK_ID SEED=$WCMCONLY_SEED"
echo "NODE=$SLURMD_NODENAME DATE=$(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-UNSET}"
echo "===================================================================================================="

"${PYTHON}" -m py_compile "${SCRIPT}"

nvidia-smi

"${PYTHON}" - <<'PY'
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

"${PYTHON}" -u "${SCRIPT}"
