#!/bin/bash
#SBATCH --job-name=VCGInterp
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=24:00:00
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
SCRIPT=/athena/madelab/scratch/iqh4001/ECG/ECG-gender/Code/WCMC_ECGSex_VCGGeometry_ResNet_FinalInterpretability.py

export VCGINTERP_MODE=ALL
export VCGINTERP_FORCE=0

export VCGINTERP_BATCH_SIZE=96
export VCGINTERP_NUM_WORKERS=8
export VCGINTERP_N_BOOT=2000

export VCGINTERP_IG_N=256
export VCGINTERP_IG_STEPS=24
export VCGINTERP_IG_BATCH_SIZE=8

echo "===================================================================================================="
echo "ECG-SEX — FINAL VCG-GEOMETRY RESNET INTERPRETABILITY"
echo "NODE: $(hostname)"
echo "DATE: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-UNSET}"
echo "MODE=$VCGINTERP_MODE"
echo "===================================================================================================="

if [ ! -f "$SCRIPT" ]; then
    echo "FATAL: missing Python script: $SCRIPT"
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

echo
echo "Starting full restartable interpretability pipeline..."
echo

$PYTHON -u "$SCRIPT"

echo
echo "===================================================================================================="
echo "INTERPRETABILITY PIPELINE COMPLETE"
echo "DATE: $(date)"
echo "===================================================================================================="
