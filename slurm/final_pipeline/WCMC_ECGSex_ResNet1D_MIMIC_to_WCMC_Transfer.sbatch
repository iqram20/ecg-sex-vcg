#!/bin/bash
#SBATCH --job-name=ResNet_M2W
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=/athena/madelab/scratch/iqh4001/ECG/ECG-gender/logs/%x-%j.out
#SBATCH --error=/athena/madelab/scratch/iqh4001/ECG/ECG-gender/logs/%x-%j.err

set -eo pipefail
set +u

mkdir -p /athena/madelab/scratch/iqh4001/ECG/ECG-gender/logs

source ~/.bashrc
conda activate fed_env
set -u

PYTHON=/home/iqh4001/.conda/envs/fed_env/bin/python
SCRIPT=/athena/madelab/scratch/iqh4001/ECG/ECG-gender/Code/WCMC_ECGSex_ResNet1D_MIMIC_to_WCMC_Transfer.py

# MIMIC pretraining — same protocol used for the transfer comparison
export RESNET_TRANSFER_MIMIC_EPOCHS=30
export RESNET_TRANSFER_MIMIC_PATIENCE=7
export RESNET_TRANSFER_MIMIC_BATCH_SIZE=128
export RESNET_TRANSFER_LR_MIMIC=3e-4
export RESNET_TRANSFER_FORCE_MIMIC=0

# WCMC full-model fine-tuning — matched to final TRiSM transfer
export RESNET_TRANSFER_WCMC_EPOCHS=25
export RESNET_TRANSFER_WCMC_PATIENCE=5
export RESNET_TRANSFER_WCMC_BATCH_SIZE=128
export RESNET_TRANSFER_LR_WCMC=1e-5
export RESNET_TRANSFER_FORCE_WCMC=0

# Common
export RESNET_TRANSFER_NUM_WORKERS=8
export RESNET_TRANSFER_N_BOOT=2000

echo "===================================================================================================="
echo "ECG-SEX — RESNET1D MIMIC -> WCMC TRANSFER"
echo "===================================================================================================="
echo "JOB ID : $SLURM_JOB_ID"
echo "NODE   : $SLURMD_NODENAME"
echo "DATE   : $(date)"
echo "PWD    : $(pwd)"
echo "SCRIPT : $SCRIPT"
echo "===================================================================================================="

echo ""
echo "Python:"
$PYTHON --version

echo ""
echo "Torch/CUDA:"
$PYTHON - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU Count:", torch.cuda.device_count())
PY

if [ ! -f "$SCRIPT" ]; then
    echo ""
    echo "ERROR: Python script not found:"
    echo "$SCRIPT"
    exit 1
fi

echo ""
echo "Configuration:"
echo "  MIMIC epochs/patience : $RESNET_TRANSFER_MIMIC_EPOCHS / $RESNET_TRANSFER_MIMIC_PATIENCE"
echo "  MIMIC batch / LR      : $RESNET_TRANSFER_MIMIC_BATCH_SIZE / $RESNET_TRANSFER_LR_MIMIC"
echo "  WCMC epochs/patience  : $RESNET_TRANSFER_WCMC_EPOCHS / $RESNET_TRANSFER_WCMC_PATIENCE"
echo "  WCMC batch / LR       : $RESNET_TRANSFER_WCMC_BATCH_SIZE / $RESNET_TRANSFER_LR_WCMC"
echo "  Bootstrap             : $RESNET_TRANSFER_N_BOOT"

echo ""
nvidia-smi

echo ""
echo "===================================================================================================="
echo "STARTING RESNET1D MIMIC PRETRAINING -> WCMC FINE-TUNING"
echo "===================================================================================================="

$PYTHON -u "$SCRIPT"

echo ""
echo "===================================================================================================="
echo "RESNET1D TRANSFER RUN COMPLETE"
echo "DATE: $(date)"
echo "===================================================================================================="
