#!/bin/bash
#SBATCH --job-name=WCMC_PostopAllECG
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=g34
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
#SBATCH --time=08:00:00
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
SCRIPT=/athena/madelab/scratch/iqh4001/ECG/ECG-gender/Code/WCMC_ECGSex_Postop_AllPreopECG_Frozen5Seed.py
OUTDIR=/athena/madelab/scratch/iqh4001/ECG/ECG-gender/Results/WCMC_PostopOutcome_AllPreopECG_Frozen5Seed

echo "===================================================================================================="
echo "WCMC ECG-SEX — ALL PREOPERATIVE ECGs / FROZEN FINAL 5-SEED ENSEMBLE"
echo "===================================================================================================="
echo "JOB ID               : ${SLURM_JOB_ID}"
echo "NODE                 : $(hostname)"
echo "DATE                 : $(date)"
echo "PYTHON               : ${PYTHON}"
echo "SCRIPT               : ${SCRIPT}"
echo "OUTDIR               : ${OUTDIR}"
echo "CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "===================================================================================================="

if [ ! -f "${SCRIPT}" ]; then
    echo "FATAL: missing script: ${SCRIPT}"
    exit 1
fi

echo
nvidia-smi

"${PYTHON}" - <<'PY'
import sys, torch, duckdb, pandas as pd, numpy as np, statsmodels
print("Python      :", sys.version)
print("Torch       :", torch.__version__)
print("CUDA build  :", torch.version.cuda)
print("CUDA visible:", torch.cuda.is_available())
print("GPU count   :", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("FATAL: CUDA unavailable")
print("GPU         :", torch.cuda.get_device_name(0))
print("DuckDB      :", duckdb.__version__)
print("pandas      :", pd.__version__)
print("numpy       :", np.__version__)
print("statsmodels :", statsmodels.__version__)
PY

export VCGFINAL_MODE=FINAL
export VCGFINAL_BATCH_SIZE=128
export VCGFINAL_NUM_WORKERS=8

"${PYTHON}" -u "${SCRIPT}" \
    --outdir "${OUTDIR}" \
    --batch-size 128 \
    --num-workers 8

echo
echo "===================================================================================================="
echo "ALL-PREOPERATIVE-ECG POSTOPERATIVE ANALYSIS COMPLETE"
echo "DATE   : $(date)"
echo "OUTDIR : ${OUTDIR}"
echo "===================================================================================================="