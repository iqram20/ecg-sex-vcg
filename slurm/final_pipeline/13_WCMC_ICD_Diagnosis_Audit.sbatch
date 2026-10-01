#!/bin/bash
#SBATCH --job-name=WCMC_ICD_Audit
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/athena/madelab/scratch/iqh4001/ECG/ECG-gender/logs/%x-%j.out
#SBATCH --error=/athena/madelab/scratch/iqh4001/ECG/ECG-gender/logs/%x-%j.err

set -eo pipefail
set +u

LOGDIR=/athena/madelab/scratch/iqh4001/ECG/ECG-gender/logs
mkdir -p "$LOGDIR"

source ~/.bashrc
conda activate fed_env

set -u

PYTHON=/home/iqh4001/.conda/envs/fed_env/bin/python
SCRIPT=/athena/madelab/scratch/iqh4001/ECG/ECG-gender/Code/13_WCMC_ICD_Diagnosis_Audit.py

echo "================================================================================"
echo "WCMC ICD / DIAGNOSIS AUDIT"
echo "================================================================================"
echo "JOB ID : ${SLURM_JOB_ID}"
echo "NODE   : $(hostname)"
echo "DATE   : $(date)"
echo "PYTHON : ${PYTHON}"
echo "SCRIPT : ${SCRIPT}"
echo "OUT    : ${LOGDIR}/WCMC_ICD_Audit-${SLURM_JOB_ID}.out"
echo "ERR    : ${LOGDIR}/WCMC_ICD_Audit-${SLURM_JOB_ID}.err"
echo "================================================================================"

if [ ! -f "$SCRIPT" ]; then
    echo "FATAL: missing script: $SCRIPT"
    exit 1
fi

"$PYTHON" - <<PY
from pathlib import Path
p = Path("$SCRIPT")
compile(p.read_text(), str(p), "exec")
print("Python syntax OK:", p)
PY

"$PYTHON" -u "$SCRIPT"

echo
echo "================================================================================"
echo "AUDIT COMPLETE"
echo "DATE: $(date)"
echo "================================================================================"
