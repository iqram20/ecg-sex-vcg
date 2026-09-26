#!/bin/bash
#SBATCH --job-name=ecgsex_bmi
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=ecgsex_bmi_%j.out
#SBATCH --error=ecgsex_bmi_%j.err

set -euo pipefail

PYTHON=/home/iqh4001/.conda/envs/fed_env/bin/python
SCRIPT=/athena/madelab/scratch/iqh4001/ECG/ECG-gender/Code/ECGSex_BMI_Sensitivity_WCMC.py
OUTDIR=/athena/madelab/scratch/iqh4001/ECG/ECG-gender/Results/BMI_sensitivity_WCMC_locked_test

PRED_SOURCE=${PRED_SOURCE:-""}

if [[ -z "${PRED_SOURCE}" ]]; then
    echo "ERROR: set PRED_SOURCE to the final VCG locked-test prediction file or final run directory."
    echo ""
    echo "Example:"
    echo "sbatch --export=ALL,PRED_SOURCE=/path/to/final_VCG_run ECGSex_BMI_Sensitivity_WCMC.sbatch"
    exit 1
fi

echo "==============================================================="
echo "WCMC/NYP ECG-SEX BMI SENSITIVITY"
echo "==============================================================="
echo "JOB ID      : ${SLURM_JOB_ID}"
echo "NODE        : $(hostname)"
echo "DATE        : $(date)"
echo "PRED_SOURCE : ${PRED_SOURCE}"
echo "OUTDIR      : ${OUTDIR}"
echo "==============================================================="

${PYTHON} "${SCRIPT}" \
    --predictions "${PRED_SOURCE}" \
    --threshold 0.484619 \
    --n-boot 2000 \
    --bmi-caliper 1.0 \
    --outdir "${OUTDIR}"

echo "DONE: $(date)"