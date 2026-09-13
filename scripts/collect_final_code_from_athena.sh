#!/usr/bin/env bash
set -euo pipefail

CODE_SRC="${CODE_SRC:-/athena/madelab/scratch/iqh4001/ECG/ECG-gender/Code}"

copy_file() {
  src="$CODE_SRC/$1"
  dst="$2/$1"
  if [[ -f "$src" ]]; then
    mkdir -p "$2"
    cp -v "$src" "$dst"
  else
    echo "MISSING: $src" >&2
  fi
}

for f in \
  WCMC_ECGSex_ResNet1D_MIMIC_to_WCMC_Transfer.py \
  WCMC_ECGSex_VCGGeometry_ResNet_ValidationScreen.py \
  WCMC_ECGSex_VCGGeometry_ResNet_5Seed_LockedTest.py; do
  copy_file "$f" scripts/training
done

for f in \
  WCMC_ECGSex_VCGGeometry_ResNet_ExternalConfirmation.py \
  PTBXL_ECGSex_08_DataEfficiency_OneJob.py \
  PTBXL_ECGSex_09_DataEfficiency_Aggregate_and_Figures.py \
  PTBXL_ECGSex_10_DataEfficiency_SeedEnsemble_PairedBootstrap.py \
  WCMC_ECGSex_06_Paired_Bootstrap_ThreeModel.py; do
  copy_file "$f" scripts/evaluation
done

for f in \
  WCMC_ECGSex_VCGGeometry_ResNet_FinalInterpretability.py \
  WCMC_ECGSex_VCGGeometry_ResNet_CorrectedIG.py; do
  copy_file "$f" scripts/interpretability
done

for f in \
  WCMC_ECGSex_ResNet1D_MIMIC_to_WCMC_Transfer.sbatch \
  WCMC_ECGSex_VCGGeometry_ResNet_ValidationScreen_ARRAY.sbatch \
  WCMC_ECGSex_VCGGeometry_ResNet_ValidationScreen_SUMMARY.sbatch \
  WCMC_ECGSex_VCGGeometry_ResNet_5Seed_ARRAY.sbatch \
  WCMC_ECGSex_VCGGeometry_ResNet_5Seed_FINAL_WCMC.sbatch \
  WCMC_ECGSex_VCGGeometry_ResNet_ExternalConfirmation_ARRAY.sbatch \
  WCMC_ECGSex_VCGGeometry_ResNet_ExternalConfirmation_FINAL.sbatch \
  PTBXL_ECGSex_08_DataEfficiency_OneJob.sbatch; do
  copy_file "$f" slurm
done

echo "Final-code collection complete. Review with git diff before committing."
