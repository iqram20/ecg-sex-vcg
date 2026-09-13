# Exact final Athena code manifest

The repository should preserve the exact scripts that generated the manuscript analyses. These files exist in the project code directory on Athena and should be copied without rewriting model logic.

## Core model development / validation

- `WCMC_ECGSex_ResNet1D_MIMIC_to_WCMC_Transfer.py`
- `WCMC_ECGSex_ResNet1D_MIMIC_to_WCMC_Transfer.sbatch`
- `WCMC_ECGSex_VCGGeometry_ResNet_ValidationScreen.py`
- `WCMC_ECGSex_VCGGeometry_ResNet_ValidationScreen_ARRAY.sbatch`
- `WCMC_ECGSex_VCGGeometry_ResNet_ValidationScreen_SUMMARY.sbatch`
- `WCMC_ECGSex_VCGGeometry_ResNet_5Seed_LockedTest.py`
- `WCMC_ECGSex_VCGGeometry_ResNet_5Seed_ARRAY.sbatch`
- `WCMC_ECGSex_VCGGeometry_ResNet_5Seed_FINAL_WCMC.sbatch`

## External validation

- `WCMC_ECGSex_VCGGeometry_ResNet_ExternalConfirmation.py`
- `WCMC_ECGSex_VCGGeometry_ResNet_ExternalConfirmation_ARRAY.sbatch`
- `WCMC_ECGSex_VCGGeometry_ResNet_ExternalConfirmation_FINAL.sbatch`

## PTB-XL data efficiency

- `PTBXL_ECGSex_08_DataEfficiency_OneJob.py`
- `PTBXL_ECGSex_08_DataEfficiency_OneJob.sbatch`
- `PTBXL_ECGSex_09_DataEfficiency_Aggregate_and_Figures.py`
- `PTBXL_ECGSex_10_DataEfficiency_SeedEnsemble_PairedBootstrap.py`

## Interpretability

- `WCMC_ECGSex_VCGGeometry_ResNet_FinalInterpretability.py`
- `WCMC_ECGSex_VCGGeometry_ResNet_CorrectedIG.py`

## Cohort / comparator provenance

- `WCMC_ECGSex_01_Build_and_Audit_Cohort_FAST_v2.py`
- `WCMC_CurrentRun_Architecture_Baseline_Benchmark_PARALLEL_FIXED.py`
- `WCMC_ECGSex_06_Paired_Bootstrap_ThreeModel.py`

## Repository rule

Copy these scripts from Athena into the matching repository folders and then run the privacy audit before committing. Do not copy raw outputs, patient-level CSVs, model checkpoints, or logs containing identifiers.
