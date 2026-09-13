# Repository status

## Current status

The validated ECG-sex reproducibility pipeline has been synchronized from the original
Weill Cornell/Athena analysis environment to this repository.

The repository now includes:

- WCMC/NYP cohort-construction code;
- MIMIC ResNet1D pretraining and WCMC transfer/fine-tuning code;
- final five-seed VCG-Geometry ResNet training and locked-test evaluation;
- frozen PTB-XL and Chapman/Ningbo external-validation code;
- PTB-XL data-efficiency analysis;
- final interpretability and corrected Integrated Gradients analyses;
- final manuscript figure and table generators;
- matching Slurm job files;
- validated pipeline lineage documentation; and
- SHA256 hashes for the canonical reproducibility source files.

Historical exploratory scripts, patient-level outputs, raw institutional ECG data,
model checkpoints, and internal audit reports are intentionally excluded.

## Validation completed

Before synchronization, the pipeline underwent source/result provenance checks,
training-lineage checks, Python syntax validation, Slurm shell validation, Git diff
validation, and a repository safety scan.

The canonical lineage is documented in `PIPELINE_LINEAGE.md`.
Exact source hashes are recorded in `FINAL_PIPELINE_SHA256.csv`.

## Before public release

1. manually inspect every tracked file for patient-level or institutional information;
2. confirm that no protected WCMC/NYP data, restricted MIMIC data, credentials,
   patient-level predictions, or non-redistributable checkpoints are present;
3. retain the environment-specific-path notice in `PIPELINE_LINEAGE.md`;
4. update `CITATION.cff` with the final publication DOI;
5. tag a release corresponding to the accepted manuscript.
