# Repository status

## Current status

The validated ECG-sex reproducibility pipeline has been synchronized from the original Weill Cornell/Athena analysis environment to this repository.

The repository now includes:

- WCMC/NYP cohort-construction code;
- MIMIC ResNet1D pretraining and WCMC transfer/fine-tuning code;
- final five-seed VCG-Geometry ResNet training and locked-test evaluation;
- frozen PTB-XL and Chapman/Ningbo external-validation code;
- PTB-XL data-efficiency analysis;
- final interpretability and corrected Integrated Gradients analyses;
- BMI/body-habitus and grouped precordial/limb-lead sensitivity analyses;
- frozen all-preoperative-ECG inference code supporting the postoperative clinical-validity extension;
- manuscript-aligned postoperative outcome documentation;
- final manuscript figure and table generators;
- matching Slurm job files; and
- validated pipeline lineage documentation.

Historical exploratory scripts, patient-level outputs, raw institutional ECG data, model checkpoints, and internal audit reports are intentionally excluded.

## Validation completed

Before synchronization, the core model pipeline underwent source/result provenance checks, training-lineage checks, Python syntax validation, Slurm shell validation, Git diff validation, and a repository safety scan.

The canonical lineage is documented in `PIPELINE_LINEAGE.md`. The postoperative fixed-index analysis is summarized separately in `CLINICAL_VALIDATION.md`.

## Before public release

1. manually inspect every tracked file for patient-level or institutional information;
2. confirm that no protected WCMC/NYP data, restricted MIMIC data, credentials, patient-level predictions, or non-redistributable checkpoints are present;
3. add the exact final source file `WCMC_ECGSex_Postop_FixedIndex_Robustness.py` and its matching Slurm file once the canonical source copy is available;
4. refresh `FINAL_PIPELINE_SHA256.csv` after the new sensitivity/clinical-validity scripts are frozen;
5. update `CITATION.cff` with the final publication DOI;
6. tag a release corresponding to the accepted manuscript.
