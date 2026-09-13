# Final ECG-sex pipeline lineage

The final manuscript pipeline was reconstructed from the Athena project using result,
checkpoint, source-code, and SLURM provenance audits.

## Canonical lineage

1. WCMC/NYP cohort construction
   - `WCMC_ECGSex_01_Build_and_Audit_Cohort_FAST_v2.py`

2. ResNet1D backbone: MIMIC pretraining followed by WCMC transfer/fine-tuning
   - `WCMC_ECGSex_ResNet1D_MIMIC_to_WCMC_Transfer.py`
   - output stage: `14_resnet1d_mimic_to_wcmc`
   - MIMIC checkpoint: `best_ResNet1D_MIMICPretrain_ECGSex.pt`

3. Final geometry-aware model
   - `WCMC_ECGSex_VCGGeometry_ResNet_5Seed_LockedTest.py`
   - seeds 11, 22, 33, 42, 55
   - initializes its waveform pathway from the Stage-14 MIMIC-pretrained ResNet weights
   - checkpoint selection is based on WCMC validation AUPRC
   - operating threshold is determined from WCMC validation predictions
   - locked WCMC test is evaluated only after checkpoint and threshold are frozen

4. Frozen external validation
   - `WCMC_ECGSex_VCGGeometry_ResNet_ExternalConfirmation.py`
   - PTB-XL and Chapman/Ningbo use frozen model parameters without external retraining,
     recalibration, or threshold selection for the primary external evaluation.

5. PTB-XL data efficiency
   - `PTBXL_ECGSex_VCGGeometry_DataEfficiency.py`

6. Interpretability
   - final lead/representation ablation
   - corrected decomposed Integrated Gradients
   - sex-specific corrected-IG attribution maps

7. Manuscript figures/tables
   - only the canonical final versions are included in this repository.

## Provenance note

The ResNet1D MIMIC→WCMC source file is stored under an Athena archive directory because
the working project code tree was reorganized after the experiments. It is included here
as a canonical reproducibility file because the provenance audit verified that it names the
same Stage-14 run and exact MIMIC checkpoint consumed by the final VCG pipeline.

Historical exploratory architectures and superseded plotting/table scripts are intentionally
excluded from the publication repository.

## Environment-specific paths

The source files in this repository are preserved from the analyses used for the
manuscript. Some scripts therefore contain absolute paths from the original
Weill Cornell/Athena computing environment. These paths identify local data,
run, and log locations and are not required to match on another system.

Users reproducing the analyses should update the corresponding input, output,
checkpoint, and dataset paths for their own computing environment. No raw
WCMC/NYP clinical data, patient-level predictions, or model checkpoints are
included in this repository.
