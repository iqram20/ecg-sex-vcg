# ECG Sex Classification with Geometry-Aware Deep Learning

Code and reproducibility materials for the manuscript **“Geometry-Aware Interpretable Deep Learning Reveals Sex-Specific Cardiac Electrophysiological Signatures from 12-Lead Electrocardiograms.”**

## Overview

This project studies biological-sex information encoded in routine 12-lead ECGs using a geometry-aware deep-learning framework. The final **VCG-Geometry ResNet** combines a 12-lead waveform ResNet1D with an explicit vectorcardiographic pathway derived from reconstructed XYZ trajectories and 13 geometry channels.

The study uses four adult ECG cohorts:

- **MIMIC-IV-ECG** — waveform pretraining
- **WCMC/NYP** — fine-tuning, validation, and locked internal testing
- **PTB-XL** — zero-shot external validation and a separate data-efficiency experiment
- **Chapman/Ningbo** — zero-shot external validation

## Final study design

- MIMIC-IV pretraining cohort: **155,626** participants
- WCMC/NYP downstream cohort: **106,868** participants
- WCMC/NYP locked test: **21,374** participants
- PTB-XL external cohort: **18,752** participants
- Chapman/Ningbo external cohort: **42,997** participants
- Final model: five-seed VCG-Geometry ResNet ensemble
- Final seeds: **11, 22, 33, 42, 55**
- Primary uncertainty: **2,000 patient-level paired bootstrap replicates**
- Prespecified age strata used throughout: **18–39, 40–49, 50–59, 60–69, 70–79, ≥80 years**

## Main reported performance

| Cohort | Model | AUROC | AUPRC | Accuracy | F1 | Brier |
|---|---|---:|---:|---:|---:|---:|
| WCMC/NYP locked test | VCG-Geometry ResNet | 0.954 | 0.951 | 0.881 | 0.879 | 0.085 |
| WCMC/NYP locked test | transferred ResNet1D | 0.950 | 0.947 | 0.875 | 0.873 | 0.089 |
| PTB-XL zero-shot | VCG-Geometry ResNet | 0.927 | 0.929 | 0.724 | 0.785 | 0.196 |
| Chapman/Ningbo zero-shot | VCG-Geometry ResNet | 0.916 | 0.935 | 0.803 | 0.844 | 0.135 |

The VCG model improved WCMC/NYP AUROC by **+0.0038 (95% CI 0.0030–0.0045)** relative to the matched transferred ResNet1D. In the PTB-XL data-efficiency analysis, transfer using 10% of local training data achieved AUROC **0.928**, compared with **0.912** for local training using 100% of the available training data.

## Model representation

The waveform branch uses 12 standardized ECG leads plus their first temporal differences (24 channels). The VCG branch reconstructs XYZ trajectories before per-lead standardization and derives 13 geometry channels comprising:

- X, Y, Z coordinates
- first temporal derivatives
- vector magnitude
- speed
- unit direction components
- radial velocity
- bounded trajectory turning

The two representations are fused using a gated residual correction to the waveform-derived biological-sex logit.

## Interpretability

The manuscript uses complementary analyses rather than a single saliency method:

- raw-signal single-lead ablation
- VCG component ablation
- corrected decomposed Integrated Gradients in representation space
- sex-specific signed attribution
- conventional ECG interval comparisons
- age-stratified performance and age-stratified lead ablation

Corrected Integrated Gradients uses separate zero-reference representation-space paths for waveform and VCG branches, 48-node Gauss–Legendre quadrature on [0,1], and the male logit as the attribution target.

## Repository layout

```text
configs/                     local path templates
docs/                        reproducibility and governance notes
results/source_data/         non-restricted source-data tables
scripts/cohort/              WCMC/NYP cohort construction
scripts/pretraining_transfer/ MIMIC pretraining and WCMC transfer
scripts/training/            final five-seed VCG-Geometry ResNet training
scripts/evaluation/          locked-test, external-validation, and data-efficiency analyses
scripts/interpretability/    final interpretability analyses
scripts/figures/             final manuscript figure generators
scripts/tables/              final manuscript table generators
slurm/final_pipeline/        matching cluster job files
```

## Important data-governance note

This repository must **not** contain WCMC/NYP patient-level data, protected health information, private metadata, raw institutional ECGs, access credentials, or redistributability-restricted model checkpoints. The repository is intentionally private while the reproducibility package is being assembled.

Public datasets must be obtained from their original sources and used under their respective terms:

- MIMIC-IV-ECG: PhysioNet credentialed access
- PTB-XL: PhysioNet
- Chapman/Ningbo: PhysioNet

## Reproducing the final analysis

1. Create the environment from `environment.yml`.
2. Copy `configs/paths.example.yaml` to `configs/paths.yaml`.
3. Set local dataset/run paths.
4. Follow the validated pipeline lineage in `docs/PIPELINE_LINEAGE.md` and verify exact source hashes using `docs/FINAL_PIPELINE_SHA256.csv`.
5. Run final manuscript figure/table scripts.

The final Athena scripts are deliberately collected by filename rather than recreating them from the manuscript. This avoids silently changing the model implementation.

## Citation

Citation metadata is provided in `CITATION.cff` and will be updated with the journal DOI after publication.

## Authors

Iqram Hussain, Joseph R. Scarpa, Antonis A. Armoundas, and Richard Boyer.
