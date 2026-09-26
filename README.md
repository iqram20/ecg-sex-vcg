# ECG Sex Classification with Geometry-Aware Deep Learning

Code and reproducibility materials for the manuscript **“Geometry-Aware Interpretable Deep Learning Reveals Sex-Associated ECG Signatures and Their Association with Postoperative Outcomes.”**

## Overview

This project studies biological-sex information encoded in routine 12-lead ECGs using a geometry-aware deep-learning framework. The final **VCG-Geometry ResNet** combines a 12-lead waveform ResNet1D with an explicit vectorcardiographic pathway derived from reconstructed orthogonal VCG trajectories and 13 geometry channels.

The study uses four adult ECG cohorts:

- **MIMIC-IV-ECG** — waveform pretraining
- **WCMC/NYP** — fine-tuning, validation, locked internal testing, body-habitus sensitivity analyses, and postoperative clinical-validity analysis
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
- Prespecified age strata: **18–39, 40–49, 50–59, 60–69, 70–79, ≥80 years**

## Main reported performance

| Cohort | Model | AUROC | AUPRC | Accuracy | F1 | Brier |
|---|---|---:|---:|---:|---:|---:|
| WCMC/NYP locked test | VCG-Geometry ResNet | 0.954 | 0.951 | 0.881 | 0.879 | 0.085 |
| WCMC/NYP locked test | transferred ResNet1D | 0.950 | 0.947 | 0.875 | 0.873 | 0.089 |
| PTB-XL zero-shot | VCG-Geometry ResNet | 0.927 | 0.929 | 0.724 | 0.785 | 0.196 |
| Chapman/Ningbo zero-shot | VCG-Geometry ResNet | 0.916 | 0.935 | 0.803 | 0.844 | 0.135 |

The VCG model improved WCMC/NYP AUROC by **+0.0038 (95% CI 0.0030–0.0045)** relative to the matched transferred ResNet1D. In the PTB-XL data-efficiency analysis, transfer using 10% of local training data achieved AUROC **0.928**, compared with **0.912** for local training using 100% of the available training data.

## Body-habitus and precordial-lead sensitivity

BMI was available for **20,133** locked-test participants. Discrimination remained stable across BMI strata (AUROC **0.951**, **0.962**, and **0.949** for BMI <25, 25–29.9, and ≥30 kg/m²). In the age- and BMI-matched sensitivity cohort (**7,906 female–male pairs**), AUROC was **0.955**. There was no evidence of a sex-by-BMI interaction in the age-adjusted model score (**P=0.737**).

Grouped ablation showed that the sex-associated signal was not confined to precordial leads: removing V3–V6 reduced AUROC by **0.113**, removing all V1–V6 by **0.244**, and removing all limb leads by **0.062**.

## Postoperative clinical-validity analysis

Postoperative outcomes were evaluated only in patients from the locked WCMC test set. A single index operation was frozen from the 90-day preoperative construction, and the 30-day primary cohort was defined as the subset of the same index operations with a qualifying ECG within 30 days.

- 30-day primary cohort: **2,513** patients; mortality data available for **2,506**; **65 deaths (2.6%)**
- 90-day sensitivity cohort: **3,118** patients; mortality data available for **3,111**; **67 deaths (2.2%)**
- Adjusted 30-day mortality association: **OR 1.84 per 1-SD greater ECG-sex discordance (95% CI 1.37–2.46; P<0.001)**
- 90-day sensitivity association: **OR 1.85 (95% CI 1.39–2.45; P<0.001)**
- Firth sensitivity analysis produced nearly identical mortality estimates
- No association was observed with postoperative AKI or ICU admission
- Adding discordance to age, sex, and ASA increased AUROC by **0.030 (95% CI 0.007–0.061)** in the 30-day cohort

See `docs/CLINICAL_VALIDATION.md` for the manuscript-aligned analysis summary.

## Model representation

The waveform branch uses 12 standardized ECG leads plus their first temporal differences (24 channels). The VCG branch reconstructs orthogonal spatial trajectories before per-lead standardization and derives 13 geometry channels comprising:

- three orthogonal VCG coordinates
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
- grouped precordial/limb-lead ablation
- BMI-stratified performance and age/BMI matching
- VCG component ablation
- corrected decomposed Integrated Gradients in representation space
- sex-specific signed attribution
- conventional ECG interval comparisons
- age-stratified performance and age-stratified lead ablation
- exploratory ECG-sex discordance association with postoperative outcomes

Corrected Integrated Gradients uses separate zero-reference representation-space paths for waveform and VCG branches, 48-node Gauss–Legendre quadrature on [0,1], and the male logit as the attribution target.

## Repository layout

```text
configs/                      local path templates
docs/                         reproducibility, governance, and clinical-validation notes
results/source_data/          non-restricted source-data tables
scripts/cohort/               WCMC/NYP cohort construction
scripts/pretraining_transfer/ MIMIC pretraining and WCMC transfer
scripts/training/             final five-seed VCG-Geometry ResNet training
scripts/evaluation/           locked-test, external-validation, data-efficiency, and postoperative analyses
scripts/interpretability/     final interpretability and body-habitus sensitivity analyses
scripts/figures/              final manuscript figure generators
scripts/tables/               final manuscript table generators
slurm/final_pipeline/         matching cluster job files
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
4. Follow the validated pipeline lineage in `docs/PIPELINE_LINEAGE.md`.
5. Run final manuscript figure/table scripts and the sensitivity/clinical-validity analyses as applicable.

The final Athena scripts are deliberately collected by filename rather than recreated from manuscript prose.

## Citation

Citation metadata is provided in `CITATION.cff` and will be updated with the journal DOI after publication.

## Authors

Iqram Hussain, Joseph R. Scarpa, and Richard Boyer.
