# Postoperative clinical-validity analysis

This document summarizes the manuscript-aligned postoperative outcome analysis for ECG-sex discordance.

## Cohort design

- patient universe: WCMC/NYP locked-test patients only
- index operation: frozen from the 90-day preoperative construction
- ECG selection: temporally closest preceding ECG
- primary analysis: same index operation with ECG within 30 days
- sensitivity analysis: 90-day preoperative window
- 30-day primary cohort: 2,513 patients; mortality data N=2,506; 65 deaths
- 90-day sensitivity cohort: 3,118 patients; mortality data N=3,111; 67 deaths

## Exposure

ECG-sex discordance is derived from the frozen five-seed ensemble probability of male sex. The sex-concordant margin is the logit-transformed male probability for males and its negative for females; discordance is the negative of that margin, so larger values represent a more opposite-sex ECG phenotype. Discordance is standardized within biological sex using the sex-specific distribution of the original locked WCMC test cohort.

## Outcomes and models

Outcomes:
- 30-day postoperative mortality
- postoperative acute kidney injury
- postoperative ICU admission

Primary model:
- logistic regression with HC3 robust standard errors
- adjusted for age, biological sex, and ASA class
- effect reported per 1-SD greater ECG-sex discordance

Sensitivity analyses:
- 90-day preoperative ECG window
- sex-by-discordance interaction
- sex-stratified models
- Firth bias-reduced logistic regression for mortality
- 2,000 patient-level bootstrap replicates for incremental AUROC, AUPRC, and Brier score

## Main results

| Window | Outcome | N | Events | Adjusted OR per 1-SD discordance |
|---|---|---:|---:|---:|
| 30 days | 30-day mortality | 2,506 | 65 | 1.84 (95% CI 1.37–2.46) |
| 30 days | postoperative AKI | 1,660 | 107 | 1.08 (0.88–1.33) |
| 30 days | postoperative ICU admission | 2,493 | 545 | 0.98 (0.87–1.10) |
| 90 days | 30-day mortality | 3,111 | 67 | 1.85 (1.39–2.45) |
| 90 days | postoperative AKI | 1,793 | 114 | 1.08 (0.88–1.33) |
| 90 days | postoperative ICU admission | 3,098 | 559 | 0.98 (0.87–1.09) |

Firth mortality estimates were 1.82 (95% CI 1.34–2.48) for the 30-day cohort and 1.84 (1.35–2.49) for the 90-day cohort.

Adding discordance to the age/sex/ASA clinical base model increased AUROC by 0.030 (95% bootstrap CI 0.007–0.061) in the 30-day cohort and 0.027 (0.006–0.056) in the 90-day cohort. AUPRC improvements were directionally positive but confidence intervals crossed zero.

## Repository provenance

The included `scripts/evaluation/WCMC_ECGSex_Postop_AllPreopECG_Frozen5Seed.py` reproduces frozen five-seed inference and preoperative ECG/surgery selection for held-out WCMC patients. The final fixed-index post-processing source `WCMC_ECGSex_Postop_FixedIndex_Robustness.py` was not available in the uploaded source bundle used for this synchronization and should be copied from the canonical Athena project before public release.
