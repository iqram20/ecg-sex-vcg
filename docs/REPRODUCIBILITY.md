# Reproducibility

## Analysis principles

- one ECG per participant in the final adult classification cohorts
- no participant overlap across WCMC/NYP train, validation, and locked-test partitions
- thresholds selected only on validation data
- WCMC/NYP locked test evaluated after model/threshold selection
- PTB-XL and Chapman/Ningbo primary external evaluations are zero-shot with no retraining, recalibration, or threshold adjustment
- patient-level paired bootstrap is used for matched model comparisons
- 95% bootstrap confidence intervals are the primary uncertainty measure
- postoperative clinical-validity analyses use only held-out WCMC locked-test patients
- the postoperative primary cohort uses the same index operation frozen from the 90-day construction

## Final WCMC/NYP operating thresholds

- VCG-Geometry ResNet ensemble: 0.484619
- matched transferred ResNet1D ensemble: 0.453711

## Final ensemble seeds

11, 22, 33, 42, 55

## Age strata

18–39, 40–49, 50–59, 60–69, 70–79, ≥80 years.

## Body-habitus sensitivity

The frozen final VCG ensemble is evaluated across BMI strata (<25, 25–29.9, and ≥30 kg/m²), in an age/BMI-matched female–male cohort, and using an age-adjusted sex-by-BMI interaction model. Grouped raw-signal ablation removes V3–V6, V1–V2, all precordial leads, or all limb leads, with 2,000 paired patient-level bootstrap replicates.

## Corrected Integrated Gradients

The original raw VCG interpolation path was ill-conditioned near alpha=0 because RMS normalization removes most interpolation scale. The final analysis therefore uses decomposed representation-space paths:

1. zero 24-channel waveform representation → observed waveform representation while observed VCG geometry is fixed;
2. zero 13-channel VCG geometry representation → observed VCG geometry while observed waveform representation is fixed.

The attribution target is the male logit and numerical integration uses 48-node Gauss–Legendre quadrature.

## Postoperative clinical-validity analysis

ECG-sex discordance is defined from the frozen five-seed male probability as the negative sex-concordant logit margin and standardized within biological sex using the original locked-test distribution. The primary regression adjusts for age, biological sex, and ASA class with HC3 robust standard errors. Sensitivity analyses include the 90-day ECG window, sex interaction/stratification, and Firth bias-reduced logistic regression. Incremental AUROC, AUPRC, and Brier differences are estimated with 2,000 patient-level bootstrap replicates.

## Exact final code

See `PIPELINE_LINEAGE.md` for the validated analysis lineage. The new BMI, grouped-lead, and frozen postoperative-inference scripts are included in this repository. The exact final fixed-index robustness post-processing source should be added from the canonical Athena copy before public release rather than reconstructed from manuscript prose.
