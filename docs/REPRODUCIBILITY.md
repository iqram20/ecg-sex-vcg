# Reproducibility

## Analysis principles

- one ECG per participant in the final adult cohorts
- no participant overlap across WCMC/NYP train, validation, and locked-test partitions
- thresholds selected only on validation data
- WCMC/NYP locked test evaluated after model/threshold selection
- PTB-XL and Chapman/Ningbo primary external evaluations are zero-shot with no retraining, recalibration, or threshold adjustment
- patient-level paired bootstrap is used for matched model comparisons
- 95% bootstrap confidence intervals are the primary uncertainty measure

## Final WCMC/NYP operating thresholds

- VCG-Geometry ResNet ensemble: 0.484619
- matched transferred ResNet1D ensemble: 0.453711

## Final ensemble seeds

11, 22, 33, 42, 55

## Age strata

18–39, 40–49, 50–59, 60–69, 70–79, ≥80 years.

## Corrected Integrated Gradients

The original raw VCG interpolation path was ill-conditioned near alpha=0 because RMS normalization removes most interpolation scale. The final analysis therefore uses decomposed representation-space paths:

1. zero 24-channel waveform representation → observed waveform representation while observed VCG geometry is fixed;
2. zero 13-channel VCG geometry representation → observed VCG geometry while observed waveform representation is fixed.

The attribution target is the male logit and numerical integration uses 48-node Gauss–Legendre quadrature.

## Exact final code

See `PIPELINE_LINEAGE.md` for the validated analysis lineage and `FINAL_PIPELINE_SHA256.csv` for SHA256 hashes of the exact canonical source files. The manuscript implementation should be reproduced from those source files rather than reconstructed from the prose description.
