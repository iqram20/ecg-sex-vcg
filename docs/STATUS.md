# Repository status

## Already uploaded

- manuscript-facing README and study overview
- MIT license and citation metadata
- conda and pip environment specifications
- repository `.gitignore` and `.gitattributes`
- path configuration template
- reproducibility documentation
- data-governance/public-release safeguards
- exact Athena final-code manifest
- Athena code collector and one-command sync/push helper
- supplementary published-study comparison source data

## Final code that must be synchronized from Athena

The exact final training/evaluation/interpretability scripts listed in `ATHENA_FINAL_CODE_MANIFEST.md` remain on the Weill Cornell/Athena filesystem and should be copied directly from there rather than recreated from manuscript prose.

Run from a clone of this repository on Athena:

```bash
bash scripts/sync_athena_final_code_and_push.sh
```

The helper copies the exact final files, runs the repository safety audit, commits them, and pushes to `main`.

## Before public release

1. manually inspect every tracked file for patient-level or institutional information;
2. confirm that no protected WCMC/NYP data, MIMIC restricted data, credentials, or non-redistributable checkpoints are present;
3. replace hard-coded institutional paths with configurable paths where practical while retaining an archival copy of exact analysis scripts if needed;
4. update `CITATION.cff` with the manuscript DOI;
5. tag a release corresponding to the accepted manuscript.
