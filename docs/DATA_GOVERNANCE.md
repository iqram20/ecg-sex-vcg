# Data governance and repository safety

Do not commit any of the following:

- WCMC/NYP patient-level rows, identifiers, dates, accession numbers, or metadata
- raw institutional ECG waveforms
- PHI or re-identifiable clinical information
- authentication tokens, passwords, SSH keys, cloud credentials, or API secrets
- restricted MIMIC-IV data files
- large model checkpoints unless redistribution is explicitly permitted

Before making this repository public, run `scripts/audit_before_public_release.py` and manually review all tracked files.

The source-data CSVs committed here should contain only aggregate or literature-level information that is safe to distribute.
