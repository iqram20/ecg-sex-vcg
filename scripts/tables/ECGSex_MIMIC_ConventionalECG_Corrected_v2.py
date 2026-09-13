#!/usr/bin/env python
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

MASTER_CSV = Path("/athena/madelab/scratch/iqh4001/ECG/ECG_mimic_IV/Data/Output/master_ecg_full_dataset_20260228_155252.csv")
MACHINE_CSV = Path("/athena/madelab/scratch/iqh4001/ECG/ECG_mimic_IV/Data/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/machine_measurements.csv")

RUN = Path("/athena/madelab/scratch/iqh4001/ECG/ECG_Echo/Runs/Master_ECG_Sex_MIMIC_EchoNext/20260609_020154_MASTER_ECG_SEX_MIMIC_ECHONEXT")
OUT_DIR = RUN / "publication_tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_MANUSCRIPT = OUT_DIR / "Table_ECG_Sex_Differences_AgeMatched.csv"
OUT_SOURCE = OUT_DIR / "Table_ECG_Sex_Differences_AgeMatched_Source.csv"
OUT_AUDIT = OUT_DIR / "Table_ECG_Sex_Differences_AgeMatched_Audit.csv"

RANDOM_SEED = 42
MAX_PAIRS_PER_VARIABLE = 20000

def clean_numeric(s):
    x = pd.to_numeric(s, errors="coerce")
    x = x.mask(x >= 29999, np.nan)
    x = x.mask(x <= 0, np.nan)
    return x

def exact_age_match(df, value_col, seed):
    x = df.dropna(subset=["gender", "age", value_col]).copy()
    x = x[x["gender"].isin(["F", "M"]) & (x["age"] >= 18)].copy()
    x["age_int"] = np.floor(x["age"]).astype(int)

    rng = np.random.default_rng(seed)
    parts = []
    pair_id = 0

    for age in sorted(x["age_int"].unique()):
        f = x[(x["age_int"] == age) & (x["gender"] == "F")]
        m = x[(x["age_int"] == age) & (x["gender"] == "M")]
        n = min(len(f), len(m))
        if n == 0:
            continue

        f_idx = rng.permutation(f.index.to_numpy())[:n]
        m_idx = rng.permutation(m.index.to_numpy())[:n]

        ff = x.loc[f_idx].copy().reset_index(drop=True)
        mm = x.loc[m_idx].copy().reset_index(drop=True)
        ids = np.arange(pair_id, pair_id + n)
        pair_id += n
        ff["pair_id"] = ids
        mm["pair_id"] = ids
        parts += [ff, mm]

    if not parts:
        raise RuntimeError(f"No matched pairs for {value_col}")

    matched = pd.concat(parts, ignore_index=True)
    pair_ids = matched["pair_id"].drop_duplicates().to_numpy()

    if len(pair_ids) > MAX_PAIRS_PER_VARIABLE:
        keep = rng.choice(pair_ids, size=MAX_PAIRS_PER_VARIABLE, replace=False)
        matched = matched[matched["pair_id"].isin(keep)].copy()

    counts = matched.groupby(["age_int", "gender"]).size().unstack(fill_value=0)
    if not (counts.get("F", 0) == counts.get("M", 0)).all():
        raise RuntimeError(f"Age matching failed for {value_col}")

    return matched

def ptxt(p):
    return "P<0.001" if p < 0.001 else f"P={p:.3f}"

print("=" * 100)
print("MIMIC-IV CORRECTED CONVENTIONAL ECG ANALYSIS")
print("=" * 100)

master = pd.read_csv(
    MASTER_CSV,
    usecols=["subject_id", "study_id", "gender", "group", "age"],
    low_memory=False,
)

mm = pd.read_csv(
    MACHINE_CSV,
    usecols=["subject_id", "study_id", "rr_interval", "p_onset", "p_end",
             "qrs_onset", "qrs_end", "t_end"],
    low_memory=False,
)

df = master.merge(mm, on=["subject_id", "study_id"], how="inner")
df = df[(df["group"] == "Control") & df["gender"].isin(["F", "M"])].copy()

for c in ["rr_interval", "p_onset", "p_end", "qrs_onset", "qrs_end", "t_end"]:
    df[c] = clean_numeric(df[c])

df["age"] = pd.to_numeric(df["age"], errors="coerce")

df["RR_ms"] = df["rr_interval"]
df["HR_bpm"] = 60000.0 / df["RR_ms"]

# CORRECT PR INTERVAL: P onset -> QRS onset
df["PR_ms"] = df["qrs_onset"] - df["p_onset"]

# Audit only: this is P-wave duration, NOT PR interval
df["P_duration_ms"] = df["p_end"] - df["p_onset"]

df["QRS_ms"] = df["qrs_end"] - df["qrs_onset"]
df["QT_ms"] = df["t_end"] - df["qrs_onset"]

rr_sec = df["RR_ms"] / 1000.0
df["QTc_Bazett_ms"] = df["QT_ms"] / np.sqrt(rr_sec)
df["QTc_Fridericia_ms"] = df["QT_ms"] / np.cbrt(rr_sec)

specs = [
    ("HR_bpm", "Heart Rate (bpm)",
     df["RR_ms"].between(300, 2000) & df["HR_bpm"].between(30, 200)),
    ("PR_ms", "PR Interval (ms)",
     df["PR_ms"].between(60, 300)),
    ("QRS_ms", "QRS Duration (ms)",
     df["QRS_ms"].between(50, 250)),
    ("QT_ms", "QT Interval (ms)",
     df["QT_ms"].between(200, 700)),
    ("QTc_Bazett_ms", "QTc Bazett (ms)",
     df["RR_ms"].between(300, 2000) &
     df["QT_ms"].between(200, 700) &
     df["QTc_Bazett_ms"].between(250, 700)),
    ("QTc_Fridericia_ms", "QTc Fridericia (ms)",
     df["RR_ms"].between(300, 2000) &
     df["QT_ms"].between(200, 700) &
     df["QTc_Fridericia_ms"].between(250, 700)),
]

rows, source_rows, audit_rows = [], [], []

for i, (key, label, mask) in enumerate(specs):
    eligible = df[mask.fillna(False)].dropna(subset=[key, "age", "gender"]).copy()
    eligible = eligible[eligible["age"] >= 18].copy()

    ef = int((eligible["gender"] == "F").sum())
    em = int((eligible["gender"] == "M").sum())

    matched = exact_age_match(eligible, key, RANDOM_SEED + i)

    f = matched.loc[matched["gender"] == "F", key].dropna()
    m = matched.loc[matched["gender"] == "M", key].dropna()

    if len(f) != len(m):
        raise RuntimeError(f"{label}: unmatched N after exact matching")

    p = float(ttest_ind(f, m, equal_var=False, nan_policy="omit").pvalue)

    fmean, fsd = float(f.mean()), float(f.std(ddof=1))
    mmean, msd = float(m.mean()), float(m.std(ddof=1))
    delta = fmean - mmean

    fa = matched.loc[matched["gender"] == "F", "age"]
    ma = matched.loc[matched["gender"] == "M", "age"]

    rows.append({
        "Variable": label,
        "Female": f"{fmean:.1f} ± {fsd:.1f}",
        "Male": f"{mmean:.1f} ± {msd:.1f}",
        "Delta_F_minus_M": round(delta, 1),
        "P_value": ptxt(p),
    })

    source_rows.append({
        "Variable": label,
        "Female_N": len(f),
        "Male_N": len(m),
        "Female_mean": fmean,
        "Female_SD": fsd,
        "Male_mean": mmean,
        "Male_SD": msd,
        "Delta_F_minus_M": delta,
        "P_value_exact": p,
    })

    audit_rows.append({
        "Variable": label,
        "Eligible_Female_N_before_matching": ef,
        "Eligible_Male_N_before_matching": em,
        "Matched_Female_N": len(f),
        "Matched_Male_N": len(m),
        "Female_age_mean": float(fa.mean()),
        "Male_age_mean": float(ma.mean()),
        "Female_age_SD": float(fa.std(ddof=1)),
        "Male_age_SD": float(ma.std(ddof=1)),
        "N_difference_after_matching": len(f) - len(m),
    })

    print(
        f"{label:24s} | eligible F/M={ef:,}/{em:,} | "
        f"matched={len(f):,}/{len(m):,} | "
        f"{fmean:.1f} ± {fsd:.1f} vs {mmean:.1f} ± {msd:.1f} | "
        f"Delta={delta:+.1f} | {ptxt(p)}"
    )

results = pd.DataFrame(rows)
source = pd.DataFrame(source_rows)
audit = pd.DataFrame(audit_rows)

print("\n" + "=" * 100)
print("CORRECTED MIMIC-IV TABLE")
print("=" * 100)
print(results.to_string(index=False))

print("\n" + "=" * 100)
print("PR SANITY CHECK")
print("=" * 100)
print("PR interval = qrs_onset - p_onset")
print("P-wave duration = p_end - p_onset (old mislabeled quantity)")

results.to_csv(OUT_MANUSCRIPT, index=False, encoding="utf-8-sig")
source.to_csv(OUT_SOURCE, index=False, encoding="utf-8-sig")
audit.to_csv(OUT_AUDIT, index=False, encoding="utf-8-sig")

print("\nSaved:")
print(OUT_MANUSCRIPT)
print(OUT_SOURCE)
print(OUT_AUDIT)
print("\nDONE")
