#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_BOOT = 2000
BOOT_SEED = 20260930

AGE_BINS = [18, 40, 50, 60, 70, 80, np.inf]
AGE_LABELS = ["18-39", "40-49", "50-59", "60-69", "70-79", ">=80"]

RUN_DIR = Path(
    "/athena/madelab/scratch/iqh4001/ECG/ECG_Echo/Runs/"
    "Master_ECG_Sex_MIMIC_EchoNext/"
    "20260609_020154_MASTER_ECG_SEX_MIMIC_ECHONEXT"
)
WCMC_ROOT = RUN_DIR / "wcmc_ecgsex_from_scratch"

TEST_CSV = (
    WCMC_ROOT / "04_wcmc_finetune_from_mimic/splits/WCMC_ECGSex_test.csv"
)
BASELINE_PRED = (
    WCMC_ROOT
    / "19_vcg_geometry_resnet_5seed_locked_wcmc"
    / "final_5seed_ensemble_bootstrap"
    / "WCMC_LockedTest_Aligned_5SeedEnsemble_Predictions.csv"
)

GROUPED_PRED_DIR = Path(
    "/athena/madelab/scratch/iqh4001/ECG/ECG-gender/Results/"
    "BreastPlacement_GroupedLeadSensitivity_WCMC/predictions"
)

PRECORDIAL_PRED = GROUPED_PRED_DIR / "remove_all_precordial.csv"
LIMB_PRED = GROUPED_PRED_DIR / "remove_all_limb.csv"

MASTER_PARQUET = Path(
    "/athena/madelab/scratch/iqh4001/Azure_Cradle_ECG/Data/"
    "ECG-Echo_MetaData/ecg_echo_master_cohort_v1.parquet"
)

OUT_DIR = Path(
    "/athena/madelab/scratch/iqh4001/ECG/ECG-gender/Results/"
    "WCMC_AgeStratified_RegionalLeadAblation"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file:\n{path}")
    return path


def metric(y, p, which):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if np.unique(y).size < 2:
        return np.nan
    if which == "AUROC":
        return float(roc_auc_score(y, p))
    if which == "AUPRC":
        return float(average_precision_score(y, p))
    raise ValueError(which)


def p_two_sided(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    p = 2.0 * min(np.mean(x <= 0), np.mean(x >= 0))
    return max(float(p), 1.0 / (len(x) + 1.0))


def get_age(test):
    age_candidates = ["age_at_ecg", "ecg_age_at_ecg", "age", "patient_age"]
    age_col = next((c for c in age_candidates if c in test.columns), None)

    if age_col is not None:
        out = test[["patient_key", "ecg_key", "sex_binary", age_col]].copy()
        out = out.rename(columns={age_col: "age"})
        out["age"] = pd.to_numeric(out["age"], errors="coerce")
        return out

    master = pd.read_parquet(
        require(MASTER_PARQUET),
        columns=["ecg_key", "ecg_age_at_ecg"],
    )
    master["ecg_key"] = master["ecg_key"].astype(str)
    master["ecg_age_at_ecg"] = pd.to_numeric(
        master["ecg_age_at_ecg"], errors="coerce"
    )

    nuniq = master.groupby("ecg_key")["ecg_age_at_ecg"].nunique(dropna=True)
    if (nuniq > 1).any():
        raise RuntimeError("Conflicting ages found for duplicated ecg_key.")

    master = master.drop_duplicates("ecg_key")

    return (
        test[["patient_key", "ecg_key", "sex_binary"]]
        .copy()
        .merge(
            master.rename(columns={"ecg_age_at_ecg": "age"}),
            on="ecg_key",
            how="left",
            validate="one_to_one",
        )
    )


def load_ablation(path, suffix):
    df = pd.read_csv(require(path), low_memory=False)
    needed = {"patient_key", "ecg_key", "sex_true_binary", "prob_male"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"{path.name} missing: {sorted(missing)}")

    df["patient_key"] = df["patient_key"].astype(str)
    df["ecg_key"] = df["ecg_key"].astype(str)

    return df[
        ["patient_key", "ecg_key", "sex_true_binary", "prob_male"]
    ].rename(
        columns={
            "sex_true_binary": f"sex_{suffix}",
            "prob_male": f"prob_{suffix}",
        }
    )


test = pd.read_csv(require(TEST_CSV), low_memory=False)
for c in ["patient_key", "ecg_key"]:
    test[c] = test[c].astype(str)

clinical = get_age(test)
for c in ["patient_key", "ecg_key"]:
    clinical[c] = clinical[c].astype(str)

baseline = pd.read_csv(require(BASELINE_PRED), low_memory=False)
needed_base = {"patient_key", "ecg_key", "sex_true_binary", "prob_vcg"}
missing = needed_base - set(baseline.columns)
if missing:
    raise RuntimeError(f"Baseline file missing: {sorted(missing)}")

for c in ["patient_key", "ecg_key"]:
    baseline[c] = baseline[c].astype(str)

baseline = baseline[
    ["patient_key", "ecg_key", "sex_true_binary", "prob_vcg"]
].rename(columns={"prob_vcg": "prob_intact"})

pre = load_ablation(PRECORDIAL_PRED, "precordial")
limb = load_ablation(LIMB_PRED, "limb")

merged = (
    clinical
    .merge(baseline, on=["patient_key", "ecg_key"], how="inner", validate="one_to_one")
    .merge(pre, on=["patient_key", "ecg_key"], how="inner", validate="one_to_one")
    .merge(limb, on=["patient_key", "ecg_key"], how="inner", validate="one_to_one")
)

truth = merged["sex_binary"].to_numpy(dtype=int)
for col in ["sex_true_binary", "sex_precordial", "sex_limb"]:
    if not np.array_equal(truth, merged[col].to_numpy(dtype=int)):
        raise RuntimeError(f"Label mismatch in {col}")

if len(merged) != len(test):
    raise RuntimeError(
        f"Aligned N={len(merged):,}; locked-test N={len(test):,}. "
        "Grouped prediction files are incomplete or misaligned."
    )

merged["age"] = pd.to_numeric(merged["age"], errors="coerce")
if merged["age"].isna().any():
    raise RuntimeError(
        f"Missing age for {int(merged['age'].isna().sum()):,} locked-test rows."
    )

merged["age_group"] = pd.cut(
    merged["age"],
    bins=AGE_BINS,
    labels=AGE_LABELS,
    right=False,
    include_lowest=True,
)

if merged["age_group"].isna().any():
    raise RuntimeError(
        f"Age grouping lost {int(merged['age_group'].isna().sum()):,} observations."
    )

merged["age_group"] = pd.Categorical(
    merged["age_group"], categories=AGE_LABELS, ordered=True
)

merged.to_csv(
    OUT_DIR / "01_aligned_locked_test_age_regional_predictions.csv",
    index=False,
)

audit = (
    merged.groupby("age_group", observed=False)
    .agg(
        N=("patient_key", "size"),
        Female_N=("sex_true_binary", lambda x: int(np.sum(np.asarray(x) == 0))),
        Male_N=("sex_true_binary", lambda x: int(np.sum(np.asarray(x) == 1))),
        Age_mean=("age", "mean"),
        Age_SD=("age", "std"),
        Age_min=("age", "min"),
        Age_max=("age", "max"),
    )
    .reset_index()
)
audit.to_csv(OUT_DIR / "02_age_group_cohort_audit.csv", index=False)

summary_rows = []
all_boot = []

for age_idx, age_group in enumerate(AGE_LABELS):
    sub = merged.loc[merged["age_group"].astype(str).eq(age_group)].copy()

    y = sub["sex_true_binary"].to_numpy(dtype=int)
    p0 = sub["prob_intact"].to_numpy(dtype=float)
    pp = sub["prob_precordial"].to_numpy(dtype=float)
    pl = sub["prob_limb"].to_numpy(dtype=float)

    if np.unique(y).size < 2:
        raise RuntimeError(f"{age_group}: both sexes required.")

    print("\n" + "=" * 110)
    print(f"AGE GROUP {age_group} | N={len(sub):,}")
    print("=" * 110)

    point = {}
    for m in ["AUROC", "AUPRC"]:
        intact = metric(y, p0, m)
        ab_pre = metric(y, pp, m)
        ab_limb = metric(y, pl, m)
        point[m] = {
            "intact": intact,
            "precordial": ab_pre,
            "limb": ab_limb,
            "drop_precordial": intact - ab_pre,
            "drop_limb": intact - ab_limb,
            "contrast": (intact - ab_pre) - (intact - ab_limb),
        }

    rng = np.random.default_rng(BOOT_SEED + 1000 * age_idx)
    rows = []
    n = len(sub)

    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if np.unique(yy).size < 2:
            continue

        row = {"age_group": age_group, "bootstrap_id": b + 1}

        for m in ["AUROC", "AUPRC"]:
            intact = metric(yy, p0[idx], m)
            ab_pre = metric(yy, pp[idx], m)
            ab_limb = metric(yy, pl[idx], m)

            drop_pre = intact - ab_pre
            drop_limb = intact - ab_limb

            row[f"{m}_intact"] = intact
            row[f"{m}_remove_precordial"] = ab_pre
            row[f"{m}_remove_limb"] = ab_limb
            row[f"{m}_drop_precordial"] = drop_pre
            row[f"{m}_drop_limb"] = drop_limb
            row[f"{m}_contrast_precordial_minus_limb"] = drop_pre - drop_limb

        rows.append(row)

    boot = pd.DataFrame(rows)
    all_boot.append(boot)

    for m in ["AUROC", "AUPRC"]:
        for region, suffix in [("Precordial", "precordial"), ("Limb", "limb")]:
            vals = boot[f"{m}_drop_{suffix}"].to_numpy(dtype=float)

            summary_rows.append(
                {
                    "age_group": age_group,
                    "region_removed": region,
                    "N": len(sub),
                    "Female_N": int(np.sum(y == 0)),
                    "Male_N": int(np.sum(y == 1)),
                    "metric": m,
                    "intact_metric": point[m]["intact"],
                    "ablated_metric": point[m][suffix],
                    "metric_drop": point[m][f"drop_{suffix}"],
                    "drop_CI_low": float(np.percentile(vals, 2.5)),
                    "drop_CI_high": float(np.percentile(vals, 97.5)),
                    "P_drop_two_sided": p_two_sided(vals),
                }
            )

        vals = boot[f"{m}_contrast_precordial_minus_limb"].to_numpy(dtype=float)

        summary_rows.append(
            {
                "age_group": age_group,
                "region_removed": "Precordial-minus-limb contrast",
                "N": len(sub),
                "Female_N": int(np.sum(y == 0)),
                "Male_N": int(np.sum(y == 1)),
                "metric": m,
                "intact_metric": point[m]["intact"],
                "ablated_metric": np.nan,
                "metric_drop": point[m]["contrast"],
                "drop_CI_low": float(np.percentile(vals, 2.5)),
                "drop_CI_high": float(np.percentile(vals, 97.5)),
                "P_drop_two_sided": p_two_sided(vals),
            }
        )

boot_all = pd.concat(all_boot, ignore_index=True)
summary = pd.DataFrame(summary_rows)

summary["age_group"] = pd.Categorical(
    summary["age_group"], categories=AGE_LABELS, ordered=True
)

summary.to_csv(
    OUT_DIR / "03_age_stratified_regional_ablation_summary.csv",
    index=False,
)
boot_all.to_csv(
    OUT_DIR / "04_age_stratified_regional_ablation_bootstrap.csv",
    index=False,
)

auroc = summary.loc[summary["metric"].eq("AUROC")].copy()
auroc.to_csv(
    OUT_DIR / "05_PRIMARY_AUROC_regional_importance_by_age.csv",
    index=False,
)

fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))
x = np.arange(len(AGE_LABELS), dtype=float)

for ax, m in zip(axes, ["AUROC", "AUPRC"]):
    for offset, region in [(-0.08, "Precordial"), (0.08, "Limb")]:
        z = summary.loc[
            summary["metric"].eq(m) & summary["region_removed"].eq(region)
        ].copy()
        z["age_group"] = pd.Categorical(
            z["age_group"], categories=AGE_LABELS, ordered=True
        )
        z = z.sort_values("age_group")

        yv = z["metric_drop"].to_numpy(dtype=float)
        lo = z["drop_CI_low"].to_numpy(dtype=float)
        hi = z["drop_CI_high"].to_numpy(dtype=float)

        ax.errorbar(
            x + offset,
            yv,
            yerr=np.vstack([yv - lo, hi - yv]),
            fmt="o-",
            capsize=3,
            linewidth=1.2,
            label=("Remove V1-V6" if region == "Precordial" else "Remove limb leads"),
        )

    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(AGE_LABELS)
    ax.set_xlabel("Age group (years)")
    ax.set_ylabel(f"{m} drop from intact model")
    ax.set_title(
        "A. Regional AUROC importance by age"
        if m == "AUROC"
        else "B. Regional AUPRC importance by age",
        loc="left",
        fontweight="bold",
    )
    ax.legend(frameon=False, fontsize=8)

fig.tight_layout()
fig.savefig(
    OUT_DIR / "Figure_AgeStratified_RegionalLeadAblation.png",
    dpi=300,
    bbox_inches="tight",
)
fig.savefig(
    OUT_DIR / "Figure_AgeStratified_RegionalLeadAblation.pdf",
    bbox_inches="tight",
)
plt.close(fig)

lines = [
    "WCMC/NYP AGE-STRATIFIED REGIONAL LEAD ABLATION",
    "=" * 105,
    "",
    "Frozen final five-seed VCG-Geometry ResNet; test-time masking only; same locked WCMC/NYP test cohort.",
    "",
    "Age groups: 18-39 | 40-49 | 50-59 | 60-69 | 70-79 | >=80 years",
    "",
    "PRIMARY: AUROC drop = intact AUROC - ablated AUROC.",
    "",
    "AUROC RESULTS",
    "-" * 105,
]

for age_group in AGE_LABELS:
    zp = auroc.loc[
        auroc["age_group"].astype(str).eq(age_group)
        & auroc["region_removed"].eq("Precordial")
    ].iloc[0]

    zl = auroc.loc[
        auroc["age_group"].astype(str).eq(age_group)
        & auroc["region_removed"].eq("Limb")
    ].iloc[0]

    zc = auroc.loc[
        auroc["age_group"].astype(str).eq(age_group)
        & auroc["region_removed"].eq("Precordial-minus-limb contrast")
    ].iloc[0]

    lines.append(
        f"{age_group}: N={int(zp['N']):,}; "
        f"remove V1-V6 ΔAUROC={zp['metric_drop']:+.4f} "
        f"({zp['drop_CI_low']:+.4f} to {zp['drop_CI_high']:+.4f}); "
        f"remove limb leads ΔAUROC={zl['metric_drop']:+.4f} "
        f"({zl['drop_CI_low']:+.4f} to {zl['drop_CI_high']:+.4f}); "
        f"precordial-minus-limb contrast={zc['metric_drop']:+.4f} "
        f"({zc['drop_CI_low']:+.4f} to {zc['drop_CI_high']:+.4f})."
    )

lines += [
    "",
    "INTERPRETATION",
    "-" * 105,
    "These analyses quantify age-specific model reliance on precordial versus limb lead groups. "
    "Because grouped masking is a test-time perturbation, results should be interpreted as "
    "model-dependence sensitivity rather than causal evidence that one anatomical lead region "
    "contains intrinsically greater biological information.",
]

text = "\n".join(lines)
(OUT_DIR / "06_MANUSCRIPT_NUMBERS.txt").write_text(text, encoding="utf-8")

metadata = {
    "analysis": "WCMC age-stratified regional lead ablation",
    "age_groups": AGE_LABELS,
    "age_bins_left_closed": [18, 40, 50, 60, 70, 80, "inf"],
    "regions": {
        "Precordial": "V1-V6",
        "Limb": "I, II, III, aVR, aVL, aVF",
    },
    "N_bootstrap": N_BOOT,
    "bootstrap_seed": BOOT_SEED,
    "baseline_prediction": str(BASELINE_PRED),
    "precordial_prediction": str(PRECORDIAL_PRED),
    "limb_prediction": str(LIMB_PRED),
    "locked_test_split": str(TEST_CSV),
    "note": "No training or neural-network inference; existing frozen-model ablation predictions reused.",
}
(OUT_DIR / "07_analysis_metadata.json").write_text(
    json.dumps(metadata, indent=2), encoding="utf-8"
)

print("\n" + text)
print("\nSaved to:", OUT_DIR)
for p in sorted(OUT_DIR.iterdir()):
    print(" ", p)
