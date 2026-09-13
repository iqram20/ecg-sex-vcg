#!/usr/bin/env python
# ==========================================================================================
# ECG-SEX — MANUSCRIPT TABLE 1 CSV
# Demographic and study characteristics across final ECG cohorts
# REVISED FOR SIX AGE STRATA + TWO-DECIMAL PERCENTAGES
# ==========================================================================================
#
# Reads the already-audited outputs from:
#   Figure2_Demographics_6AgeGroups_v7.py
#
# Age groups:
#   18–39, 40–49, 50–59, 60–69, 70–79, >=80 years
#
# Writes:
#   Table1_ECGSex_Demographic_and_Study_Characteristics.csv
#
# Format:
#   rows    = manuscript characteristics
#   columns = MIMIC-IV, WCMC/NYP, PTB-XL, Chapman/Ningbo
#   encoding = UTF-8 with BOM (Excel-safe; preserves en dashes and >= symbol)
# ==========================================================================================

from pathlib import Path
import pandas as pd


# ==========================================================================================
# PATHS
# ==========================================================================================

RUN = Path(
    "/athena/madelab/scratch/iqh4001/ECG/ECG_Echo/Runs/"
    "Master_ECG_Sex_MIMIC_EchoNext/"
    "20260609_020154_MASTER_ECG_SEX_MIMIC_ECHONEXT"
)

SUPP = RUN / "publication_figures" / "Supplementary"

# Must match the revised demographics script.
STEM = SUPP / "Figure2_Comprehensive_Demographics_FinalAnalysisCohorts_6AgeGroups_v7"

SUMMARY_CSV = Path(str(STEM) + "_Summary.csv")
AGEGROUP_CSV = Path(str(STEM) + "_AgeGroups.csv")
AUDIT_CSV = Path(str(STEM) + "_CohortAudit.csv")

OUT_DIR = RUN / "publication_tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "Table1_ECGSex_Demographic_and_Study_Characteristics.csv"


# ==========================================================================================
# FINAL COHORT DEFINITIONS
# ==========================================================================================

ORDER = [
    "MIMIC-IV",
    "WCMC/NYP",
    "PTB-XL",
    "Chapman/Ningbo",
]

EXPECTED_N = {
    "MIMIC-IV": 155626,
    "WCMC/NYP": 106868,
    "PTB-XL": 18752,
    "Chapman/Ningbo": 42997,
}

STUDY_ROLE = {
    "MIMIC-IV": "Pretraining",
    "WCMC/NYP": "Fine-tuning / primary evaluation",
    "PTB-XL": "External validation; data-efficiency analysis",
    "Chapman/Ningbo": "External validation",
}

TRAIN_N = {
    "MIMIC-IV": 132282,
    "WCMC/NYP": 74807,
    "PTB-XL": None,
    "Chapman/Ningbo": None,
}

VALIDATION_N = {
    "MIMIC-IV": 23344,
    "WCMC/NYP": 10687,
    "PTB-XL": None,
    "Chapman/Ningbo": None,
}

TEST_N = {
    "MIMIC-IV": None,
    "WCMC/NYP": 21374,
    "PTB-XL": 18752,
    "Chapman/Ningbo": 42997,
}

AGE_LABELS = [
    "18–39",
    "40–49",
    "50–59",
    "60–69",
    "70–79",
    "≥80",
]


# ==========================================================================================
# HELPERS
# ==========================================================================================

def require(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required demographics output not found:\n{path}\n\n"
            "Run Figure2_Demographics_6AgeGroups_v7.py first."
        )


def nfmt(x):
    if x is None or pd.isna(x):
        return "—"
    return f"{int(x):,}"


def pctfmt(x):
    if pd.isna(x):
        return "—"
    return f"{float(x):.2f}"


def n_pct(n, pct):
    if pd.isna(n) or pd.isna(pct):
        return "—"
    return f"{int(n):,} ({float(pct):.2f})"


def add_row(rows, characteristic, values):
    rows.append(
        {
            "Characteristic": characteristic,
            **{dataset: values[dataset] for dataset in ORDER},
        }
    )


# ==========================================================================================
# LOAD AUDITED DEMOGRAPHICS OUTPUTS
# ==========================================================================================

for p in [SUMMARY_CSV, AGEGROUP_CSV, AUDIT_CSV]:
    require(p)

summary = pd.read_csv(SUMMARY_CSV)
agegroups = pd.read_csv(AGEGROUP_CSV)
audit = pd.read_csv(AUDIT_CSV)

required_summary_cols = {
    "dataset",
    "N_analysis",
    "Female_N",
    "Male_N",
    "Female_pct",
    "Male_pct",
    "Age_exact_N",
    "Age_exact_pct",
    "Age_mean",
    "Age_SD",
    "Age_median",
    "Age_Q1",
    "Age_Q3",
}

required_age_cols = {
    "dataset",
    "age_group",
    "N",
    "denominator",
    "Percent",
}

required_audit_cols = {
    "dataset",
    "expected_N",
    "observed_N",
    "unique_subjects",
    "unique_ECGs",
    "exact_age_N",
    "age_band_N",
    "PTB_privacy_ge90_N",
}

for label, df, required in [
    ("Summary", summary, required_summary_cols),
    ("Age-group", agegroups, required_age_cols),
    ("Audit", audit, required_audit_cols),
]:
    missing = sorted(required.difference(df.columns))
    if missing:
        raise RuntimeError(
            f"{label} file is missing required columns: {missing}\n"
            f"Observed columns: {df.columns.tolist()}"
        )

# Require exactly the expected four datasets.
for label, df in [
    ("Summary", summary),
    ("Age-group", agegroups),
    ("Audit", audit),
]:
    observed_datasets = set(df["dataset"].astype(str))
    expected_datasets = set(ORDER)

    extra = sorted(observed_datasets - expected_datasets)
    missing = sorted(expected_datasets - observed_datasets)

    if extra or missing:
        raise RuntimeError(
            f"{label} dataset mismatch.\n"
            f"Missing: {missing}\n"
            f"Unexpected: {extra}"
        )

summary = summary.set_index("dataset").loc[ORDER]
audit = audit.set_index("dataset").loc[ORDER]


# ==========================================================================================
# HARD COHORT QC
# ==========================================================================================

for dataset in ORDER:
    expected = EXPECTED_N[dataset]

    observed = int(summary.loc[dataset, "N_analysis"])
    if observed != expected:
        raise RuntimeError(
            f"{dataset}: summary N mismatch. "
            f"Expected {expected:,}, observed {observed:,}."
        )

    audit_expected = int(audit.loc[dataset, "expected_N"])
    audit_observed = int(audit.loc[dataset, "observed_N"])
    unique_subjects = int(audit.loc[dataset, "unique_subjects"])
    unique_ecgs = int(audit.loc[dataset, "unique_ECGs"])

    if audit_expected != expected:
        raise RuntimeError(
            f"{dataset}: audit expected_N={audit_expected:,}, "
            f"but frozen expected N={expected:,}."
        )

    if audit_observed != expected:
        raise RuntimeError(
            f"{dataset}: audit observed_N={audit_observed:,}, "
            f"expected {expected:,}."
        )

    if unique_subjects != expected:
        raise RuntimeError(
            f"{dataset}: unique_subjects={unique_subjects:,}, "
            f"expected one participant per cohort row ({expected:,})."
        )

    if unique_ecgs != expected:
        raise RuntimeError(
            f"{dataset}: unique_ECGs={unique_ecgs:,}, "
            f"expected one ECG per participant ({expected:,})."
        )


# ==========================================================================================
# STUDY-STAGE ALLOCATION QC
# ==========================================================================================

for dataset in ORDER:
    allocated = sum(
        x
        for x in [
            TRAIN_N[dataset],
            VALIDATION_N[dataset],
            TEST_N[dataset],
        ]
        if x is not None
    )

    if allocated != EXPECTED_N[dataset]:
        raise RuntimeError(
            f"{dataset}: train/validation/test allocation sums to "
            f"{allocated:,}, not {EXPECTED_N[dataset]:,}."
        )


# ==========================================================================================
# SIX-AGE-BAND QC + LOOKUP
# ==========================================================================================

# This prevents accidental use of the old four-bin demographics file.
observed_age_labels = set(agegroups["age_group"].astype(str))
expected_age_labels = set(AGE_LABELS)

missing_age_labels = sorted(expected_age_labels - observed_age_labels)
unexpected_age_labels = sorted(observed_age_labels - expected_age_labels)

if missing_age_labels or unexpected_age_labels:
    raise RuntimeError(
        "Age-group file does not match the revised six-stratum definition.\n"
        f"Expected: {AGE_LABELS}\n"
        f"Missing: {missing_age_labels}\n"
        f"Unexpected: {unexpected_age_labels}\n"
        "Run Figure2_Demographics_6AgeGroups_v7.py before creating Table 1."
    )

age_lookup = {}

for dataset in ORDER:
    age_lookup[dataset] = {}

    dataset_rows = agegroups.loc[
        agegroups["dataset"].eq(dataset)
    ].copy()

    if len(dataset_rows) != len(AGE_LABELS):
        raise RuntimeError(
            f"{dataset}: expected {len(AGE_LABELS)} age-group rows; "
            f"found {len(dataset_rows)}."
        )

    denominators = dataset_rows["denominator"].dropna().unique()

    if len(denominators) != 1:
        raise RuntimeError(
            f"{dataset}: inconsistent age-group denominators: "
            f"{denominators.tolist()}"
        )

    age_denominator = int(denominators[0])
    expected_age_band_n = int(audit.loc[dataset, "age_band_N"])

    if age_denominator != expected_age_band_n:
        raise RuntimeError(
            f"{dataset}: age-group denominator {age_denominator:,} "
            f"does not match audited age_band_N {expected_age_band_n:,}."
        )

    band_n_total = int(dataset_rows["N"].sum())

    if band_n_total != age_denominator:
        raise RuntimeError(
            f"{dataset}: age-band counts sum to {band_n_total:,}, "
            f"but denominator is {age_denominator:,}."
        )

    pct_total = float(dataset_rows["Percent"].sum())

    if abs(pct_total - 100.0) > 0.05:
        raise RuntimeError(
            f"{dataset}: age-band percentages sum to {pct_total:.4f}%, "
            "expected approximately 100%."
        )

    for age_group in AGE_LABELS:
        row = dataset_rows.loc[
            dataset_rows["age_group"].eq(age_group)
        ]

        if len(row) != 1:
            raise RuntimeError(
                f"Expected exactly one age-group row for "
                f"{dataset} / {age_group}; found {len(row)}."
            )

        age_lookup[dataset][age_group] = {
            "N": int(row.iloc[0]["N"]),
            "Percent": float(row.iloc[0]["Percent"]),
            "Denominator": int(row.iloc[0]["denominator"]),
        }


# ==========================================================================================
# BUILD MANUSCRIPT TABLE
# ==========================================================================================

rows = []

add_row(
    rows,
    "Study role",
    {d: STUDY_ROLE[d] for d in ORDER},
)

add_row(
    rows,
    "Participants, N",
    {d: nfmt(summary.loc[d, "N_analysis"]) for d in ORDER},
)

add_row(
    rows,
    "ECGs, N",
    {d: nfmt(summary.loc[d, "N_analysis"]) for d in ORDER},
)

add_row(
    rows,
    "Training, N",
    {d: nfmt(TRAIN_N[d]) for d in ORDER},
)

add_row(
    rows,
    "Validation, N",
    {d: nfmt(VALIDATION_N[d]) for d in ORDER},
)

add_row(
    rows,
    "Test / external evaluation, N",
    {d: nfmt(TEST_N[d]) for d in ORDER},
)

add_row(
    rows,
    "Female, N (%)",
    {
        d: n_pct(
            summary.loc[d, "Female_N"],
            summary.loc[d, "Female_pct"],
        )
        for d in ORDER
    },
)

add_row(
    rows,
    "Male, N (%)",
    {
        d: n_pct(
            summary.loc[d, "Male_N"],
            summary.loc[d, "Male_pct"],
        )
        for d in ORDER
    },
)

add_row(
    rows,
    "Exact age available, N (%)",
    {
        d: n_pct(
            summary.loc[d, "Age_exact_N"],
            summary.loc[d, "Age_exact_pct"],
        )
        for d in ORDER
    },
)

add_row(
    rows,
    "Age, years, mean ± SD",
    {
        d: (
            f"{summary.loc[d, 'Age_mean']:.1f} ± "
            f"{summary.loc[d, 'Age_SD']:.1f}"
        )
        for d in ORDER
    },
)

add_row(
    rows,
    "Age, years, median [IQR]",
    {
        d: (
            f"{summary.loc[d, 'Age_median']:.0f} "
            f"[{summary.loc[d, 'Age_Q1']:.0f}–"
            f"{summary.loc[d, 'Age_Q3']:.0f}]"
        )
        for d in ORDER
    },
)

for age_group in AGE_LABELS:
    add_row(
        rows,
        f"Age {age_group} years, %",
        {
            d: pctfmt(
                age_lookup[d][age_group]["Percent"]
            )
            for d in ORDER
        },
    )

# PTB-XL age-privacy note.
ptb_privacy_n = int(audit.loc["PTB-XL", "PTB_privacy_ge90_N"])

if ptb_privacy_n != 257:
    raise RuntimeError(
        f"PTB-XL privacy-coded >=90 count changed: "
        f"expected 257, observed {ptb_privacy_n}."
    )

# Exact-age missingness note.
mimic_missing_age = int(
    EXPECTED_N["MIMIC-IV"] - summary.loc["MIMIC-IV", "Age_exact_N"]
)

wcmc_missing_age = int(
    EXPECTED_N["WCMC/NYP"] - summary.loc["WCMC/NYP", "Age_exact_N"]
)

if mimic_missing_age != 27:
    raise RuntimeError(
        f"MIMIC-IV exact-age missing count changed: "
        f"expected 27, observed {mimic_missing_age}."
    )

if wcmc_missing_age != 4:
    raise RuntimeError(
        f"WCMC/NYP exact-age missing count changed: "
        f"expected 4, observed {wcmc_missing_age}."
    )

footnote_text = (
    "Percentages may not sum to 100% because of rounding. "
    "Training, validation, and test/external-evaluation counts refer to the "
    "primary study-stage allocation. The separate PTB-XL data-efficiency "
    "experiment used predefined folds 1–8 for training (n=14,926), fold 9 "
    "for validation (n=1,933), and fold 10 for testing (n=1,893). "
    f"Exact age was unavailable for {mimic_missing_age} MIMIC-IV participants "
    f"and {wcmc_missing_age} WCMC/NYP participants; therefore, exact-age "
    "availability rounds to 100.0% in these cohorts when reported to one "
    "decimal place. "
    f"In PTB-XL, {ptb_privacy_n} participants had privacy-coded age ≥90 years; "
    "these participants were retained in the ≥80-year category but excluded "
    "from exact-age mean, SD, median, and IQR calculations."
)

add_row(
    rows,
    "Footnote",
    {
        "MIMIC-IV": footnote_text,
        "WCMC/NYP": "",
        "PTB-XL": "",
        "Chapman/Ningbo": "",
    },
)

table = pd.DataFrame(rows)


# ==========================================================================================
# SAVE
# ==========================================================================================

table.to_csv(
    OUT_CSV,
    index=False,
    encoding="utf-8-sig",
)

print("=" * 120)
print("MANUSCRIPT TABLE 1 CSV CREATED — SIX AGE STRATA")
print("=" * 120)

print("\nInput summary:")
print(SUMMARY_CSV)

print("\nInput age groups:")
print(AGEGROUP_CSV)

print("\nInput audit:")
print(AUDIT_CSV)

print("\nOutput:")
print(OUT_CSV)

print("\nFinal manuscript Table 1:")
print(table.to_string(index=False))

print("\nDONE")
