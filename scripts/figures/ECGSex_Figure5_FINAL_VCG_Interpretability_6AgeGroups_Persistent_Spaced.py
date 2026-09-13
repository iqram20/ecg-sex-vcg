#!/usr/bin/env python
# ==========================================================================================
# ECG-SEX — FIGURE 5 FINAL VCG INTERPRETABILITY — SIX AGE GROUPS
# CURRENT FINAL RUN | 2 × 2 | PLASMA
# ==========================================================================================
#
# A. Overall single-lead ablation — horizontal bar plot + paired-bootstrap 95% CI
# B. Precordial vs limb lead importance — boxplot + individual leads
# C. VCG mechanistic ablation — horizontal bar plot + paired-bootstrap 95% CI
# D. Age-stratified single-lead ablation — grouped forest plot + 95% CI
#
# AGE GROUPS
# ----------
# 18–39, 40–49, 50–59, 60–69, 70–79, >=80 years
#
# IMPORTANT
# ---------
# - NO waveform panel.
# - NO heatmap.
# - NO model inference/training.
# - Panel D reuses the EXISTING final 5-seed ensemble condition predictions:
#       21_vcg_geometry_resnet_interpretability/01_condition_predictions/baseline.csv
#       21_vcg_geometry_resnet_interpretability/01_condition_predictions/lead_ablation_<LEAD>.csv
# - Age is taken from the locked WCMC test split when available; otherwise it is joined
#   by ecg_key from the known WCMC master parquet.
# - Panel D uses paired patient-level bootstrap within each age stratum.
# - Results are cached, so subsequent figure runs are fast.
# - Lead ablation quantifies model reliance under perturbation, not causality.
#
# Female=0, Male=1
# ==========================================================================================

from pathlib import Path
import os
import json
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")


# ==========================================================================================
# CONFIG
# ==========================================================================================

N_BOOT = int(
    os.environ.get(
        "AGELEAD_N_BOOT",
        "2000",
    )
)

N_JOBS = int(
    os.environ.get(
        "AGELEAD_N_JOBS",
        "8",
    )
)

BOOT_SEED = 42

# Reuse saved six-group results by default.
# Set AGELEAD_FORCE_RECOMPUTE=1 only when you intentionally want to rerun
# all 72 age × lead paired-bootstrap analyses.
FORCE_RECOMPUTE = (
    os.environ.get(
        "AGELEAD_FORCE_RECOMPUTE",
        "0",
    )
    ==
    "1"
)

LEADS = [
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]

PRECORDIAL = [
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]

AGE_LABELS = [
    "18–39",
    "40–49",
    "50–59",
    "60–69",
    "70–79",
    "≥80",
]

AGE_BINS = [
    18,
    40,
    50,
    60,
    70,
    80,
    np.inf,
]


# ==========================================================================================
# EXACT CURRENT-RUN PATHS
# ==========================================================================================

RUN_DIR = Path(
    "/athena/madelab/scratch/iqh4001/ECG/ECG_Echo/Runs/"
    "Master_ECG_Sex_MIMIC_EchoNext/"
    "20260609_020154_MASTER_ECG_SEX_MIMIC_ECHONEXT"
)

WCMC_ROOT = (
    RUN_DIR
    /
    "wcmc_ecgsex_from_scratch"
)

INTERP_ROOT = (
    WCMC_ROOT
    /
    "21_vcg_geometry_resnet_interpretability"
)

PRED_DIR = (
    INTERP_ROOT
    /
    "01_condition_predictions"
)

FINAL_DIR = (
    INTERP_ROOT
    /
    "04_final_summary"
)

TEST_CSV = (
    WCMC_ROOT
    /
    "04_wcmc_finetune_from_mimic"
    /
    "splits"
    /
    "WCMC_ECGSex_test.csv"
)

# Known WCMC master cohort containing ecg_age_at_ecg.
MASTER_PARQUET = Path(
    "/athena/madelab/scratch/iqh4001/Azure_Cradle_ECG/"
    "Data/ECG-Echo_MetaData/ecg_echo_master_cohort_v1.parquet"
)

BASELINE_PRED = (
    PRED_DIR
    /
    "baseline.csv"
)

LEAD_PRED = {
    lead:
        PRED_DIR
        /
        f"lead_ablation_{lead}.csv"
    for lead in LEADS
}

OVERALL_LEAD_CSV = (
    FINAL_DIR
    /
    "LeadAblation_Ranked_By_AUROC_Drop.csv"
)

MECH_CSV = (
    FINAL_DIR
    /
    "VCG_Mechanistic_Ablation_Ranked.csv"
)

AGE_CACHE = (
    FINAL_DIR
    /
    "LeadAblation_AgeGrouped_6AgeGroups_AUROCdrop_2000Bootstrap.csv"
)

AGE_COHORT_AUDIT = (
    FINAL_DIR
    /
    "LeadAblation_AgeGrouped_6AgeGroups_CohortAudit.csv"
)

# Additional persistent files for later reuse / provenance.
AGE_CACHE_PARQUET = (
    FINAL_DIR
    /
    "LeadAblation_AgeGrouped_6AgeGroups_AUROCdrop_2000Bootstrap.parquet"
)

AGE_METADATA_JSON = (
    FINAL_DIR
    /
    "LeadAblation_AgeGrouped_6AgeGroups_Metadata.json"
)

OUT_DIR = (
    RUN_DIR
    /
    "publication_figures"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

STEM = (
    OUT_DIR
    /
    "Figure5_FINAL_VCG_Interpretability_2x2_6AgeGroups_Plasma_Spaced"
)

PNG_OUT = Path(
    str(STEM)
    +
    ".png"
)

PDF_OUT = Path(
    str(STEM)
    +
    ".pdf"
)

SVG_OUT = Path(
    str(STEM)
    +
    ".svg"
)

SOURCE_OUT = Path(
    str(STEM)
    +
    "_SourceData.csv"
)


# ==========================================================================================
# FILE AUDIT
# ==========================================================================================

# Panels A-C always require their final summary tables.
required_plot_files = [
    OVERALL_LEAD_CSV,
    MECH_CSV,
]

missing_plot_files = [
    p
    for p in required_plot_files
    if not p.exists()
]

if missing_plot_files:
    raise FileNotFoundError(
        "Missing required final summary file(s):\n"
        +
        "\n".join(
            str(p)
            for p in missing_plot_files
        )
    )

# Raw Panel-D inputs are checked only if the six-group result cache has to be rebuilt.
# This allows later plotting/reporting directly from the saved result table.


# ==========================================================================================
# HELPERS
# ==========================================================================================

def require_columns(
    df,
    cols,
    label,
):
    missing_cols = [
        c
        for c in cols
        if c not in df.columns
    ]

    if missing_cols:
        raise RuntimeError(
            f"{label} missing columns: {missing_cols}\n"
            f"Observed: {df.columns.tolist()}"
        )


def load_condition_prediction(
    path,
):
    df = pd.read_csv(
        path,
        low_memory=False,
    )

    require_columns(
        df,
        [
            "patient_key",
            "ecg_key",
            "sex_true_binary",
            "prob_male",
        ],
        str(
            path
        ),
    )

    out = df[
        [
            "patient_key",
            "ecg_key",
            "sex_true_binary",
            "prob_male",
        ]
    ].copy()

    out[
        "patient_key"
    ] = out[
        "patient_key"
    ].astype(
        str
    )

    out[
        "ecg_key"
    ] = out[
        "ecg_key"
    ].astype(
        str
    )

    out[
        "sex_true_binary"
    ] = pd.to_numeric(
        out[
            "sex_true_binary"
        ],
        errors="raise",
    ).astype(
        int
    )

    out[
        "prob_male"
    ] = pd.to_numeric(
        out[
            "prob_male"
        ],
        errors="raise",
    )

    if out[
        [
            "patient_key",
            "ecg_key",
        ]
    ].duplicated().any():
        raise RuntimeError(
            f"Duplicate patient/ECG rows in {path}"
        )

    return out


def require_age_rebuild_inputs():
    """
    Raw files needed only when the age-stratified six-group results must be recomputed.
    Cached-result plotting does not depend on these files.
    """
    required = [
        TEST_CSV,
        BASELINE_PRED,
    ] + list(
        LEAD_PRED.values()
    )

    missing = [
        p
        for p in required
        if not p.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Six-group age cache is unavailable/invalid and raw rebuild inputs are missing:\n"
            +
            "\n".join(
                str(p)
                for p in missing
            )
        )


def resolve_age_table():
    """
    Prefer age already carried in exact locked-test split.
    If absent, use WCMC master parquet and join by ecg_key.
    """
    test = pd.read_csv(
        TEST_CSV,
        low_memory=False,
    )

    require_columns(
        test,
        [
            "patient_key",
            "ecg_key",
        ],
        "Locked WCMC test split",
    )

    test[
        "patient_key"
    ] = test[
        "patient_key"
    ].astype(
        str
    )

    test[
        "ecg_key"
    ] = test[
        "ecg_key"
    ].astype(
        str
    )

    age_candidates = [
        "ecg_age_at_ecg",
        "age",
        "age_at_ecg",
        "patient_age",
    ]

    age_col = next(
        (
            c
            for c in age_candidates
            if c in test.columns
        ),
        None,
    )

    if age_col is not None:
        print(
            f"Age source: locked test split | column={age_col}"
        )

        age = test[
            [
                "patient_key",
                "ecg_key",
                age_col,
            ]
        ].copy()

        age = age.rename(
            columns={
                age_col:
                    "age"
            }
        )

    else:
        if not MASTER_PARQUET.exists():
            raise FileNotFoundError(
                "Age not present in locked test split and master parquet is missing:\n"
                f"{MASTER_PARQUET}"
            )

        print(
            "Age not present in test split."
        )

        print(
            "Age source: WCMC master parquet"
        )

        print(
            MASTER_PARQUET
        )

        master = pd.read_parquet(
            MASTER_PARQUET,
            columns=[
                "patient_key",
                "ecg_key",
                "ecg_age_at_ecg",
            ],
        )

        master[
            "patient_key"
        ] = master[
            "patient_key"
        ].astype(
            str
        )

        master[
            "ecg_key"
        ] = master[
            "ecg_key"
        ].astype(
            str
        )

        master[
            "ecg_age_at_ecg"
        ] = pd.to_numeric(
            master[
                "ecg_age_at_ecg"
            ],
            errors="coerce",
        )

        # Exact ECG key should be unique in the master.
        if master[
            "ecg_key"
        ].duplicated().any():
            # Conservatively require identical age if duplicates occur.
            age_nunique = (
                master
                .groupby(
                    "ecg_key"
                )[
                    "ecg_age_at_ecg"
                ]
                .nunique(
                    dropna=True
                )
            )

            bad = age_nunique[
                age_nunique
                >
                1
            ]

            if len(
                bad
            ):
                raise RuntimeError(
                    "Conflicting ages for duplicated ecg_key values in master parquet."
                )

            master = master.drop_duplicates(
                "ecg_key"
            )

        age = (
            test[
                [
                    "patient_key",
                    "ecg_key",
                ]
            ]
            .merge(
                master[
                    [
                        "ecg_key",
                        "ecg_age_at_ecg",
                    ]
                ],
                on="ecg_key",
                how="left",
                validate="one_to_one",
            )
            .rename(
                columns={
                    "ecg_age_at_ecg":
                        "age"
                }
            )
        )

    age[
        "age"
    ] = pd.to_numeric(
        age[
            "age"
        ],
        errors="coerce",
    )

    age = age.loc[
        age[
            "age"
        ]
        >=
        18
    ].copy()

    age[
        "age_group"
    ] = pd.cut(
        age[
            "age"
        ],
        bins=AGE_BINS,
        labels=AGE_LABELS,
        right=False,
        include_lowest=True,
    )

    age = age.dropna(
        subset=[
            "age_group"
        ]
    ).copy()

    age[
        "age_group"
    ] = age[
        "age_group"
    ].astype(
        str
    )

    if age[
        "patient_key"
    ].duplicated().any():
        raise RuntimeError(
            "Age table contains duplicate locked-test patients."
        )

    return age


def calc_auc(
    y,
    p,
):
    y = np.asarray(
        y,
        dtype=int,
    )

    p = np.asarray(
        p,
        dtype=float,
    )

    if np.unique(
        y
    ).size < 2:
        return np.nan

    return float(
        roc_auc_score(
            y,
            p,
        )
    )


def age_lead_bootstrap(
    age_group,
    lead_name,
    data,
    n_boot,
    seed,
):
    """
    Paired bootstrap of:
        AUROC(baseline) - AUROC(lead-ablated)
    within one age stratum.
    """
    sub = data.loc[
        data[
            "age_group"
        ]
        ==
        age_group
    ].copy()

    y = sub[
        "sex_true_binary"
    ].to_numpy(
        dtype=int
    )

    pb = sub[
        "prob_baseline"
    ].to_numpy(
        dtype=float
    )

    pc = sub[
        "prob_condition"
    ].to_numpy(
        dtype=float
    )

    n = len(
        sub
    )

    n_female = int(
        np.sum(
            y
            ==
            0
        )
    )

    n_male = int(
        np.sum(
            y
            ==
            1
        )
    )

    if (
        n < 30
        or
        n_female == 0
        or
        n_male == 0
    ):
        return {
            "age_group":
                age_group,

            "lead":
                lead_name,

            "N":
                n,

            "Female_N":
                n_female,

            "Male_N":
                n_male,

            "AUROC_baseline":
                np.nan,

            "AUROC_after_removal":
                np.nan,

            "AUROC_drop":
                np.nan,

            "AUROC_drop_CI_low":
                np.nan,

            "AUROC_drop_CI_high":
                np.nan,

            "bootstrap_replicates":
                0,
        }

    auc_base = calc_auc(
        y,
        pb,
    )

    auc_cond = calc_auc(
        y,
        pc,
    )

    point_drop = (
        auc_base
        -
        auc_cond
    )

    rng = np.random.default_rng(
        seed
    )

    boot = np.empty(
        n_boot,
        dtype=np.float64,
    )

    n_valid = 0

    for _ in range(
        n_boot
    ):
        idx = rng.integers(
            0,
            n,
            size=n,
        )

        yb = y[
            idx
        ]

        if np.unique(
            yb
        ).size < 2:
            continue

        boot[
            n_valid
        ] = (
            roc_auc_score(
                yb,
                pb[
                    idx
                ],
            )
            -
            roc_auc_score(
                yb,
                pc[
                    idx
                ],
            )
        )

        n_valid += 1

    if n_valid < max(
        100,
        int(
            0.8
            *
            n_boot
        ),
    ):
        raise RuntimeError(
            f"Too few valid bootstrap replicates for age={age_group}, lead={lead_name}: "
            f"{n_valid}/{n_boot}"
        )

    boot = boot[
        :
        n_valid
    ]

    return {
        "age_group":
            age_group,

        "lead":
            lead_name,

        "N":
            n,

        "Female_N":
            n_female,

        "Male_N":
            n_male,

        "AUROC_baseline":
            auc_base,

        "AUROC_after_removal":
            auc_cond,

        "AUROC_drop":
            point_drop,

        "AUROC_drop_CI_low":
            float(
                np.percentile(
                    boot,
                    2.5,
                )
            ),

        "AUROC_drop_CI_high":
            float(
                np.percentile(
                    boot,
                    97.5,
                )
            ),

        "bootstrap_replicates":
            int(
                n_valid
            ),
    }


def build_age_grouped_results():
    require_age_rebuild_inputs()

    print(
        "=" * 120
    )

    print(
        "BUILDING AGE-STRATIFIED LEAD ABLATION FROM EXISTING CURRENT-RUN PREDICTIONS"
    )

    print(
        "=" * 120
    )

    baseline = load_condition_prediction(
        BASELINE_PRED
    ).rename(
        columns={
            "prob_male":
                "prob_baseline"
        }
    )

    age = resolve_age_table()

    base_age = baseline.merge(
        age[
            [
                "patient_key",
                "ecg_key",
                "age",
                "age_group",
            ]
        ],
        on=[
            "patient_key",
            "ecg_key",
        ],
        how="inner",
        validate="one_to_one",
    )

    if len(
        base_age
    ) != len(
        baseline
    ):
        print(
            f"WARNING: baseline N={len(baseline):,}; age-linked N={len(base_age):,}"
        )

    if len(
        base_age
    ) < 0.95 * len(
        baseline
    ):
        raise RuntimeError(
            "More than 5% of locked-test predictions lack age linkage."
        )

    cohort_audit = (
        base_age
        .groupby(
            "age_group",
            observed=False,
        )
        .agg(
            N=(
                "patient_key",
                "size",
            ),

            Female_N=(
                "sex_true_binary",
                lambda x:
                    int(
                        np.sum(
                            np.asarray(
                                x
                            )
                            ==
                            0
                        )
                    ),
            ),

            Male_N=(
                "sex_true_binary",
                lambda x:
                    int(
                        np.sum(
                            np.asarray(
                                x
                            )
                            ==
                            1
                        )
                    ),
            ),

            Age_mean=(
                "age",
                "mean",
            ),

            Age_SD=(
                "age",
                "std",
            ),

            Age_min=(
                "age",
                "min",
            ),

            Age_max=(
                "age",
                "max",
            ),
        )
        .reset_index()
    )

    cohort_audit[
        "age_group"
    ] = pd.Categorical(
        cohort_audit[
            "age_group"
        ],
        categories=AGE_LABELS,
        ordered=True,
    )

    cohort_audit = cohort_audit.sort_values(
        "age_group"
    )

    cohort_audit.to_csv(
        AGE_COHORT_AUDIT,
        index=False,
    )

    print(
        "\nAge-group cohort audit:"
    )

    print(
        cohort_audit.to_string(
            index=False
        )
    )

    # Read each ablation prediction once and create 12 aligned tables.
    aligned_by_lead = {}

    for lead_name in LEADS:
        cond = load_condition_prediction(
            LEAD_PRED[
                lead_name
            ]
        ).rename(
            columns={
                "prob_male":
                    "prob_condition",

                "sex_true_binary":
                    "sex_condition",
            }
        )

        merged = base_age.merge(
            cond[
                [
                    "patient_key",
                    "ecg_key",
                    "sex_condition",
                    "prob_condition",
                ]
            ],
            on=[
                "patient_key",
                "ecg_key",
            ],
            how="inner",
            validate="one_to_one",
        )

        if not np.array_equal(
            merged[
                "sex_true_binary"
            ].to_numpy(
                dtype=int
            ),
            merged[
                "sex_condition"
            ].to_numpy(
                dtype=int
            ),
        ):
            raise RuntimeError(
                f"Label mismatch for lead {lead_name}"
            )

        if len(
            merged
        ) != len(
            base_age
        ):
            raise RuntimeError(
                f"Alignment mismatch for {lead_name}: {len(merged):,} vs {len(base_age):,}"
            )

        aligned_by_lead[
            lead_name
        ] = merged

    # 72 independent lead × age calculations can be parallelized.
    jobs = []

    for lead_idx, lead_name in enumerate(
        LEADS
    ):
        for age_idx, age_group in enumerate(
            AGE_LABELS
        ):
            deterministic_seed = (
                BOOT_SEED
                +
                1000
                *
                lead_idx
                +
                37
                *
                age_idx
            )

            jobs.append(
                (
                    age_group,
                    lead_name,
                    aligned_by_lead[
                        lead_name
                    ],
                    deterministic_seed,
                )
            )

    rows = Parallel(
        n_jobs=N_JOBS,
        backend="loky",
        verbose=5,
    )(
        delayed(
            age_lead_bootstrap
        )(
            age_group,
            lead_name,
            data,
            N_BOOT,
            seed,
        )
        for (
            age_group,
            lead_name,
            data,
            seed,
        ) in jobs
    )

    out = pd.DataFrame(
        rows
    )

    out[
        "age_group"
    ] = pd.Categorical(
        out[
            "age_group"
        ],
        categories=AGE_LABELS,
        ordered=True,
    )

    lead_cat = pd.CategoricalDtype(
        categories=LEADS,
        ordered=True,
    )

    out[
        "lead"
    ] = out[
        "lead"
    ].astype(
        lead_cat
    )

    out = out.sort_values(
        [
            "age_group",
            "lead",
        ]
    ).reset_index(
        drop=True
    )

    out.to_csv(
        AGE_CACHE,
        index=False,
    )

    # Also save Parquet when the environment supports it. CSV remains the
    # canonical portable result file.
    try:
        out.to_parquet(
            AGE_CACHE_PARQUET,
            index=False,
        )
        parquet_status = str(
            AGE_CACHE_PARQUET
        )
    except Exception as exc:
        parquet_status = (
            "Parquet not written: "
            +
            repr(
                exc
            )
        )

    metadata = {
        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "analysis":
            "WCMC/NYP six-age-group single-lead ablation",

        "age_groups":
            AGE_LABELS,

        "age_bins_left_closed":
            [
                18,
                40,
                50,
                60,
                70,
                80,
                "inf",
            ],

        "n_boot":
            int(
                N_BOOT
            ),

        "bootstrap_seed":
            int(
                BOOT_SEED
            ),

        "expected_rows":
            int(
                len(
                    AGE_LABELS
                )
                *
                len(
                    LEADS
                )
            ),

        "summary_csv":
            str(
                AGE_CACHE
            ),

        "summary_parquet":
            parquet_status,

        "cohort_audit_csv":
            str(
                AGE_COHORT_AUDIT
            ),

        "baseline_prediction":
            str(
                BASELINE_PRED
            ),

        "lead_prediction_directory":
            str(
                PRED_DIR
            ),

        "test_split":
            str(
                TEST_CSV
            ),
    }

    AGE_METADATA_JSON.write_text(
        json.dumps(
            metadata,
            indent=2,
        )
    )

    print(
        "\nSaved age-grouped lead ablation:"
    )

    print(
        AGE_CACHE
    )

    print(
        "Metadata:"
    )

    print(
        AGE_METADATA_JSON
    )

    print(
        "Parquet status:"
    )

    print(
        parquet_status
    )

    return out


# ==========================================================================================
# LOAD OVERALL A / B / C
# ==========================================================================================

lead = pd.read_csv(
    OVERALL_LEAD_CSV,
    low_memory=False,
)

mech = pd.read_csv(
    MECH_CSV,
    low_memory=False,
)

require_columns(
    lead,
    [
        "lead",
        "AUROC_drop",
        "AUROC_drop_CI_low",
        "AUROC_drop_CI_high",
    ],
    "Overall lead ablation",
)

require_columns(
    mech,
    [
        "condition",
        "AUROC_drop",
        "AUROC_drop_CI_low",
        "AUROC_drop_CI_high",
    ],
    "VCG mechanistic ablation",
)

for c in [
    "AUROC_drop",
    "AUROC_drop_CI_low",
    "AUROC_drop_CI_high",
]:
    lead[
        c
    ] = pd.to_numeric(
        lead[
            c
        ],
        errors="raise",
    )

    mech[
        c
    ] = pd.to_numeric(
        mech[
            c
        ],
        errors="raise",
    )

lead[
    "region"
] = np.where(
    lead[
        "lead"
    ].isin(
        PRECORDIAL
    ),
    "Precordial",
    "Limb",
)

lead = (
    lead
    .sort_values(
        "AUROC_drop",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)

OVERALL_LEAD_ORDER = lead[
    "lead"
].tolist()


MECH_LABELS = {
    "vcg_branch_off":
        "VCG branch off",

    "vcg_no_direct_xyz":
        "No direct XYZ",

    "vcg_no_magnitude_direction":
        "No magnitude/direction",

    "vcg_no_trajectory_turning":
        "No trajectory turning",

    "vcg_no_velocity_dynamics":
        "No velocity dynamics",

    "fixed_inverse_dower":
        "Fixed inverse Dower",
}

mech[
    "display"
] = mech[
    "condition"
].map(
    MECH_LABELS
).fillna(
    mech[
        "condition"
    ]
)

mech = (
    mech
    .sort_values(
        "AUROC_drop",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)


# ==========================================================================================
# LOAD/BUILD AGE RESULTS FOR D
# ==========================================================================================

def six_group_cache_is_valid(path):
    """
    Accept cached results only if they contain exactly
    6 age groups × 12 leads = 72 rows with the expected labels.
    """
    if not path.exists():
        return False

    try:
        cached = pd.read_csv(
            path,
            low_memory=False,
        )
    except Exception:
        return False

    required_cols = {
        "age_group",
        "lead",
        "N",
        "Female_N",
        "Male_N",
        "AUROC_drop",
        "AUROC_drop_CI_low",
        "AUROC_drop_CI_high",
    }

    if not required_cols.issubset(
        cached.columns
    ):
        return False

    observed_groups = set(
        cached[
            "age_group"
        ]
        .astype(str)
        .dropna()
        .unique()
        .tolist()
    )

    if observed_groups != set(
        AGE_LABELS
    ):
        return False

    if len(
        cached
    ) != (
        len(
            AGE_LABELS
        )
        *
        len(
            LEADS
        )
    ):
        return False

    for age_group in AGE_LABELS:
        n_unique_leads = (
            cached.loc[
                cached[
                    "age_group"
                ]
                ==
                age_group,
                "lead",
            ]
            .astype(str)
            .nunique()
        )

        if int(
            n_unique_leads
        ) != len(
            LEADS
        ):
            return False

    return True


cache_valid = six_group_cache_is_valid(
    AGE_CACHE
)

if (
    cache_valid
    and
    not FORCE_RECOMPUTE
):
    print(
        "=" * 120
    )

    print(
        "PULLING SAVED SIX-AGE-GROUP LEAD-ABLATION RESULTS"
    )

    print(
        "=" * 120
    )

    # Prefer Parquet when available, but always fall back to the canonical CSV.
    if AGE_CACHE_PARQUET.exists():
        try:
            age_lead = pd.read_parquet(
                AGE_CACHE_PARQUET
            )

            print(
                "Loaded saved results from Parquet:"
            )

            print(
                AGE_CACHE_PARQUET
            )

        except Exception:
            age_lead = pd.read_csv(
                AGE_CACHE,
                low_memory=False,
            )

            print(
                "Parquet could not be read; loaded saved CSV:"
            )

            print(
                AGE_CACHE
            )

    else:
        age_lead = pd.read_csv(
            AGE_CACHE,
            low_memory=False,
        )

        print(
            "Loaded saved CSV:"
        )

        print(
            AGE_CACHE
        )

    print(
        "No bootstrap recomputation was performed."
    )

else:
    if FORCE_RECOMPUTE:
        print(
            "AGELEAD_FORCE_RECOMPUTE=1 -> rebuilding six-group results."
        )

    elif AGE_CACHE.exists():
        print(
            "Existing six-group cache failed validation -> rebuilding."
        )

    else:
        print(
            "No saved six-group result cache found -> building results."
        )

    age_lead = build_age_grouped_results()


require_columns(
    age_lead,
    [
        "age_group",
        "lead",
        "N",
        "AUROC_drop",
        "AUROC_drop_CI_low",
        "AUROC_drop_CI_high",
    ],
    "Age-grouped lead ablation",
)

age_lead[
    "age_group"
] = age_lead[
    "age_group"
].astype(
    str
)

age_lead[
    "lead"
] = age_lead[
    "lead"
].astype(
    str
)

# Refresh a lightweight provenance manifest on every run so later users can
# determine whether the figure was generated from cached or recomputed results.
existing_metadata = {}

if AGE_METADATA_JSON.exists():
    try:
        existing_metadata = json.loads(
            AGE_METADATA_JSON.read_text()
        )
    except Exception:
        existing_metadata = {}

existing_metadata.update(
    {
        "last_used_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "last_run_mode":
            (
                "recomputed"
                if FORCE_RECOMPUTE
                or not cache_valid
                else
                "loaded_saved_results"
            ),

        "age_groups":
            AGE_LABELS,

        "n_boot":
            int(
                N_BOOT
            ),

        "bootstrap_seed":
            int(
                BOOT_SEED
            ),

        "summary_csv":
            str(
                AGE_CACHE
            ),

        "summary_parquet":
            str(
                AGE_CACHE_PARQUET
            )
            if AGE_CACHE_PARQUET.exists()
            else
            None,

        "cohort_audit_csv":
            str(
                AGE_COHORT_AUDIT
            )
            if AGE_COHORT_AUDIT.exists()
            else
            None,
    }
)

AGE_METADATA_JSON.write_text(
    json.dumps(
        existing_metadata,
        indent=2,
    )
)


# Strict final six-group audit.
observed_age_groups = set(
    age_lead[
        "age_group"
    ]
    .dropna()
    .unique()
    .tolist()
)

if observed_age_groups != set(
    AGE_LABELS
):
    raise RuntimeError(
        "Age-group mismatch in final lead-ablation table. "
        f"Expected {AGE_LABELS}; observed {sorted(observed_age_groups)}"
    )

expected_rows = (
    len(
        AGE_LABELS
    )
    *
    len(
        LEADS
    )
)

if len(
    age_lead
) != expected_rows:
    raise RuntimeError(
        f"Expected {expected_rows} lead × age rows; observed {len(age_lead)}."
    )

for age_group in AGE_LABELS:
    n_unique_leads = (
        age_lead.loc[
            age_lead[
                "age_group"
            ]
            ==
            age_group,
            "lead",
        ]
        .nunique()
    )

    if int(
        n_unique_leads
    ) != len(
        LEADS
    ):
        raise RuntimeError(
            f"{age_group}: expected {len(LEADS)} leads; observed {n_unique_leads}."
        )


# ==========================================================================================
# REGIONAL SUMMARY FOR B
# ==========================================================================================

regional_rows = []

for region in [
    "Precordial",
    "Limb",
]:
    vals = lead.loc[
        lead[
            "region"
        ]
        ==
        region,
        "AUROC_drop",
    ].to_numpy(
        dtype=float
    )

    regional_rows.append(
        {
            "region":
                region,

            "N_leads":
                len(
                    vals
                ),

            "mean_AUROC_drop":
                float(
                    np.mean(
                        vals
                    )
                ),

            "median_AUROC_drop":
                float(
                    np.median(
                        vals
                    )
                ),

            "Q1":
                float(
                    np.percentile(
                        vals,
                        25,
                    )
                ),

            "Q3":
                float(
                    np.percentile(
                        vals,
                        75,
                    )
                ),
        }
    )

regional = pd.DataFrame(
    regional_rows
)


# ==========================================================================================
# COLORS / STYLE
# ==========================================================================================

CMAP = plt.get_cmap(
    "plasma"
)

C_LIMB = CMAP(
    0.25
)

C_PRE = CMAP(
    0.76
)

AGE_COLORS = {
    "18–39":
        CMAP(
            0.10
        ),

    "40–49":
        CMAP(
            0.24
        ),

    "50–59":
        CMAP(
            0.40
        ),

    "60–69":
        CMAP(
            0.56
        ),

    "70–79":
        CMAP(
            0.72
        ),

    "≥80":
        CMAP(
            0.88
        ),
}

plt.rcParams.update(
    {
        "font.family":
            "DejaVu Sans",

        "font.size":
            10,

        "axes.titlesize":
            11.5,

        "axes.titleweight":
            "bold",

        "axes.labelsize":
            10.4,

        "axes.labelweight":
            "normal",

        "xtick.labelsize":
            8.9,

        "ytick.labelsize":
            8.9,

        "legend.fontsize":
            8.1,

        "axes.spines.top":
            False,

        "axes.spines.right":
            False,

        "figure.dpi":
            120,

        "savefig.dpi":
            600,
    }
)


# ==========================================================================================
# FIGURE
# ==========================================================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(
        15.6,
        11.4,
    ),
    gridspec_kw={
        "width_ratios": [
            1.00,
            1.22,
        ],
        "height_ratios": [
            0.96,
            1.14,
        ],
    },
)

axA, axB = axes[
    0
]

axC, axD = axes[
    1
]


# ==========================================================================================
# A — OVERALL LEAD ABLATION BAR + CI
# ==========================================================================================

plotA = lead.iloc[
    ::-1
].reset_index(
    drop=True
)

yA = np.arange(
    len(
        plotA
    )
)

valA = plotA[
    "AUROC_drop"
].to_numpy(
    dtype=float
)

loA = plotA[
    "AUROC_drop_CI_low"
].to_numpy(
    dtype=float
)

hiA = plotA[
    "AUROC_drop_CI_high"
].to_numpy(
    dtype=float
)

colorsA = [
    C_PRE
    if x
    ==
    "Precordial"
    else
    C_LIMB
    for x in plotA[
        "region"
    ]
]

axA.barh(
    yA,
    valA,
    color=colorsA,
    edgecolor="black",
    linewidth=0.45,
    zorder=2,
)

axA.errorbar(
    valA,
    yA,
    xerr=np.vstack(
        [
            valA
            -
            loA,

            hiA
            -
            valA,
        ]
    ),
    fmt="none",
    ecolor="black",
    elinewidth=1.0,
    capsize=2.8,
    zorder=3,
)

axA.axvline(
    0,
    color="0.6",
    linewidth=0.8,
)

axA.set_yticks(
    yA
)

axA.set_yticklabels(
    plotA[
        "lead"
    ]
)

axA.set_xlabel(
    "AUROC decrease after lead removal"
)

axA.set_title(
    "Lead importance by single-lead ablation",
    fontweight="bold",
    pad=8,
)

axA.grid(
    axis="x",
    alpha=0.14,
)

axA.set_xlim(
    min(
        -0.0005,
        float(
            loA.min()
        )
        -
        0.0003,
    ),
    float(
        hiA.max()
    )
    +
    0.0037,
)

for y, point, upper in zip(
    yA,
    valA,
    hiA,
):
    axA.text(
        upper + 0.00035,
        y,
        f"{point:.3f}",
        va="center",
        ha="left",
        fontsize=7.5,
    )


# ==========================================================================================
# B — REGION BOXPLOT
# ==========================================================================================

pre = lead.loc[
    lead[
        "region"
    ]
    ==
    "Precordial",
    "AUROC_drop",
].to_numpy(
    dtype=float
)

limb = lead.loc[
    lead[
        "region"
    ]
    ==
    "Limb",
    "AUROC_drop",
].to_numpy(
    dtype=float
)

bp = axB.boxplot(
    [
        pre,
        limb,
    ],
    tick_labels=[
        "Precordial",
        "Limb",
    ],
    patch_artist=True,
    widths=0.55,
    showfliers=False,
    medianprops=dict(
        color="black",
        linewidth=1.5,
    ),
    whiskerprops=dict(
        color="black",
        linewidth=1.0,
    ),
    capprops=dict(
        color="black",
        linewidth=1.0,
    ),
    boxprops=dict(
        color="black",
        linewidth=0.9,
    ),
)

bp[
    "boxes"
][
    0
].set_facecolor(
    C_PRE
)

bp[
    "boxes"
][
    1
].set_facecolor(
    C_LIMB
)

for box in bp[
    "boxes"
]:
    box.set_alpha(
        0.80
    )

rng = np.random.default_rng(
    42
)

for xpos, vals, color in [
    (
        1,
        pre,
        C_PRE,
    ),
    (
        2,
        limb,
        C_LIMB,
    ),
]:
    jitter = rng.uniform(
        -0.075,
        0.075,
        size=len(
            vals
        ),
    )

    axB.scatter(
        xpos
        +
        jitter,
        vals,
        s=46,
        color=color,
        edgecolor="black",
        linewidth=0.45,
        zorder=3,
    )

axB.set_ylabel(
    "AUROC decrease after lead removal"
)

axB.set_title(
    "Regional lead importance",
    fontweight="bold",
    pad=8,
)

axB.grid(
    axis="y",
    alpha=0.14,
)



# ==========================================================================================
# C — VCG MECHANISTIC ABLATION BAR + CI
# ==========================================================================================

plotC = mech.iloc[
    ::-1
].reset_index(
    drop=True
)

yC = np.arange(
    len(
        plotC
    )
)

valC = plotC[
    "AUROC_drop"
].to_numpy(
    dtype=float
)

loC = plotC[
    "AUROC_drop_CI_low"
].to_numpy(
    dtype=float
)

hiC = plotC[
    "AUROC_drop_CI_high"
].to_numpy(
    dtype=float
)

colorsC = [
    CMAP(
        x
    )
    for x in np.linspace(
        0.32,
        0.86,
        len(
            plotC
        ),
    )
]

axC.barh(
    yC,
    valC,
    color=colorsC,
    edgecolor="black",
    linewidth=0.45,
    zorder=2,
)

axC.errorbar(
    valC,
    yC,
    xerr=np.vstack(
        [
            valC
            -
            loC,

            hiC
            -
            valC,
        ]
    ),
    fmt="none",
    ecolor="black",
    elinewidth=1.0,
    capsize=2.8,
    zorder=3,
)

axC.axvline(
    0,
    color="0.6",
    linewidth=0.8,
)

axC.set_yticks(
    yC
)

axC.set_yticklabels(
    plotC[
        "display"
    ]
)

axC.set_xlabel(
    "AUROC decrease after component ablation"
)

axC.set_title(
    "VCG component ablation",
    fontweight="bold",
    pad=8,
)

axC.grid(
    axis="x",
    alpha=0.14,
)

axC.set_xlim(
    min(
        -0.0005,
        float(
            loC.min()
        )
        -
        0.00025,
    ),
    float(
        hiC.max()
    )
    +
    0.0030,
)

for y, point, upper in zip(
    yC,
    valC,
    hiC,
):
    axC.text(
        upper + 0.00018,
        y,
        (
            f"{point:.5f}"
            if abs(
                point
            )
            <
            0.001
            else
            f"{point:.4f}"
        ),
        va="center",
        ha="left",
        fontsize=7.4,
    )


# ==========================================================================================
# D — AGE-STRATIFIED LEAD ABLATION FOREST
# ==========================================================================================

# Same visual top-to-bottom order as Panel A.
D_LEADS = OVERALL_LEAD_ORDER[
    ::-1
]

base_y = {
    lead_name:
        i
    for i, lead_name in enumerate(
        D_LEADS
    )
}

offsets = {
    "18–39":
        -0.35,

    "40–49":
        -0.21,

    "50–59":
        -0.07,

    "60–69":
        +0.07,

    "70–79":
        +0.21,

    "≥80":
        +0.35,
}

for age_group in AGE_LABELS:
    sub = age_lead.loc[
        age_lead[
            "age_group"
        ]
        ==
        age_group
    ].copy()

    sub = sub.loc[
        sub[
            "lead"
        ].isin(
            D_LEADS
        )
    ].copy()

    sub[
        "y"
    ] = (
        sub[
            "lead"
        ].map(
            base_y
        ).astype(
            float
        )
        +
        offsets[
            age_group
        ]
    )

    sub = sub.dropna(
        subset=[
            "AUROC_drop",
            "AUROC_drop_CI_low",
            "AUROC_drop_CI_high",
        ]
    )

    x = sub[
        "AUROC_drop"
    ].to_numpy(
        dtype=float
    )

    low = sub[
        "AUROC_drop_CI_low"
    ].to_numpy(
        dtype=float
    )

    high = sub[
        "AUROC_drop_CI_high"
    ].to_numpy(
        dtype=float
    )

    y = sub[
        "y"
    ].to_numpy(
        dtype=float
    )

    axD.errorbar(
        x,
        y,
        xerr=np.vstack(
            [
                x
                -
                low,

                high
                -
                x,
            ]
        ),
        fmt="o",
        markersize=4.5,
        linewidth=0.95,
        capsize=1.9,
        color=AGE_COLORS[
            age_group
        ],
        ecolor=AGE_COLORS[
            age_group
        ],
        markeredgecolor="black",
        markeredgewidth=0.35,
        label=age_group,
        zorder=3,
    )


axD.axvline(
    0,
    color="0.6",
    linewidth=0.8,
)

axD.set_yticks(
    np.arange(
        len(
            D_LEADS
        )
    )
)

axD.set_yticklabels(
    D_LEADS,
    fontsize=8.5,
)

axD.set_xlabel(
    "AUROC decrease after lead removal"
)

axD.set_title(
    "Age-stratified lead importance",
    fontweight="bold",
    pad=8,
)

axD.grid(
    axis="x",
    alpha=0.14,
)

# Extra horizontal breathing room for the six-group forest plot.
d_lo = float(
    age_lead[
        "AUROC_drop_CI_low"
    ].min()
)

d_hi = float(
    age_lead[
        "AUROC_drop_CI_high"
    ].max()
)

d_span = max(
    d_hi - d_lo,
    0.01,
)

axD.set_xlim(
    min(
        -0.005,
        d_lo - 0.03 * d_span,
    ),
    d_hi + 0.10 * d_span,
)

legendD = axD.legend(
    title="Age, years",
    frameon=False,
    loc="lower right",
    ncol=3,
    handletextpad=0.4,
    columnspacing=0.9,
)

legendD.get_title().set_fontweight("bold")

axD.text(
    0.02,
    0.985,
    "Paired bootstrap within each of six age strata",
    transform=axD.transAxes,
    ha="left",
    va="top",
    fontsize=7.2,
    fontweight="normal",
)


# ==========================================================================================
# PANEL LABELS
# ==========================================================================================

for label, ax in zip(
    [
        "A",
        "B",
        "C",
        "D",
    ],
    [
        axA,
        axB,
        axC,
        axD,
    ],
):
    ax.text(
        -0.12,
        1.055,
        label,
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        ha="left",
        va="top",
    )


# ==========================================================================================
# SOURCE DATA
# ==========================================================================================

def block(
    df,
    panel,
    source,
):
    out = df.copy()

    out.insert(
        0,
        "panel",
        panel,
    )

    out.insert(
        1,
        "source",
        source,
    )

    return out


pd.concat(
    [
        block(
            lead,
            "A-B",
            "Final WCMC single-lead ablation",
        ),

        block(
            regional,
            "B",
            "Descriptive precordial-vs-limb summary",
        ),

        block(
            mech,
            "C",
            "Final VCG mechanistic ablation",
        ),

        block(
            age_lead,
            "D",
            "Age-stratified paired-bootstrap single-lead ablation",
        ),
    ],
    ignore_index=True,
    sort=False,
).to_csv(
    SOURCE_OUT,
    index=False,
)


# ==========================================================================================
# SAVE
# ==========================================================================================

fig.subplots_adjust(
    left=0.14,
    right=0.988,
    bottom=0.08,
    top=0.965,
    wspace=0.30,
    hspace=0.32,
)

fig.savefig(
    PNG_OUT,
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
)

fig.savefig(
    PDF_OUT,
    bbox_inches="tight",
    facecolor="white",
)

fig.savefig(
    SVG_OUT,
    bbox_inches="tight",
    facecolor="white",
)

print(
    "\nDisplaying Figure 5..."
)

plt.show()


# ==========================================================================================
# SUMMARY
# ==========================================================================================

print(
    "\n"
    +
    "=" * 120
)

print(
    "FIGURE 5 COMPLETE"
)

print(
    "=" * 120
)

print(
    "\nPanel A — overall lead importance"
)

print(
    lead[
        [
            "lead",
            "region",
            "AUROC_drop",
            "AUROC_drop_CI_low",
            "AUROC_drop_CI_high",
        ]
    ].to_string(
        index=False
    )
)

print(
    "\nPanel B — regional summary"
)

print(
    regional.to_string(
        index=False
    )
)

print(
    "\nPanel C — VCG mechanistic ablation"
)

print(
    mech[
        [
            "display",
            "AUROC_drop",
            "AUROC_drop_CI_low",
            "AUROC_drop_CI_high",
        ]
    ].to_string(
        index=False
    )
)

print(
    "\nPanel D — age-stratified lead importance"
)

print(
    "Age groups: 18–39 | 40–49 | 50–59 | 60–69 | 70–79 | ≥80"
)

print(
    age_lead[
        [
            "age_group",
            "lead",
            "N",
            "Female_N",
            "Male_N",
            "AUROC_drop",
            "AUROC_drop_CI_low",
            "AUROC_drop_CI_high",
        ]
    ].to_string(
        index=False
    )
)

print(
    "\nSaved:"
)

print(
    PNG_OUT
)

print(
    PDF_OUT
)

print(
    SVG_OUT
)

print(
    SOURCE_OUT
)

print(
    "\nPersistent six-group result files:"
)

print(
    "Summary CSV:"
)

print(
    AGE_CACHE
)

print(
    "Summary Parquet (if available):"
)

print(
    AGE_CACHE_PARQUET
)

print(
    "Cohort audit:"
)

print(
    AGE_COHORT_AUDIT
)

print(
    "Metadata/provenance:"
)

print(
    AGE_METADATA_JSON
)

print(
    "\nTo reuse later: run normally; saved six-group results are loaded automatically."
)

print(
    "To intentionally recompute: AGELEAD_FORCE_RECOMPUTE=1 python "
    "ECGSex_Figure5_FINAL_VCG_Interpretability_6AgeGroups_Persistent_Spaced.py"
)

print(
    "\nDONE"
)
