#!/usr/bin/env python
# ==========================================================================================
# ECG-SEX — AGE-STRATIFIED PERFORMANCE
# FINAL CURRENT RUN: VCG-GEOMETRY RESNET vs MATCHED RESNET
# WCMC/NYP LOCKED TEST | 2 × 2 | PLASMA
# ==========================================================================================
#
# Panels
# ------
# A. AUROC by age group
# B. AUPRC by age group
# C. F1 by age group
# D. Brier score by age group
#
# Age groups
# ----------
# 18–39, 40–59, 60–79, >=80 years
#
# IMPORTANT
# ---------
# - Uses the exact FINAL combined locked-test prediction file:
#     WCMC_LockedTest_Aligned_5SeedEnsemble_Predictions.csv
# - VCG probability column:    prob_vcg
# - ResNet probability column: prob_resnet
# - Ground truth:              sex_true_binary
# - Frozen validation thresholds:
#       VCG    = 0.484619
#       ResNet = 0.453711
# - Age is taken from the exact locked WCMC test split.
# - 2,000 paired patient-level bootstrap replicates within each age stratum.
# - No model inference or retraining.
# ==========================================================================================

from pathlib import Path
import os
import json
import hashlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    brier_score_loss,
)

from joblib import Parallel, delayed


# ==========================================================================================
# CONFIG
# ==========================================================================================

N_BOOT = int(
    os.environ.get(
        "AGEPERF_N_BOOT",
        "2000",
    )
)

N_JOBS = int(
    os.environ.get(
        "AGEPERF_N_JOBS",
        "8",
    )
)

BASE_SEED = 20260812

VCG_THRESHOLD = 0.484619
RESNET_THRESHOLD = 0.453711

AGE_BINS = [
    18,
    40,
    60,
    80,
    np.inf,
]

AGE_LABELS = [
    "18–39",
    "40–59",
    "60–79",
    "≥80",
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

FINAL_WCMC_DIR = (
    WCMC_ROOT
    /
    "19_vcg_geometry_resnet_5seed_locked_wcmc"
    /
    "final_5seed_ensemble_bootstrap"
)

PRED_CSV = (
    FINAL_WCMC_DIR
    /
    "WCMC_LockedTest_Aligned_5SeedEnsemble_Predictions.csv"
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
    "Figure_AgeStratified_Performance_FINAL_VCG_vs_ResNet_2x2_Plasma"
)

PNG_OUT = Path(str(STEM) + ".png")
PDF_OUT = Path(str(STEM) + ".pdf")
SVG_OUT = Path(str(STEM) + ".svg")
SOURCE_OUT = Path(str(STEM) + "_SourceData.csv")
DIFF_OUT = Path(str(STEM) + "_PairedDifferences.csv")
AUDIT_OUT = Path(str(STEM) + "_CohortAudit.csv")
BOOT_OUT = Path(str(STEM) + "_BootstrapDistributions.csv")
META_OUT = Path(str(STEM) + "_Metadata.json")


# ==========================================================================================
# FILE AUDIT
# ==========================================================================================

for p in [
    PRED_CSV,
    TEST_CSV,
]:
    if not p.exists():
        raise FileNotFoundError(
            f"Missing required current-run file:\n{p}"
        )


# ==========================================================================================
# HELPERS
# ==========================================================================================

def require_columns(
    df,
    cols,
    label,
):
    missing = [
        c
        for c in cols
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"{label} missing columns: {missing}\n"
            f"Observed columns: {df.columns.tolist()}"
        )


def stable_seed(
    *parts,
):
    token = "||".join(
        str(x)
        for x in parts
    )

    h = hashlib.sha256(
        token.encode(
            "utf-8"
        )
    ).hexdigest()

    return (
        BASE_SEED
        +
        int(
            h[
                :8
            ],
            16,
        )
    ) % (
        2**32
        -
        1
    )


def classification_metrics(
    y,
    p,
    threshold,
):
    y = np.asarray(
        y,
        dtype=int,
    )

    p = np.asarray(
        p,
        dtype=float,
    )

    yhat = (
        p
        >=
        threshold
    ).astype(
        int
    )

    tn, fp, fn, tp = confusion_matrix(
        y,
        yhat,
        labels=[
            0,
            1,
        ],
    ).ravel()

    specificity = (
        tn
        /
        (
            tn
            +
            fp
        )
        if (
            tn
            +
            fp
        )
        >
        0
        else
        np.nan
    )

    return {
        "AUROC":
            float(
                roc_auc_score(
                    y,
                    p,
                )
            ),

        "AUPRC":
            float(
                average_precision_score(
                    y,
                    p,
                )
            ),

        "F1":
            float(
                f1_score(
                    y,
                    yhat,
                    zero_division=0,
                )
            ),

        "Accuracy":
            float(
                accuracy_score(
                    y,
                    yhat,
                )
            ),

        "Balanced_Accuracy":
            float(
                balanced_accuracy_score(
                    y,
                    yhat,
                )
            ),

        "Precision":
            float(
                precision_score(
                    y,
                    yhat,
                    zero_division=0,
                )
            ),

        "Sensitivity":
            float(
                recall_score(
                    y,
                    yhat,
                    zero_division=0,
                )
            ),

        "Specificity":
            float(
                specificity
            ),

        "Brier":
            float(
                brier_score_loss(
                    y,
                    p,
                )
            ),
    }


def bootstrap_one_replicate(
    seed,
    y,
    p_vcg,
    p_resnet,
):
    rng = np.random.default_rng(
        seed
    )

    n = len(
        y
    )

    # Resample rows; the WCMC locked test contains one ECG per patient.
    idx = rng.integers(
        0,
        n,
        size=n,
    )

    yb = y[
        idx
    ]

    # AUROC/AUPRC require both classes.
    if np.unique(
        yb
    ).size < 2:
        return None

    vb = p_vcg[
        idx
    ]

    rb = p_resnet[
        idx
    ]

    vm = classification_metrics(
        yb,
        vb,
        VCG_THRESHOLD,
    )

    rm = classification_metrics(
        yb,
        rb,
        RESNET_THRESHOLD,
    )

    return {
        "VCG":
            vm,

        "ResNet":
            rm,
    }


def bootstrap_age_group(
    age_group,
    df,
):
    y = df[
        "sex_true_binary"
    ].to_numpy(
        dtype=int
    )

    p_vcg = df[
        "prob_vcg"
    ].to_numpy(
        dtype=float
    )

    p_resnet = df[
        "prob_resnet"
    ].to_numpy(
        dtype=float
    )

    n = len(
        df
    )

    if n < 30:
        raise RuntimeError(
            f"Age group {age_group} has only N={n}"
        )

    if np.unique(
        y
    ).size < 2:
        raise RuntimeError(
            f"Age group {age_group} contains only one sex class."
        )

    point_vcg = classification_metrics(
        y,
        p_vcg,
        VCG_THRESHOLD,
    )

    point_resnet = classification_metrics(
        y,
        p_resnet,
        RESNET_THRESHOLD,
    )

    seeds = [
        stable_seed(
            age_group,
            i,
        )
        for i in range(
            N_BOOT
        )
    ]

    reps = Parallel(
        n_jobs=N_JOBS,
        backend="loky",
        verbose=3,
    )(
        delayed(
            bootstrap_one_replicate
        )(
            seed,
            y,
            p_vcg,
            p_resnet,
        )
        for seed in seeds
    )

    reps = [
        x
        for x in reps
        if x is not None
    ]

    if len(
        reps
    ) < int(
        0.95
        *
        N_BOOT
    ):
        raise RuntimeError(
            f"Too few valid bootstrap replicates for {age_group}: "
            f"{len(reps)}/{N_BOOT}"
        )

    metrics = list(
        point_vcg.keys()
    )

    summary_rows = []
    diff_rows = []
    boot_rows = []

    for model_name, point, threshold in [
        (
            "VCG-Geometry ResNet",
            point_vcg,
            VCG_THRESHOLD,
        ),
        (
            "ResNet",
            point_resnet,
            RESNET_THRESHOLD,
        ),
    ]:
        model_key = (
            "VCG"
            if model_name
            ==
            "VCG-Geometry ResNet"
            else
            "ResNet"
        )

        for metric in metrics:
            arr = np.asarray(
                [
                    r[
                        model_key
                    ][
                        metric
                    ]
                    for r in reps
                ],
                dtype=float,
            )

            if metric == metrics[
                0
            ]:
                pass

            # bootstrap values saved below

        row = {
            "age_group":
                age_group,

            "model":
                model_name,

            "N":
                n,

            "Female_N":
                int(
                    np.sum(
                        y
                        ==
                        0
                    )
                ),

            "Male_N":
                int(
                    np.sum(
                        y
                        ==
                        1
                    )
                ),

            "threshold":
                threshold,

            "bootstrap_replicates":
                len(
                    reps
                ),
        }

        for metric in metrics:
            arr = np.asarray(
                [
                    r[
                        model_key
                    ][
                        metric
                    ]
                    for r in reps
                ],
                dtype=float,
            )

            row[
                metric
            ] = point[
                metric
            ]

            row[
                f"{metric}_CI_low"
            ] = float(
                np.percentile(
                    arr,
                    2.5,
                )
            )

            row[
                f"{metric}_CI_high"
            ] = float(
                np.percentile(
                    arr,
                    97.5,
                )
            )

        summary_rows.append(
            row
        )

    # paired differences: VCG - ResNet
    for metric in metrics:
        arr_v = np.asarray(
            [
                r[
                    "VCG"
                ][
                    metric
                ]
                for r in reps
            ],
            dtype=float,
        )

        arr_r = np.asarray(
            [
                r[
                    "ResNet"
                ][
                    metric
                ]
                for r in reps
            ],
            dtype=float,
        )

        diff = (
            arr_v
            -
            arr_r
        )

        point_diff = (
            point_vcg[
                metric
            ]
            -
            point_resnet[
                metric
            ]
        )

        # two-sided paired-bootstrap tail probability
        p_two = min(
            1.0,
            2.0
            *
            min(
                float(
                    np.mean(
                        diff
                        <=
                        0
                    )
                ),
                float(
                    np.mean(
                        diff
                        >=
                        0
                    )
                ),
            ),
        )

        # finite-resolution correction consistent with 2,000 replicates
        p_two = max(
            p_two,
            1.0
            /
            (
                len(
                    diff
                )
                +
                1
            ),
        )

        diff_rows.append(
            {
                "age_group":
                    age_group,

                "metric":
                    metric,

                "N":
                    n,

                "Female_N":
                    int(
                        np.sum(
                            y
                            ==
                            0
                        )
                    ),

                "Male_N":
                    int(
                        np.sum(
                            y
                            ==
                            1
                        )
                    ),

                "delta_VCG_minus_ResNet":
                    point_diff,

                "CI_low":
                    float(
                        np.percentile(
                            diff,
                            2.5,
                        )
                    ),

                "CI_high":
                    float(
                        np.percentile(
                            diff,
                            97.5,
                        )
                    ),

                "P_two_sided":
                    p_two,

                "bootstrap_replicates":
                    len(
                        diff
                    ),
            }
        )

    # Long bootstrap table for reproducibility / optional violin plots.
    for i, r in enumerate(
        reps
    ):
        for metric in metrics:
            boot_rows.append(
                {
                    "age_group":
                        age_group,

                    "bootstrap_id":
                        i
                        +
                        1,

                    "metric":
                        metric,

                    "VCG":
                        r[
                            "VCG"
                        ][
                            metric
                        ],

                    "ResNet":
                        r[
                            "ResNet"
                        ][
                            metric
                        ],

                    "delta_VCG_minus_ResNet":
                        (
                            r[
                                "VCG"
                            ][
                                metric
                            ]
                            -
                            r[
                                "ResNet"
                            ][
                                metric
                            ]
                        ),
                }
            )

    return (
        summary_rows,
        diff_rows,
        boot_rows,
    )


# ==========================================================================================
# LOAD FINAL COMBINED PREDICTIONS
# ==========================================================================================

pred = pd.read_csv(
    PRED_CSV,
    low_memory=False,
)

require_columns(
    pred,
    [
        "sex_true_binary",
        "prob_vcg",
        "prob_resnet",
    ],
    "Final locked-test prediction file",
)

pred[
    "sex_true_binary"
] = pd.to_numeric(
    pred[
        "sex_true_binary"
    ],
    errors="raise",
).astype(
    int
)

pred[
    "prob_vcg"
] = pd.to_numeric(
    pred[
        "prob_vcg"
    ],
    errors="raise",
)

pred[
    "prob_resnet"
] = pd.to_numeric(
    pred[
        "prob_resnet"
    ],
    errors="raise",
)

if len(
    pred
) != 21374:
    raise RuntimeError(
        f"Expected locked WCMC test N=21,374; found {len(pred):,}"
    )


# ==========================================================================================
# LOAD AGE FROM EXACT LOCKED TEST SPLIT
# ==========================================================================================

test = pd.read_csv(
    TEST_CSV,
    low_memory=False,
)

require_columns(
    test,
    [
        "patient_key",
        "ecg_key",
        "age_at_ecg",
        "sex_binary",
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

test[
    "age_at_ecg"
] = pd.to_numeric(
    test[
        "age_at_ecg"
    ],
    errors="raise",
)

test[
    "sex_binary"
] = pd.to_numeric(
    test[
        "sex_binary"
    ],
    errors="raise",
).astype(
    int
)

if len(
    test
) != 21374:
    raise RuntimeError(
        f"Expected locked WCMC test split N=21,374; found {len(test):,}"
    )

if test[
    "patient_key"
].duplicated().any():
    raise RuntimeError(
        "Locked test split contains duplicate patient_key values."
    )

if test[
    "ecg_key"
].duplicated().any():
    raise RuntimeError(
        "Locked test split contains duplicate ecg_key values."
    )


# ==========================================================================================
# ALIGN PREDICTIONS TO TEST SPLIT
# ==========================================================================================

pred_has_keys = (
    "patient_key"
    in pred.columns
    and
    "ecg_key"
    in pred.columns
)

if pred_has_keys:
    print(
        "Alignment mode: patient_key + ecg_key"
    )

    pred[
        "patient_key"
    ] = pred[
        "patient_key"
    ].astype(
        str
    )

    pred[
        "ecg_key"
    ] = pred[
        "ecg_key"
    ].astype(
        str
    )

    merged = test[
        [
            "patient_key",
            "ecg_key",
            "age_at_ecg",
            "sex_binary",
        ]
    ].merge(
        pred[
            [
                "patient_key",
                "ecg_key",
                "sex_true_binary",
                "prob_vcg",
                "prob_resnet",
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
        merged
    ) != 21374:
        raise RuntimeError(
            f"Key alignment produced N={len(merged):,}; expected 21,374."
        )

else:
    print(
        "Prediction file has no patient/ecg keys."
    )

    print(
        "Alignment mode: locked-row order with strict label audit"
    )

    # The final combined file was generated from the same locked-test ordering.
    if not np.array_equal(
        test[
            "sex_binary"
        ].to_numpy(
            dtype=int
        ),
        pred[
            "sex_true_binary"
        ].to_numpy(
            dtype=int
        ),
    ):
        raise RuntimeError(
            "Cannot safely use row-order alignment: test labels and prediction labels differ."
        )

    merged = test[
        [
            "patient_key",
            "ecg_key",
            "age_at_ecg",
            "sex_binary",
        ]
    ].copy()

    merged[
        "sex_true_binary"
    ] = pred[
        "sex_true_binary"
    ].to_numpy(
        dtype=int
    )

    merged[
        "prob_vcg"
    ] = pred[
        "prob_vcg"
    ].to_numpy(
        dtype=float
    )

    merged[
        "prob_resnet"
    ] = pred[
        "prob_resnet"
    ].to_numpy(
        dtype=float
    )


# Ground-truth consistency.
if not np.array_equal(
    merged[
        "sex_binary"
    ].to_numpy(
        dtype=int
    ),
    merged[
        "sex_true_binary"
    ].to_numpy(
        dtype=int
    ),
):
    raise RuntimeError(
        "Locked-test sex labels do not match final prediction labels."
    )


# ==========================================================================================
# AGE GROUPS
# ==========================================================================================

merged[
    "age_group"
] = pd.cut(
    merged[
        "age_at_ecg"
    ],
    bins=AGE_BINS,
    labels=AGE_LABELS,
    right=False,
    include_lowest=True,
)

merged = merged.dropna(
    subset=[
        "age_group"
    ]
).copy()

merged[
    "age_group"
] = merged[
    "age_group"
].astype(
    str
)

if len(
    merged
) != 21374:
    raise RuntimeError(
        f"Age stratification lost observations: retained {len(merged):,}/21,374"
    )


# ==========================================================================================
# COHORT AUDIT
# ==========================================================================================

audit = (
    merged
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
            "age_at_ecg",
            "mean",
        ),

        Age_SD=(
            "age_at_ecg",
            "std",
        ),

        Age_min=(
            "age_at_ecg",
            "min",
        ),

        Age_max=(
            "age_at_ecg",
            "max",
        ),
    )
    .reset_index()
)

audit[
    "age_group"
] = pd.Categorical(
    audit[
        "age_group"
    ],
    categories=AGE_LABELS,
    ordered=True,
)

audit = audit.sort_values(
    "age_group"
).reset_index(
    drop=True
)

audit.to_csv(
    AUDIT_OUT,
    index=False,
)

print(
    "=" * 120
)

print(
    "AGE-STRATIFIED FINAL WCMC PERFORMANCE"
)

print(
    "=" * 120
)

print(
    "\nPrediction file:"
)

print(
    PRED_CSV
)

print(
    "\nLocked test split:"
)

print(
    TEST_CSV
)

print(
    "\nFrozen thresholds:"
)

print(
    f"VCG    = {VCG_THRESHOLD:.6f}"
)

print(
    f"ResNet = {RESNET_THRESHOLD:.6f}"
)

print(
    "\nAge-group cohort audit:"
)

print(
    audit.to_string(
        index=False
    )
)


# ==========================================================================================
# BOOTSTRAP AGE GROUPS
# ==========================================================================================

all_summary = []
all_diff = []
all_boot = []

for age_group in AGE_LABELS:
    sub = merged.loc[
        merged[
            "age_group"
        ]
        ==
        age_group
    ].copy()

    print(
        f"\nBootstrap age group {age_group}: N={len(sub):,}"
    )

    summary_rows, diff_rows, boot_rows = bootstrap_age_group(
        age_group,
        sub,
    )

    all_summary.extend(
        summary_rows
    )

    all_diff.extend(
        diff_rows
    )

    all_boot.extend(
        boot_rows
    )


summary = pd.DataFrame(
    all_summary
)

diff = pd.DataFrame(
    all_diff
)

boot = pd.DataFrame(
    all_boot
)

summary[
    "age_group"
] = pd.Categorical(
    summary[
        "age_group"
    ],
    categories=AGE_LABELS,
    ordered=True,
)

diff[
    "age_group"
] = pd.Categorical(
    diff[
        "age_group"
    ],
    categories=AGE_LABELS,
    ordered=True,
)

summary = summary.sort_values(
    [
        "age_group",
        "model",
    ]
).reset_index(
    drop=True
)

diff = diff.sort_values(
    [
        "age_group",
        "metric",
    ]
).reset_index(
    drop=True
)

summary.to_csv(
    SOURCE_OUT,
    index=False,
)

diff.to_csv(
    DIFF_OUT,
    index=False,
)

boot.to_csv(
    BOOT_OUT,
    index=False,
)


# ==========================================================================================
# CONSOLE TABLES
# ==========================================================================================

print(
    "\nPerformance:"
)

print(
    summary[
        [
            "age_group",
            "model",
            "N",
            "AUROC",
            "AUROC_CI_low",
            "AUROC_CI_high",
            "AUPRC",
            "AUPRC_CI_low",
            "AUPRC_CI_high",
            "F1",
            "F1_CI_low",
            "F1_CI_high",
            "Brier",
            "Brier_CI_low",
            "Brier_CI_high",
        ]
    ].to_string(
        index=False
    )
)

print(
    "\nPaired VCG − ResNet differences:"
)

print(
    diff.loc[
        diff[
            "metric"
        ].isin(
            [
                "AUROC",
                "AUPRC",
                "F1",
                "Brier",
            ]
        ),
        [
            "age_group",
            "metric",
            "delta_VCG_minus_ResNet",
            "CI_low",
            "CI_high",
            "P_two_sided",
        ],
    ].to_string(
        index=False
    )
)


# ==========================================================================================
# FIGURE
# ==========================================================================================

CMAP = plt.get_cmap(
    "plasma"
)

COLORS = {
    "ResNet":
        CMAP(
            0.32
        ),

    "VCG-Geometry ResNet":
        CMAP(
            0.82
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

        "axes.labelsize":
            10.4,

        "xtick.labelsize":
            9.0,

        "ytick.labelsize":
            9.0,

        "legend.fontsize":
            8.6,

        "axes.spines.top":
            False,

        "axes.spines.right":
            False,

        "figure.dpi":
            120,
    }
)

fig, axes = plt.subplots(
    2,
    2,
    figsize=(
        13.4,
        9.4,
    ),
)

axA, axB = axes[
    0
]

axC, axD = axes[
    1
]

panels = [
    (
        axA,
        "AUROC",
        "AUROC",
        "Age-stratified discrimination",
    ),

    (
        axB,
        "AUPRC",
        "AUPRC",
        "Age-stratified precision–recall",
    ),

    (
        axC,
        "F1",
        "F1 score",
        "Age-stratified F1",
    ),

    (
        axD,
        "Brier",
        "Brier score",
        "Age-stratified Brier score",
    ),
]

x = np.arange(
    len(
        AGE_LABELS
    )
)

model_offsets = {
    "ResNet":
        -0.10,

    "VCG-Geometry ResNet":
        +0.10,
}

for ax, metric, ylabel, title in panels:
    values_seen = []

    for model in [
        "ResNet",
        "VCG-Geometry ResNet",
    ]:
        sub = summary.loc[
            summary[
                "model"
            ]
            ==
            model
        ].copy()

        sub[
            "age_group"
        ] = pd.Categorical(
            sub[
                "age_group"
            ],
            categories=AGE_LABELS,
            ordered=True,
        )

        sub = sub.sort_values(
            "age_group"
        )

        vals = sub[
            metric
        ].to_numpy(
            dtype=float
        )

        low = sub[
            f"{metric}_CI_low"
        ].to_numpy(
            dtype=float
        )

        high = sub[
            f"{metric}_CI_high"
        ].to_numpy(
            dtype=float
        )

        xpos = (
            x
            +
            model_offsets[
                model
            ]
        )

        ax.errorbar(
            xpos,
            vals,
            yerr=np.vstack(
                [
                    vals
                    -
                    low,

                    high
                    -
                    vals,
                ]
            ),
            fmt="o-",
            markersize=6.0,
            linewidth=1.8,
            capsize=3.0,
            color=COLORS[
                model
            ],
            markeredgecolor="black",
            markeredgewidth=0.45,
            label=model,
            zorder=3,
        )

        values_seen.extend(
            list(
                low
            )
        )

        values_seen.extend(
            list(
                high
            )
        )

        for xx, yy in zip(
            xpos,
            vals,
        ):
            ax.text(
                xx,
                yy,
                f" {yy:.3f}",
                fontsize=7.3,
                va="bottom",
                ha="left",
            )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        AGE_LABELS
    )

    ax.set_xlabel(
        "Age, years"
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_title(
        title
    )

    ax.grid(
        axis="y",
        alpha=0.14,
    )

    ymin = float(
        np.min(
            values_seen
        )
    )

    ymax = float(
        np.max(
            values_seen
        )
    )

    span = max(
        ymax
        -
        ymin,
        0.01,
    )

    # Brier lower is better, but otherwise no special axis transformation.
    ax.set_ylim(
        max(
            0,
            ymin
            -
            0.18
            *
            span,
        ),
        ymax
        +
        0.23
        *
        span,
    )


# Shared legend.
handles, labels = axA.get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    bbox_to_anchor=(
        0.5,
        0.012,
    ),
    ncol=2,
    frameon=False,
)


# Add cohort N under age labels to Panel A only.
n_map = {
    str(
        row[
            "age_group"
        ]
    ):
        int(
            row[
                "N"
            ]
        )
    for _, row in audit.iterrows()
}

for i, age_group in enumerate(
    AGE_LABELS
):
    axA.text(
        i,
        -0.145,
        f"N={n_map[age_group]:,}",
        transform=axA.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=7.7,
    )


# Panel labels.
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
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        ha="left",
        va="top",
    )


fig.subplots_adjust(
    left=0.09,
    right=0.985,
    top=0.965,
    bottom=0.11,
    wspace=0.26,
    hspace=0.34,
)


# ==========================================================================================
# SAVE
# ==========================================================================================

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

plt.show()


# ==========================================================================================
# METADATA
# ==========================================================================================

meta = {
    "prediction_file":
        str(
            PRED_CSV
        ),

    "test_split":
        str(
            TEST_CSV
        ),

    "locked_test_N":
        int(
            len(
                merged
            )
        ),

    "VCG_threshold":
        VCG_THRESHOLD,

    "ResNet_threshold":
        RESNET_THRESHOLD,

    "bootstrap_replicates":
        N_BOOT,

    "age_groups":
        AGE_LABELS,

    "age_source_column":
        "age_at_ecg",

    "sex_coding":
        "female=0, male=1",
}

with open(
    META_OUT,
    "w",
) as f:
    json.dump(
        meta,
        f,
        indent=2,
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
    DIFF_OUT
)

print(
    AUDIT_OUT
)

print(
    BOOT_OUT
)

print(
    META_OUT
)

print(
    "\nDONE"
)
