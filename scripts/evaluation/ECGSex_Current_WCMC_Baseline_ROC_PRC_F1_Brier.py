#!/usr/bin/env python
# ==========================================================================================
# ECG-SEX — CURRENT FINAL WCMC BASELINE COMPARISON
# A ROC | B PRC | C F1 BAR | D BRIER BAR
# ==========================================================================================
#
# CURRENT WCMC/NYP LOCKED TEST
#
# A. ROC curves
# B. Precision–Recall curves
# C. F1 score — bar plot
# D. Brier score — bar plot
#
# Curve logic
# -----------
# Exact current-run prediction sources are used first:
#   - Final VCG-Geometry ResNet 5-seed ensemble
#   - Matched ResNet1D 5-seed ensemble
#   - CNN–TCN–Transformer MIMIC→WCMC
#   - CNN–TCN–Transformer WCMC-only
#
# The script then searches CURRENT WCMC output directories for aligned prediction CSVs for:
#   - ResNet1D WCMC-only
#   - CNN–LSTM WCMC-only
#   - CNN WCMC-only
#   - Transformer WCMC-only
#
# Optional curve files are accepted ONLY when:
#   1. N == 21,374
#   2. sex_true_binary exists
#   3. a probability column exists
#   4. ground-truth labels exactly match the locked-test reference ordering
#
# If an optional curve prediction is not found, the figure DOES NOT fail:
# the model remains in C/D using the completed locked-test point estimate and is reported
# in the console as "bar-only".
#
# C/D values are completed current locked-test results; no model inference/training is run.
# ==========================================================================================

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)

warnings.filterwarnings("ignore")


# ==========================================================================================
# PATHS
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

FINAL_5SEED_CSV = (
    WCMC_ROOT
    /
    "19_vcg_geometry_resnet_5seed_locked_wcmc"
    /
    "final_5seed_ensemble_bootstrap"
    /
    "WCMC_LockedTest_Aligned_5SeedEnsemble_Predictions.csv"
)

TRISM_TRANSFER_CSV = (
    WCMC_ROOT
    /
    "04_wcmc_finetune_from_mimic"
    /
    "predictions"
    /
    "WCMC_ECGSex_finetuned_locked_test_predictions.csv"
)

TRISM_LOCAL_CSV = (
    WCMC_ROOT
    /
    "05_wcmc_from_scratch"
    /
    "predictions"
    /
    "WCMC_ECGSex_from_scratch_locked_test_predictions.csv"
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
    /
    "Figure_Current_WCMC_Baseline_Comparison"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

STEM = (
    OUT_DIR
    /
    "Figure_Current_WCMC_Baseline_ROC_PRC_F1_Brier"
)

PNG_OUT = Path(str(STEM) + ".png")
PDF_OUT = Path(str(STEM) + ".pdf")
SVG_OUT = Path(str(STEM) + ".svg")
TABLE_OUT = Path(str(STEM) + "_SourceData.csv")
CURVE_AUDIT_OUT = Path(str(STEM) + "_CurveSourceAudit.csv")


# ==========================================================================================
# CURRENT LOCKED-TEST MODEL RESULTS
# ==========================================================================================

MODEL_ROWS = [
    {
        "key": "VCG",
        "label": "VCG-Geometry ResNet",
        "protocol": "MIMIC→WCMC, 5-seed ensemble",
        "AUROC": 0.954230,
        "AUPRC": 0.950889,
        "F1": 0.878859,
        "Brier": 0.085006,
    },
    {
        "key": "ResNetTransfer",
        "label": "ResNet1D",
        "protocol": "MIMIC→WCMC, 5-seed ensemble",
        "AUROC": 0.950463,
        "AUPRC": 0.946670,
        "F1": 0.873242,
        "Brier": 0.088621,
    },
    {
        "key": "TRiSMTransfer",
        "label": "CNN–TCN–Transformer",
        "protocol": "MIMIC→WCMC",
        "AUROC": 0.944491,
        "AUPRC": 0.939860,
        "F1": 0.868093,
        "Brier": 0.094134,
    },
    {
        "key": "ResNetLocal",
        "label": "ResNet1D",
        "protocol": "WCMC-only",
        "AUROC": 0.938674,
        "AUPRC": 0.932386,
        "F1": 0.860929,
        "Brier": 0.099295,
    },
    {
        "key": "TRiSMLocal",
        "label": "CNN–TCN–Transformer",
        "protocol": "WCMC-only",
        "AUROC": 0.929352,
        "AUPRC": 0.922941,
        "F1": 0.846638,
        "Brier": 0.110256,
    },
    {
        "key": "CNNLSTM",
        "label": "CNN–LSTM",
        "protocol": "WCMC-only",
        "AUROC": 0.926467,
        "AUPRC": 0.920764,
        "F1": 0.843049,
        "Brier": 0.116783,
    },
    {
        "key": "CNN",
        "label": "CNN",
        "protocol": "WCMC-only",
        "AUROC": 0.918361,
        "AUPRC": 0.912966,
        "F1": 0.835764,
        "Brier": 0.120049,
    },
    {
        "key": "Transformer",
        "label": "Transformer",
        "protocol": "WCMC-only",
        "AUROC": 0.910683,
        "AUPRC": 0.902892,
        "F1": 0.822756,
        "Brier": 0.122543,
    },
]

perf = pd.DataFrame(MODEL_ROWS)

perf["display"] = (
    perf["label"]
    +
    "\n("
    +
    perf["protocol"]
    +
    ")"
)

perf = (
    perf
    .sort_values(
        "AUROC",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)

perf.to_csv(
    TABLE_OUT,
    index=False,
)


# ==========================================================================================
# HELPERS
# ==========================================================================================

def require_file(path):
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Missing required current-run file:\n{path}"
        )


def require_columns(df, cols, label):
    missing = [
        c
        for c in cols
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"{label} missing columns: {missing}\n"
            f"Observed: {df.columns.tolist()}"
        )


def standardize_prediction(
    df,
    prob_col=None,
):
    require_columns(
        df,
        [
            "sex_true_binary",
        ],
        "Prediction dataframe",
    )

    if prob_col is None:
        probability_candidates = [
            "prob_male",
            "prob_vcg",
            "prob_resnet",
            "y_prob",
            "pred_prob",
            "probability",
        ]

        prob_col = next(
            (
                c
                for c in probability_candidates
                if c in df.columns
            ),
            None,
        )

    if prob_col is None:
        raise RuntimeError(
            "Could not identify probability column."
        )

    out = df[
        [
            "sex_true_binary",
            prob_col,
        ]
    ].copy()

    out.columns = [
        "y",
        "p",
    ]

    out[
        "y"
    ] = pd.to_numeric(
        out[
            "y"
        ],
        errors="raise",
    ).astype(
        int
    )

    out[
        "p"
    ] = pd.to_numeric(
        out[
            "p"
        ],
        errors="raise",
    )

    if out[
        "p"
    ].isna().any():
        raise RuntimeError(
            "Prediction probability contains NaN."
        )

    return out


def curve_display_label(row):
    if row["key"] == "VCG":
        return "VCG-Geometry ResNet"

    if row["key"] == "ResNetTransfer":
        return "ResNet1D (transfer)"

    if row["key"] == "TRiSMTransfer":
        return "CNN–TCN–Transformer (transfer)"

    if row["key"] == "ResNetLocal":
        return "ResNet1D (WCMC-only)"

    if row["key"] == "TRiSMLocal":
        return "CNN–TCN–Transformer (WCMC-only)"

    return row["label"]


# ==========================================================================================
# REFERENCE LOCKED-TEST LABEL ORDER
# ==========================================================================================

require_file(
    TEST_CSV
)

test = pd.read_csv(
    TEST_CSV,
    low_memory=False,
)

require_columns(
    test,
    [
        "sex_binary",
    ],
    "Locked WCMC test split",
)

reference_y = pd.to_numeric(
    test[
        "sex_binary"
    ],
    errors="raise",
).astype(
    int
).to_numpy()

N_LOCKED = len(
    reference_y
)

if N_LOCKED != 21374:
    raise RuntimeError(
        f"Expected locked WCMC N=21,374; observed {N_LOCKED:,}"
    )


# ==========================================================================================
# EXACT CURVE SOURCES
# ==========================================================================================

curve_predictions = {}
curve_audit = []


# Final VCG + ResNet from exact aligned combined file.
require_file(
    FINAL_5SEED_CSV
)

final_pred = pd.read_csv(
    FINAL_5SEED_CSV,
    low_memory=False,
)

require_columns(
    final_pred,
    [
        "sex_true_binary",
        "prob_vcg",
        "prob_resnet",
    ],
    "Final 5-seed prediction file",
)

for model_key, prob_col in [
    (
        "VCG",
        "prob_vcg",
    ),
    (
        "ResNetTransfer",
        "prob_resnet",
    ),
]:
    d = standardize_prediction(
        final_pred,
        prob_col=prob_col,
    )

    if len(
        d
    ) != N_LOCKED:
        raise RuntimeError(
            f"{model_key} prediction N={len(d):,}; expected {N_LOCKED:,}"
        )

    if not np.array_equal(
        d[
            "y"
        ].to_numpy(
            dtype=int
        ),
        reference_y,
    ):
        raise RuntimeError(
            f"{model_key}: exact final prediction labels do not match locked-test order."
        )

    curve_predictions[
        model_key
    ] = d

    curve_audit.append(
        {
            "model_key":
                model_key,

            "status":
                "exact",

            "prediction_file":
                str(
                    FINAL_5SEED_CSV
                ),

            "probability_column":
                prob_col,

            "N":
                len(
                    d
                ),
        }
    )


# TRiSM transfer/local exact files.
for model_key, path in [
    (
        "TRiSMTransfer",
        TRISM_TRANSFER_CSV,
    ),
    (
        "TRiSMLocal",
        TRISM_LOCAL_CSV,
    ),
]:
    require_file(
        path
    )

    raw = pd.read_csv(
        path,
        low_memory=False,
    )

    d = standardize_prediction(
        raw,
        prob_col="prob_male",
    )

    if len(
        d
    ) != N_LOCKED:
        raise RuntimeError(
            f"{model_key}: N={len(d):,}; expected {N_LOCKED:,}"
        )

    if not np.array_equal(
        d[
            "y"
        ].to_numpy(
            dtype=int
        ),
        reference_y,
    ):
        raise RuntimeError(
            f"{model_key}: labels do not match locked-test order."
        )

    curve_predictions[
        model_key
    ] = d

    curve_audit.append(
        {
            "model_key":
                model_key,

            "status":
                "exact",

            "prediction_file":
                str(
                    path
                ),

            "probability_column":
                "prob_male",

            "N":
                len(
                    d
                ),
        }
    )


# ==========================================================================================
# OPTIONAL CURRENT-RUN CURVE DISCOVERY
# ==========================================================================================
#
# This is deliberately conservative and cannot override exact sources.
# A candidate must pass the exact label-order audit.
# ==========================================================================================

OPTIONAL_SEARCH = {
    "ResNetLocal": {
        "include_any": [
            "resnet",
            "resnet1d",
        ],
        "exclude": [
            "5seed",
            "vcg",
            "external",
            "ptb",
            "chapman",
            "transfer",
            "mimic_to_wcmc",
        ],
    },

    "CNNLSTM": {
        "include_any": [
            "cnnlstm",
            "cnn_lstm",
            "cnn-lstm",
        ],
        "exclude": [
            "external",
            "ptb",
            "chapman",
        ],
    },

    "CNN": {
        "include_any": [
            "simplecnn",
            "cnn_only",
            "cnn1d",
        ],
        "exclude": [
            "cnnlstm",
            "cnn_lstm",
            "resnet",
            "external",
            "ptb",
            "chapman",
        ],
    },

    "Transformer": {
        "include_any": [
            "puretransformer",
            "transformer_only",
            "transformer",
        ],
        "exclude": [
            "trism",
            "cnn",
            "tcn",
            "external",
            "ptb",
            "chapman",
        ],
    },
}


def candidate_score(
    path,
    spec,
):
    s = str(
        path
    ).lower()

    name = path.name.lower()

    if not any(
        token
        in
        s
        for token in spec[
            "include_any"
        ]
    ):
        return -999

    if any(
        token
        in
        s
        for token in spec[
            "exclude"
        ]
    ):
        return -999

    score = 0

    if "predict" in name:
        score += 8

    if "locked" in name:
        score += 6

    if "test" in name:
        score += 5

    if "wcmc" in s:
        score += 4

    if "baseline" in s:
        score += 2

    return score


all_csv = sorted(
    WCMC_ROOT.rglob(
        "*.csv"
    )
)

for model_key, spec in OPTIONAL_SEARCH.items():
    if model_key in curve_predictions:
        continue

    candidates = []

    for path in all_csv:
        score = candidate_score(
            path,
            spec,
        )

        if score < 0:
            continue

        try:
            head = pd.read_csv(
                path,
                nrows=5,
                low_memory=False,
            )

            if "sex_true_binary" not in head.columns:
                continue

            prob_col = next(
                (
                    c
                    for c in [
                        "prob_male",
                        "y_prob",
                        "pred_prob",
                        "probability",
                    ]
                    if c in head.columns
                ),
                None,
            )

            if prob_col is None:
                continue

            candidates.append(
                (
                    score,
                    path,
                    prob_col,
                )
            )

        except Exception:
            continue

    candidates = sorted(
        candidates,
        key=lambda x:
            (
                -x[
                    0
                ],
                str(
                    x[
                        1
                    ]
                ),
            ),
    )

    accepted = None

    for score, path, prob_col in candidates:
        try:
            raw = pd.read_csv(
                path,
                low_memory=False,
            )

            d = standardize_prediction(
                raw,
                prob_col=prob_col,
            )

            if len(
                d
            ) != N_LOCKED:
                continue

            if not np.array_equal(
                d[
                    "y"
                ].to_numpy(
                    dtype=int
                ),
                reference_y,
            ):
                continue

            accepted = (
                d,
                path,
                prob_col,
            )

            break

        except Exception:
            continue

    if accepted is None:
        curve_audit.append(
            {
                "model_key":
                    model_key,

                "status":
                    "bar-only: aligned prediction CSV not found",

                "prediction_file":
                    "",

                "probability_column":
                    "",

                "N":
                    N_LOCKED,
            }
        )

    else:
        d, path, prob_col = accepted

        curve_predictions[
            model_key
        ] = d

        curve_audit.append(
            {
                "model_key":
                    model_key,

                "status":
                    "discovered + exact label audit passed",

                "prediction_file":
                    str(
                        path
                    ),

                "probability_column":
                    prob_col,

                "N":
                    len(
                        d
                    ),
            }
        )


curve_audit_df = pd.DataFrame(
    curve_audit
)

curve_audit_df.to_csv(
    CURVE_AUDIT_OUT,
    index=False,
)


# ==========================================================================================
# CURVE METRIC AUDIT AGAINST STORED POINT ESTIMATES
# ==========================================================================================

curve_rows = []

for _, row in perf.iterrows():
    key = row[
        "key"
    ]

    if key not in curve_predictions:
        continue

    d = curve_predictions[
        key
    ]

    y = d[
        "y"
    ].to_numpy(
        dtype=int
    )

    p = d[
        "p"
    ].to_numpy(
        dtype=float
    )

    auc = float(
        roc_auc_score(
            y,
            p,
        )
    )

    ap = float(
        average_precision_score(
            y,
            p,
        )
    )

    curve_rows.append(
        {
            "key":
                key,

            "display":
                curve_display_label(
                    row
                ),

            "AUROC_curve":
                auc,

            "AUROC_stored":
                float(
                    row[
                        "AUROC"
                    ]
                ),

            "AUROC_abs_difference":
                abs(
                    auc
                    -
                    float(
                        row[
                            "AUROC"
                        ]
                    )
                ),

            "AUPRC_curve":
                ap,

            "AUPRC_stored":
                float(
                    row[
                        "AUPRC"
                    ]
                ),

            "AUPRC_abs_difference":
                abs(
                    ap
                    -
                    float(
                        row[
                            "AUPRC"
                        ]
                    )
                ),
        }
    )

curve_metric_audit = pd.DataFrame(
    curve_rows
)

print(
    "=" * 120
)

print(
    "CURVE SOURCE AUDIT"
)

print(
    "=" * 120
)

print(
    curve_audit_df.to_string(
        index=False
    )
)

print(
    "\nCurve metric audit:"
)

print(
    curve_metric_audit.to_string(
        index=False
    )
)


# ==========================================================================================
# COLOR MAP
# ==========================================================================================

CMAP = plt.get_cmap(
    "plasma"
)

MODEL_COLORS = {
    "VCG":
        CMAP(
            0.94
        ),

    "ResNetTransfer":
        CMAP(
            0.77
        ),

    "TRiSMTransfer":
        CMAP(
            0.63
        ),

    "ResNetLocal":
        CMAP(
            0.51
        ),

    "TRiSMLocal":
        CMAP(
            0.41
        ),

    "CNNLSTM":
        CMAP(
            0.31
        ),

    "CNN":
        CMAP(
            0.22
        ),

    "Transformer":
        CMAP(
            0.13
        ),
}


# ==========================================================================================
# STYLE
# ==========================================================================================

plt.rcParams.update(
    {
        "font.family":
            "DejaVu Sans",

        "font.size":
            10.2,

        "axes.titlesize":
            11.8,

        "axes.labelsize":
            10.4,

        "xtick.labelsize":
            8.8,

        "ytick.labelsize":
            8.8,

        "legend.fontsize":
            7.7,

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
        14.2,
        10.8,
    ),
)

axA, axB = axes[
    0
]

axC, axD = axes[
    1
]


# ==========================================================================================
# A — ROC CURVES
# ==========================================================================================

for _, row in perf.iterrows():
    key = row[
        "key"
    ]

    if key not in curve_predictions:
        continue

    d = curve_predictions[
        key
    ]

    y = d[
        "y"
    ].to_numpy(
        dtype=int
    )

    p = d[
        "p"
    ].to_numpy(
        dtype=float
    )

    fpr, tpr, _ = roc_curve(
        y,
        p,
    )

    auc = roc_auc_score(
        y,
        p,
    )

    axA.plot(
        fpr,
        tpr,
        color=MODEL_COLORS[
            key
        ],
        linewidth=(
            2.8
            if key
            ==
            "VCG"
            else
            2.0
        ),
        label=(
            f"{curve_display_label(row)} "
            f"(AUROC {auc:.3f})"
        ),
    )

axA.plot(
    [
        0,
        1,
    ],
    [
        0,
        1,
    ],
    linestyle="--",
    color="0.55",
    linewidth=1.0,
)

axA.set_xlim(
    0,
    1
)

axA.set_ylim(
    0,
    1.01
)

axA.set_xlabel(
    "False positive rate"
)

axA.set_ylabel(
    "True positive rate"
)

axA.set_title(
    "ROC curves"
)

axA.grid(
    alpha=0.15
)

axA.legend(
    frameon=False,
    loc="lower right",
)


# ==========================================================================================
# B — PRECISION–RECALL CURVES
# ==========================================================================================

prevalence = float(
    np.mean(
        reference_y
        ==
        1
    )
)

for _, row in perf.iterrows():
    key = row[
        "key"
    ]

    if key not in curve_predictions:
        continue

    d = curve_predictions[
        key
    ]

    y = d[
        "y"
    ].to_numpy(
        dtype=int
    )

    p = d[
        "p"
    ].to_numpy(
        dtype=float
    )

    precision, recall, _ = precision_recall_curve(
        y,
        p,
    )

    ap = average_precision_score(
        y,
        p,
    )

    axB.plot(
        recall,
        precision,
        color=MODEL_COLORS[
            key
        ],
        linewidth=(
            2.8
            if key
            ==
            "VCG"
            else
            2.0
        ),
        label=(
            f"{curve_display_label(row)} "
            f"(AUPRC {ap:.3f})"
        ),
    )

axB.axhline(
    prevalence,
    linestyle="--",
    color="0.55",
    linewidth=1.0,
    label=f"Prevalence ({prevalence:.3f})",
)

axB.set_xlim(
    0,
    1
)

axB.set_ylim(
    0,
    1.01
)

axB.set_xlabel(
    "Recall"
)

axB.set_ylabel(
    "Precision"
)

axB.set_title(
    "Precision–Recall curves"
)

axB.grid(
    alpha=0.15
)

axB.legend(
    frameon=False,
    loc="lower left",
)


# ==========================================================================================
# BAR HELPER
# ==========================================================================================

def metric_bar(
    ax,
    metric,
    ylabel,
    lower_is_better=False,
):
    plot_df = perf.iloc[
        ::-1
    ].copy()

    y = np.arange(
        len(
            plot_df
        )
    )

    values = plot_df[
        metric
    ].to_numpy(
        dtype=float
    )

    colors = [
        MODEL_COLORS[
            key
        ]
        for key in plot_df[
            "key"
        ]
    ]

    bars = ax.barh(
        y,
        values,
        color=colors,
        edgecolor="black",
        linewidth=0.45,
        height=0.66,
    )

    # Hatch single-model comparator bars; final two ensembles remain solid.
    for bar, key in zip(
        bars,
        plot_df[
            "key"
        ],
    ):
        if key not in [
            "VCG",
            "ResNetTransfer",
        ]:
            bar.set_hatch(
                "//"
            )

    labels = []

    for _, row in plot_df.iterrows():
        if row[
            "key"
        ] == "VCG":
            labels.append(
                "VCG-Geometry ResNet\n(MIMIC→WCMC)"
            )

        elif row[
            "key"
        ] == "ResNetTransfer":
            labels.append(
                "ResNet1D\n(MIMIC→WCMC)"
            )

        elif row[
            "key"
        ] == "TRiSMTransfer":
            labels.append(
                "CNN–TCN–Transformer\n(MIMIC→WCMC)"
            )

        else:
            labels.append(
                f"{row['label']}\n(WCMC-only)"
            )

    ax.set_yticks(
        y
    )

    ax.set_yticklabels(
        labels
    )

    ax.set_xlabel(
        ylabel
    )

    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.16,
    )

    vmin = float(
        values.min()
    )

    vmax = float(
        values.max()
    )

    span = max(
        vmax
        -
        vmin,
        0.01,
    )

    left = max(
        0,
        vmin
        -
        0.30
        *
        span,
    )

    right = vmax + 0.42 * span

    if not lower_is_better:
        right = min(
            1.0,
            right,
        )

    ax.set_xlim(
        left,
        right,
    )

    for bar, value, key in zip(
        bars,
        values,
        plot_df[
            "key"
        ],
    ):
        ax.text(
            value
            +
            0.018
            *
            (
                right
                -
                left
            ),
            bar.get_y()
            +
            bar.get_height()
            /
            2,
            f"{value:.3f}",
            va="center",
            ha="left",
            fontsize=7.7,
            fontweight=(
                "bold"
                if key
                ==
                "VCG"
                else
                "normal"
            ),
        )


# ==========================================================================================
# C — F1 BAR
# ==========================================================================================

metric_bar(
    axC,
    metric="F1",
    ylabel="F1 score",
    lower_is_better=False,
)

axC.set_title(
    "F1 score"
)


# ==========================================================================================
# D — BRIER BAR
# ==========================================================================================

metric_bar(
    axD,
    metric="Brier",
    ylabel="Brier score (lower is better)",
    lower_is_better=True,
)

axD.set_title(
    "Brier score"
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
# TITLE / FOOTNOTE
# ==========================================================================================

fig.suptitle(
    "Current WCMC locked-test baseline comparison",
    fontsize=16.5,
    fontweight="bold",
    y=0.992,
)

n_curves = len(
    curve_predictions
)

fig.text(
    0.5,
    0.965,
    (
        f"Locked test N={N_LOCKED:,}. "
        f"A–B show {n_curves} models with verified aligned prediction files; "
        "C–D show all completed current-run comparators."
    ),
    ha="center",
    va="top",
    fontsize=9.2,
)

fig.subplots_adjust(
    left=0.16,
    right=0.985,
    bottom=0.08,
    top=0.925,
    wspace=0.30,
    hspace=0.32,
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
# SUMMARY
# ==========================================================================================

print(
    "\n"
    +
    "=" * 120
)

print(
    "CURRENT WCMC BASELINE COMPARISON"
)

print(
    "=" * 120
)

print(
    "\nCompleted locked-test point estimates:"
)

print(
    perf[
        [
            "label",
            "protocol",
            "AUROC",
            "AUPRC",
            "F1",
            "Brier",
        ]
    ].to_string(
        index=False
    )
)

print(
    "\nModels included in ROC / PR curves:"
)

for key in perf[
    "key"
]:
    status = (
        "CURVE + BAR"
        if key in curve_predictions
        else
        "BAR ONLY"
    )

    print(
        f"{key:<18} {status}"
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
    TABLE_OUT
)

print(
    CURVE_AUDIT_OUT
)

print(
    "\nDONE"
)
