#!/usr/bin/env python
# ==========================================================================================
# ECG-SEX — CURRENT FINAL BASELINE MODEL COMPARISON
# WCMC/NYP LOCKED TEST | FINAL VCG STUDY
# ==========================================================================================
#
# Replaces the legacy EchoNext baseline-comparison figure.
#
# Panels
# ------
# A. AUROC
# B. AUPRC
# C. F1 score
# D. Brier score
#
# CURRENT LOCKED-TEST COMPARATORS
# -------------------------------
# Final transferred 5-seed ensembles:
#   1. VCG-Geometry ResNet
#   2. ResNet1D
#
# Completed current-run single-model comparators:
#   3. CNN–TCN–Transformer (MIMIC→WCMC)
#   4. ResNet1D (WCMC-only)
#   5. CNN–TCN–Transformer (WCMC-only)
#   6. CNN–LSTM (WCMC-only)
#   7. CNN (WCMC-only)
#   8. Transformer (WCMC-only)
#
# IMPORTANT
# ---------
# - All values below are the already-completed CURRENT locked WCMC test results.
# - This script is a publication plotting script; it performs NO model inference/training.
# - The final VCG and matched ResNet entries are 5-seed probability ensembles.
# - Earlier architecture comparators are the completed single-model locked-test benchmarks.
# - Higher is better for AUROC, AUPRC, and F1.
# - Lower is better for Brier.
# - Classical EchoNext engineered-feature baselines from the legacy figure are intentionally
#   removed because they are not part of the final WCMC locked-test architecture comparison.
# ==========================================================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ==========================================================================================
# PATHS
# ==========================================================================================

RUN_DIR = Path(
    "/athena/madelab/scratch/iqh4001/ECG/ECG_Echo/Runs/"
    "Master_ECG_Sex_MIMIC_EchoNext/"
    "20260609_020154_MASTER_ECG_SEX_MIMIC_ECHONEXT"
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

PNG_OUT = (
    OUT_DIR
    /
    "Figure_Current_WCMC_Baseline_Comparison_2x2_Plasma.png"
)

PDF_OUT = (
    OUT_DIR
    /
    "Figure_Current_WCMC_Baseline_Comparison_2x2_Plasma.pdf"
)

SVG_OUT = (
    OUT_DIR
    /
    "Figure_Current_WCMC_Baseline_Comparison_2x2_Plasma.svg"
)

CSV_OUT = (
    OUT_DIR
    /
    "Table_Current_WCMC_Baseline_Comparison.csv"
)

DELTA_OUT = (
    OUT_DIR
    /
    "Table_Current_WCMC_Baseline_Delta_vs_FinalVCG.csv"
)


# ==========================================================================================
# CURRENT FINAL LOCKED-TEST SOURCE DATA
# ==========================================================================================
#
# These are the completed current-run WCMC locked-test point estimates.
#
# Final 5-seed:
#   VCG     AUROC .954230 | AUPRC .950889 | F1 .878859 | Brier .085006
#   ResNet  AUROC .950463 | AUPRC .946670 | F1 .873242 | Brier .088621
#
# Current architecture comparators:
#   TRiSM MIMIC→WCMC .944491/.939860/.868093/.094134
#   ResNet local     .938674/.932386/.860929/.099295
#   TRiSM local      .929352/.922941/.846638/.110256
#   CNN-LSTM         .926467/.920764/.843049/.116783
#   CNN              .918361/.912966/.835764/.120049
#   Transformer      .910683/.902892/.822756/.122543
# ==========================================================================================

rows = [
    {
        "Model":
            "VCG-Geometry ResNet\n(MIMIC→WCMC)",

        "Short":
            "VCG-ResNet",

        "Protocol":
            "Final transferred 5-seed ensemble",

        "Display_Group":
            "Final 5-seed ensemble",

        "AUROC":
            0.954230,

        "AUPRC":
            0.950889,

        "F1":
            0.878859,

        "Brier":
            0.085006,
    },

    {
        "Model":
            "ResNet1D\n(MIMIC→WCMC)",

        "Short":
            "ResNet1D",

        "Protocol":
            "Matched transferred 5-seed ensemble",

        "Display_Group":
            "Final 5-seed ensemble",

        "AUROC":
            0.950463,

        "AUPRC":
            0.946670,

        "F1":
            0.873242,

        "Brier":
            0.088621,
    },

    {
        "Model":
            "CNN–TCN–Transformer\n(MIMIC→WCMC)",

        "Short":
            "TRiSM transfer",

        "Protocol":
            "Transferred single-model comparator",

        "Display_Group":
            "Completed single-model comparator",

        "AUROC":
            0.944491,

        "AUPRC":
            0.939860,

        "F1":
            0.868093,

        "Brier":
            0.094134,
    },

    {
        "Model":
            "ResNet1D\n(WCMC-only)",

        "Short":
            "ResNet local",

        "Protocol":
            "WCMC-only single-model comparator",

        "Display_Group":
            "Completed single-model comparator",

        "AUROC":
            0.938674,

        "AUPRC":
            0.932386,

        "F1":
            0.860929,

        "Brier":
            0.099295,
    },

    {
        "Model":
            "CNN–TCN–Transformer\n(WCMC-only)",

        "Short":
            "TRiSM local",

        "Protocol":
            "WCMC-only single-model comparator",

        "Display_Group":
            "Completed single-model comparator",

        "AUROC":
            0.929352,

        "AUPRC":
            0.922941,

        "F1":
            0.846638,

        "Brier":
            0.110256,
    },

    {
        "Model":
            "CNN–LSTM\n(WCMC-only)",

        "Short":
            "CNN-LSTM",

        "Protocol":
            "WCMC-only single-model comparator",

        "Display_Group":
            "Completed single-model comparator",

        "AUROC":
            0.926467,

        "AUPRC":
            0.920764,

        "F1":
            0.843049,

        "Brier":
            0.116783,
    },

    {
        "Model":
            "CNN\n(WCMC-only)",

        "Short":
            "CNN",

        "Protocol":
            "WCMC-only single-model comparator",

        "Display_Group":
            "Completed single-model comparator",

        "AUROC":
            0.918361,

        "AUPRC":
            0.912966,

        "F1":
            0.835764,

        "Brier":
            0.120049,
    },

    {
        "Model":
            "Transformer\n(WCMC-only)",

        "Short":
            "Transformer",

        "Protocol":
            "WCMC-only single-model comparator",

        "Display_Group":
            "Completed single-model comparator",

        "AUROC":
            0.910683,

        "AUPRC":
            0.902892,

        "F1":
            0.822756,

        "Brier":
            0.122543,
    },
]

df = pd.DataFrame(
    rows
)

# Keep a single consistent ranking defined by AUROC.
df = (
    df
    .sort_values(
        "AUROC",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)

df.insert(
    0,
    "Rank_by_AUROC",
    np.arange(
        1,
        len(
            df
        )
        +
        1,
    ),
)

df.to_csv(
    CSV_OUT,
    index=False,
)


# ==========================================================================================
# DELTA TABLE vs FINAL VCG
# ==========================================================================================

final = df.loc[
    df[
        "Short"
    ]
    ==
    "VCG-ResNet"
].iloc[
    0
]

delta_rows = []

for _, row in df.iterrows():
    delta_rows.append(
        {
            "Model":
                row[
                    "Short"
                ],

            "Delta_AUROC_vs_VCG":
                float(
                    final[
                        "AUROC"
                    ]
                    -
                    row[
                        "AUROC"
                    ]
                ),

            "Delta_AUPRC_vs_VCG":
                float(
                    final[
                        "AUPRC"
                    ]
                    -
                    row[
                        "AUPRC"
                    ]
                ),

            "Delta_F1_vs_VCG":
                float(
                    final[
                        "F1"
                    ]
                    -
                    row[
                        "F1"
                    ]
                ),

            # Positive here means VCG has a lower (better) Brier.
            "Brier_improvement_VCG":
                float(
                    row[
                        "Brier"
                    ]
                    -
                    final[
                        "Brier"
                    ]
                ),
        }
    )

delta_df = pd.DataFrame(
    delta_rows
)

delta_df.to_csv(
    DELTA_OUT,
    index=False,
)


# ==========================================================================================
# COLORS
# ==========================================================================================

CMAP = plt.get_cmap(
    "plasma"
)

# Final models deliberately distinct; remaining comparators use ordered plasma tones.
color_map = {
    "VCG-ResNet":
        CMAP(
            0.94
        ),

    "ResNet1D":
        CMAP(
            0.72
        ),

    "TRiSM transfer":
        CMAP(
            0.58
        ),

    "ResNet local":
        CMAP(
            0.47
        ),

    "TRiSM local":
        CMAP(
            0.38
        ),

    "CNN-LSTM":
        CMAP(
            0.30
        ),

    "CNN":
        CMAP(
            0.22
        ),

    "Transformer":
        CMAP(
            0.14
        ),
}

colors = [
    color_map[
        short
    ]
    for short in df[
        "Short"
    ]
]


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
            8.6,

        "ytick.labelsize":
            8.7,

        "legend.fontsize":
            8.2,

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
# PLOTTING HELPER
# ==========================================================================================

def horizontal_metric_panel(
    ax,
    metric,
    xlabel,
    panel_label,
    higher_is_better=True,
):
    # Keep best-at-top order.
    plot_df = df.iloc[
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

    bar_colors = [
        color_map[
            s
        ]
        for s in plot_df[
            "Short"
        ]
    ]

    bars = ax.barh(
        y,
        values,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.50,
        height=0.68,
    )

    # Hatch completed single-model comparators so ensemble vs single-model
    # status is visually explicit.
    for bar, group in zip(
        bars,
        plot_df[
            "Display_Group"
        ],
    ):
        if group == "Completed single-model comparator":
            bar.set_hatch(
                "//"
            )

    ax.set_yticks(
        y
    )

    ax.set_yticklabels(
        plot_df[
            "Model"
        ]
    )

    ax.set_xlabel(
        xlabel
    )

    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.18,
    )

    # Data-driven compact axis so differences are legible.
    vmin = float(
        np.min(
            values
        )
    )

    vmax = float(
        np.max(
            values
        )
    )

    span = max(
        vmax
        -
        vmin,
        0.01,
    )

    if higher_is_better:
        left = max(
            0,
            vmin
            -
            0.30
            *
            span,
        )

        right = min(
            1.0,
            vmax
            +
            0.42
            *
            span,
        )

    else:
        left = max(
            0,
            vmin
            -
            0.35
            *
            span,
        )

        right = vmax + 0.30 * span

    ax.set_xlim(
        left,
        right,
    )

    for bar, value in zip(
        bars,
        values,
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
                if
                np.isclose(
                    value,
                    df.loc[
                        df[
                            "Short"
                        ]
                        ==
                        "VCG-ResNet",
                        metric,
                    ].iloc[
                        0
                    ],
                )
                else
                "normal"
            ),
        )

    ax.text(
        -0.18,
        1.045,
        panel_label,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        ha="left",
        va="top",
    )


# ==========================================================================================
# FIGURE
# ==========================================================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(
        14.2,
        11.0,
    ),
)

axA, axB = axes[
    0
]

axC, axD = axes[
    1
]

horizontal_metric_panel(
    axA,
    metric="AUROC",
    xlabel="AUROC",
    panel_label="A",
    higher_is_better=True,
)

horizontal_metric_panel(
    axB,
    metric="AUPRC",
    xlabel="AUPRC",
    panel_label="B",
    higher_is_better=True,
)

horizontal_metric_panel(
    axC,
    metric="F1",
    xlabel="F1 score",
    panel_label="C",
    higher_is_better=True,
)

horizontal_metric_panel(
    axD,
    metric="Brier",
    xlabel="Brier score (lower is better)",
    panel_label="D",
    higher_is_better=False,
)


# ==========================================================================================
# FINAL MODEL CALLOUT
# ==========================================================================================

# Add concise improvement annotation in A only: final VCG vs matched final ResNet.
vcg_auc = float(
    df.loc[
        df[
            "Short"
        ]
        ==
        "VCG-ResNet",
        "AUROC",
    ].iloc[
        0
    ]
)

res_auc = float(
    df.loc[
        df[
            "Short"
        ]
        ==
        "ResNet1D",
        "AUROC",
    ].iloc[
        0
    ]
)

axA.text(
    0.98,
    0.04,
    f"VCG vs matched ResNet: ΔAUROC = +{vcg_auc - res_auc:.4f}",
    transform=axA.transAxes,
    ha="right",
    va="bottom",
    fontsize=8.2,
)


# ==========================================================================================
# LEGEND
# ==========================================================================================

legend_handles = [
    Patch(
        facecolor=CMAP(
            0.82
        ),
        edgecolor="black",
        label="Final transferred 5-seed ensemble",
    ),

    Patch(
        facecolor=CMAP(
            0.34
        ),
        edgecolor="black",
        hatch="//",
        label="Completed single-model comparator",
    ),
]

fig.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(
        0.5,
        0.012,
    ),
    ncol=2,
    frameon=False,
)


# ==========================================================================================
# TITLE / NOTE
# ==========================================================================================

fig.suptitle(
    "Current WCMC locked-test baseline comparison",
    fontsize=16.5,
    fontweight="bold",
    y=0.992,
)

fig.text(
    0.5,
    0.965,
    (
        "Final VCG-Geometry ResNet and matched ResNet1D are five-seed probability ensembles; "
        "remaining models are completed locked-test single-model comparators."
    ),
    ha="center",
    va="top",
    fontsize=9.2,
)


fig.subplots_adjust(
    left=0.27,
    right=0.985,
    bottom=0.08,
    top=0.925,
    wspace=0.31,
    hspace=0.31,
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
# CONSOLE SUMMARY
# ==========================================================================================

print(
    "=" * 120
)

print(
    "CURRENT WCMC LOCKED-TEST BASELINE COMPARISON"
)

print(
    "=" * 120
)

print(
    df[
        [
            "Rank_by_AUROC",
            "Short",
            "Protocol",
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
    "\nDelta vs final VCG-Geometry ResNet:"
)

print(
    delta_df.to_string(
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
    CSV_OUT
)

print(
    DELTA_OUT
)

print(
    "\nDONE"
)
