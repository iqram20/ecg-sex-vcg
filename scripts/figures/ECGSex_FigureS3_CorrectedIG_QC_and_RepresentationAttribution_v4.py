#!/usr/bin/env python
# ==========================================================================================
# ECG-SEX — FIGURE S3
# CORRECTED INTEGRATED GRADIENTS QC + REPRESENTATION ATTRIBUTION
# FINAL VCG-GEOMETRY RESNET | CURRENT WCMC RUN
# ==========================================================================================
#
# Panels
# ------
# A. Corrected IG completeness error distributions
# B. Waveform-path lead importance
# C. VCG geometry-group importance
# D. Sex-specific true-class evidence across VCG geometry groups
#
# Uses ONLY the completeness-validated corrected representation-level IG outputs.
# The invalid initial raw-input VCG IG is not used.
#
# No model inference or training.
# ==========================================================================================

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ==========================================================================================
# PATHS
# ==========================================================================================

RUN_DIR = Path(
    "/athena/madelab/scratch/iqh4001/ECG/ECG_Echo/Runs/"
    "Master_ECG_Sex_MIMIC_EchoNext/"
    "20260609_020154_MASTER_ECG_SEX_MIMIC_ECHONEXT"
)

CORRECTED_IG_DIR = (
    RUN_DIR
    /
    "wcmc_ecgsex_from_scratch"
    /
    "22_vcg_geometry_resnet_corrected_ig"
)

QC_JSON = (
    CORRECTED_IG_DIR
    /
    "CorrectedIG_QC_Summary.json"
)

COMPLETENESS_CSV = (
    CORRECTED_IG_DIR
    /
    "CorrectedIG_Completeness_Audit.csv"
)

WAVEFORM_CSV = (
    CORRECTED_IG_DIR
    /
    "CorrectedIG_Waveform_LeadImportance.csv"
)

GEOMETRY_GROUP_CSV = (
    CORRECTED_IG_DIR
    /
    "CorrectedIG_VCGGeometry_GroupImportance.csv"
)

SEX_EVIDENCE_CSV = (
    CORRECTED_IG_DIR
    /
    "CorrectedIG_SexSpecific_Evidence.csv"
)

OUT_DIR = (
    RUN_DIR
    /
    "publication_figures"
    /
    "Supplementary"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

STEM = (
    OUT_DIR
    /
    "FigureS3_CorrectedIG_QC_and_RepresentationAttribution"
)

PNG_OUT = Path(str(STEM) + ".png")
PDF_OUT = Path(str(STEM) + ".pdf")
SVG_OUT = Path(str(STEM) + ".svg")
SOURCE_OUT = Path(str(STEM) + "_SourceData.csv")


# ==========================================================================================
# REQUIRED FILES
# ==========================================================================================

for p in [
    QC_JSON,
    COMPLETENESS_CSV,
    WAVEFORM_CSV,
    GEOMETRY_GROUP_CSV,
    SEX_EVIDENCE_CSV,
]:
    if not p.exists():
        raise FileNotFoundError(
            f"Missing corrected-IG output:\n{p}"
        )


# ==========================================================================================
# LOAD + QC GATE
# ==========================================================================================

with open(
    QC_JSON,
    "r",
) as f:
    qc = json.load(
        f
    )

if not bool(
    qc.get(
        "overall_pass",
        False,
    )
):
    raise RuntimeError(
        "Corrected IG overall completeness QC did not pass."
    )

audit = pd.read_csv(
    COMPLETENESS_CSV,
    low_memory=False,
)

wave = pd.read_csv(
    WAVEFORM_CSV,
    low_memory=False,
)

geom = pd.read_csv(
    GEOMETRY_GROUP_CSV,
    low_memory=False,
)

sex_ev = pd.read_csv(
    SEX_EVIDENCE_CSV,
    low_memory=False,
)


# ==========================================================================================
# COLUMN AUDIT
# ==========================================================================================

def require_cols(
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
            f"Observed: {df.columns.tolist()}"
        )


require_cols(
    audit,
    [
        "waveform_abs_error",
        "waveform_relative_error",
        "geometry_abs_error",
        "geometry_relative_error",
    ],
    "Completeness audit",
)

require_cols(
    wave,
    [
        "lead",
        "mean_abs_IG",
        "normalized_importance",
        "rank",
    ],
    "Waveform lead importance",
)

require_cols(
    geom,
    [
        "group",
        "mean_abs_IG",
        "normalized_importance",
        "female_trueclass_evidence",
        "male_trueclass_evidence",
        "rank",
    ],
    "Geometry group importance",
)

require_cols(
    sex_ev,
    [
        "representation",
        "feature",
        "female_trueclass_evidence",
        "male_trueclass_evidence",
    ],
    "Sex evidence",
)


# ==========================================================================================
# PREP
# ==========================================================================================

wave = (
    wave
    .sort_values(
        "rank"
    )
    .reset_index(
        drop=True
    )
)

geom = (
    geom
    .sort_values(
        "rank"
    )
    .reset_index(
        drop=True
    )
)

# For horizontal plots: reverse so top-ranked appears at top.
wave_plot = wave.iloc[
    ::-1
].copy()

geom_plot = geom.iloc[
    ::-1
].copy()


# ==========================================================================================
# COLORS / STYLE
# ==========================================================================================

CMAP = plt.get_cmap(
    "plasma"
)

COL_WAVE = CMAP(
    0.35
)

COL_GEOM = CMAP(
    0.82
)

COL_FEMALE = CMAP(
    0.24
)

COL_MALE = CMAP(
    0.82
)

plt.rcParams.update(
    {
        "font.family":
            "DejaVu Sans",

        "font.size":
            10,

        "axes.titlesize":
            11.5,

        "axes.labelsize":
            10.2,

        "xtick.labelsize":
            8.6,

        "ytick.labelsize":
            8.6,

        "legend.fontsize":
            8.5,

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
        13.2,
        10.3,
    ),
)

axA, axB = axes[
    0
]

axC, axD = axes[
    1
]


# ==========================================================================================
# A — COMPLETENESS ERRORS
# ==========================================================================================

wave_abs = pd.to_numeric(
    audit[
        "waveform_abs_error"
    ],
    errors="coerce",
).dropna().to_numpy(
    float
)

geom_abs = pd.to_numeric(
    audit[
        "geometry_abs_error"
    ],
    errors="coerce",
).dropna().to_numpy(
    float
)

# log10 only for display; add tiny epsilon in case numerical zero occurs.
eps = 1e-12

data_a = [
    np.log10(
        wave_abs
        +
        eps
    ),
    np.log10(
        geom_abs
        +
        eps
    ),
]

vp = axA.violinplot(
    data_a,
    positions=[
        1,
        2,
    ],
    widths=0.65,
    showmeans=False,
    showmedians=False,
    showextrema=False,
)

for body, col in zip(
    vp[
        "bodies"
    ],
    [
        COL_WAVE,
        COL_GEOM,
    ],
):
    body.set_facecolor(
        col
    )

    body.set_edgecolor(
        "black"
    )

    body.set_alpha(
        0.55
    )

for pos, arr in zip(
    [
        1,
        2,
    ],
    data_a,
):
    q1, med, q3 = np.percentile(
        arr,
        [
            25,
            50,
            75,
        ],
    )

    axA.vlines(
        pos,
        q1,
        q3,
        color="black",
        linewidth=1.4,
    )

    axA.scatter(
        pos,
        med,
        s=28,
        facecolor="white",
        edgecolor="black",
        zorder=4,
    )

axA.set_xticks(
    [
        1,
        2,
    ]
)

axA.set_xticklabels(
    [
        "Waveform pathway",
        "VCG geometry pathway",
    ]
)

axA.set_ylabel(
    r"$\log_{10}$ absolute completeness error"
)

axA.set_title(
    "Corrected IG completeness",
    pad=10,
)

axA.grid(
    axis="y",
    alpha=0.15,
)

wave_med = float(
    np.median(
        wave_abs
    )
)

wave_p95 = float(
    np.percentile(
        wave_abs,
        95,
    )
)

geom_med = float(
    np.median(
        geom_abs
    )
)

geom_p95 = float(
    np.percentile(
        geom_abs,
        95,
    )
)

# QC summary / legend — lower-left of Panel A.
# Keep the annotation compact and away from the central violin bodies.
axA.text(
    0.03,
    0.04,
    (
        f"Waveform: median {wave_med:.2e}; p95 {wave_p95:.2e}\n"
        f"Geometry: median {geom_med:.2e}; p95 {geom_p95:.2e}\n"
        "Completeness QC: PASS"
    ),
    transform=axA.transAxes,
    ha="left",
    va="bottom",
    fontsize=7.6,
    bbox=dict(
        boxstyle="round,pad=0.30",
        facecolor="white",
        edgecolor="0.82",
        linewidth=0.6,
        alpha=0.96,
    ),
    zorder=10,
)


# ==========================================================================================
# B — WAVEFORM LEAD IMPORTANCE
# ==========================================================================================

y = np.arange(
    len(
        wave_plot
    )
)

wave_pct = (
    pd.to_numeric(
        wave_plot[
            "normalized_importance"
        ],
        errors="raise",
    ).to_numpy(
        float
    )
    *
    100.0
)

axB.barh(
    y,
    wave_pct,
    color=COL_WAVE,
    edgecolor="black",
    linewidth=0.5,
)

axB.set_yticks(
    y
)

axB.set_yticklabels(
    wave_plot[
        "lead"
    ]
)

axB.set_xlabel(
    "Normalized absolute IG (%)"
)

axB.set_title(
    "Waveform-path lead attribution"
)

axB.grid(
    axis="x",
    alpha=0.15,
)

for yy, value in zip(
    y,
    wave_pct,
):
    axB.text(
        value
        +
        0.15,
        yy,
        f"{value:.1f}",
        va="center",
        fontsize=7.5,
    )


# ==========================================================================================
# C — VCG GEOMETRY GROUP IMPORTANCE
# ==========================================================================================

y = np.arange(
    len(
        geom_plot
    )
)

geom_pct = (
    pd.to_numeric(
        geom_plot[
            "normalized_importance"
        ],
        errors="raise",
    ).to_numpy(
        float
    )
    *
    100.0
)

axC.barh(
    y,
    geom_pct,
    color=COL_GEOM,
    edgecolor="black",
    linewidth=0.5,
)

axC.set_yticks(
    y
)

axC.set_yticklabels(
    geom_plot[
        "group"
    ]
)

axC.set_xlabel(
    "Normalized absolute IG (%)"
)

axC.set_title(
    "VCG geometry-group attribution"
)

axC.grid(
    axis="x",
    alpha=0.15,
)

for yy, value in zip(
    y,
    geom_pct,
):
    axC.text(
        value
        +
        0.25,
        yy,
        f"{value:.1f}",
        va="center",
        fontsize=7.5,
    )

# Useful synthesis.
top3 = (
    geom
    .head(
        3
    )[
        "normalized_importance"
    ]
    .sum()
    *
    100.0
)

axC.text(
    0.98,
    0.04,
    f"Top 3 geometry groups: {top3:.1f}% of normalized |IG|",
    transform=axC.transAxes,
    ha="right",
    va="bottom",
    fontsize=7.8,
)


# ==========================================================================================
# D — SEX-SPECIFIC GEOMETRY EVIDENCE
# ==========================================================================================

geom_ev = sex_ev.loc[
    sex_ev[
        "representation"
    ]
    ==
    "VCG geometry group"
].copy()

if len(
    geom_ev
) == 0:
    raise RuntimeError(
        "No VCG geometry group rows in CorrectedIG_SexSpecific_Evidence.csv"
    )

# Preserve official geometry importance rank.
rank_map = {
    row[
        "group"
    ]:
        int(
            row[
                "rank"
            ]
        )
    for _, row in geom.iterrows()
}

geom_ev[
    "rank"
] = geom_ev[
    "feature"
].map(
    rank_map
)

geom_ev = (
    geom_ev
    .sort_values(
        "rank"
    )
    .iloc[
        ::-1
    ]
    .reset_index(
        drop=True
    )
)

y = np.arange(
    len(
        geom_ev
    )
)

female = pd.to_numeric(
    geom_ev[
        "female_trueclass_evidence"
    ],
    errors="raise",
).to_numpy(
    float
)

male = pd.to_numeric(
    geom_ev[
        "male_trueclass_evidence"
    ],
    errors="raise",
).to_numpy(
    float
)

height = 0.34

axD.barh(
    y
    -
    height
    /
    2,
    female,
    height=height,
    color=COL_FEMALE,
    edgecolor="black",
    linewidth=0.45,
    label="Female",
)

axD.barh(
    y
    +
    height
    /
    2,
    male,
    height=height,
    color=COL_MALE,
    edgecolor="black",
    linewidth=0.45,
    label="Male",
)

axD.axvline(
    0,
    color="0.45",
    linewidth=0.8,
)

axD.set_yticks(
    y
)

axD.set_yticklabels(
    geom_ev[
        "feature"
    ]
)

axD.set_xlabel(
    "Mean true-class evidence"
)

axD.set_title(
    "Sex-specific VCG geometry evidence"
)

axD.grid(
    axis="x",
    alpha=0.15,
)

axD.legend(
    frameon=False,
    loc="lower right",
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
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=15.5,
        fontweight="bold",
        ha="left",
        va="top",
    )


# ==========================================================================================
# TITLE
# ==========================================================================================

fig.suptitle(
    "Figure S3. Corrected Integrated Gradients quality control and representation attribution",
    fontsize=15.2,
    fontweight="bold",
    y=0.992,
)

fig.text(
    0.5,
    0.963,
    (
        "Final five-seed VCG-Geometry ResNet; corrected representation-level IG; "
        f"N={len(audit)} balanced WCMC ECGs"
    ),
    ha="center",
    va="top",
    fontsize=9.1,
)

fig.subplots_adjust(
    left=0.16,
    right=0.985,
    bottom=0.08,
    top=0.92,
    wspace=0.31,
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
# SOURCE DATA
# ==========================================================================================

source_rows = []

for value in wave_abs:
    source_rows.append(
        {
            "panel":
                "A",

            "category":
                "Waveform completeness absolute error",

            "value":
                value,
        }
    )

for value in geom_abs:
    source_rows.append(
        {
            "panel":
                "A",

            "category":
                "Geometry completeness absolute error",

            "value":
                value,
        }
    )

for _, row in wave.iterrows():
    source_rows.append(
        {
            "panel":
                "B",

            "category":
                row[
                    "lead"
                ],

            "value":
                row[
                    "normalized_importance"
                ],
        }
    )

for _, row in geom.iterrows():
    source_rows.append(
        {
            "panel":
                "C",

            "category":
                row[
                    "group"
                ],

            "value":
                row[
                    "normalized_importance"
                ],
        }
    )

for _, row in geom_ev.iterrows():
    source_rows.append(
        {
            "panel":
                "D",

            "category":
                row[
                    "feature"
                ],

            "Female":
                row[
                    "female_trueclass_evidence"
                ],

            "Male":
                row[
                    "male_trueclass_evidence"
                ],
        }
    )

pd.DataFrame(
    source_rows
).to_csv(
    SOURCE_OUT,
    index=False,
)


# ==========================================================================================
# SUMMARY
# ==========================================================================================

print(
    "=" * 120
)

print(
    "FIGURE S3 COMPLETE"
)

print(
    "=" * 120
)

print(
    "\nCorrected IG completeness QC: PASS"
)

print(
    f"Waveform median abs error : {wave_med:.6g}"
)

print(
    f"Waveform p95 abs error    : {wave_p95:.6g}"
)

print(
    f"Geometry median abs error : {geom_med:.6g}"
)

print(
    f"Geometry p95 abs error    : {geom_p95:.6g}"
)

print(
    "\nWaveform lead importance:"
)

print(
    wave[
        [
            "rank",
            "lead",
            "normalized_importance",
        ]
    ].to_string(
        index=False
    )
)

print(
    "\nVCG geometry-group importance:"
)

print(
    geom[
        [
            "rank",
            "group",
            "normalized_importance",
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
    "\nDONE"
)
