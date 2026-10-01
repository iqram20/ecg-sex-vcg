#!/usr/bin/env python3
# ==========================================================================================
# ECG-SEX — FIGURE 5 FINAL VCG INTERPRETABILITY
# CURRENT FINAL RUN | 2 × 2 | PLASMA
# ==========================================================================================
#
# A. Overall single-lead ablation — horizontal bar plot + paired-bootstrap 95% CI
# B. Precordial vs limb lead importance — boxplot + individual leads
# C. VCG mechanistic ablation — horizontal bar plot + paired-bootstrap 95% CI
# D. CURRENT age-stratified REGIONAL lead ablation — grouped forest plot + 95% CI
#
# IMPORTANT
# ---------
# - NO waveform panel.
# - NO heatmap.
# - NO model inference/training.
# - Panels A-C reuse the frozen/current final interpretability summaries.
# - Panel D now uses the completed CURRENT reviewer analysis:
#       Results/WCMC_AgeStratified_RegionalLeadAblation/
#       05_PRIMARY_AUROC_regional_importance_by_age.csv
# - Panel D shows test-time perturbation/model reliance, not causal physiology.
# - Female=0, Male=1.
# ==========================================================================================

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ==========================================================================================
# EXACT CURRENT-RUN PATHS
# ==========================================================================================

PROJECT = Path("/athena/madelab/scratch/iqh4001/ECG/ECG-gender")

RUN_DIR = Path(
    "/athena/madelab/scratch/iqh4001/ECG/ECG_Echo/Runs/"
    "Master_ECG_Sex_MIMIC_EchoNext/"
    "20260609_020154_MASTER_ECG_SEX_MIMIC_ECHONEXT"
)

WCMC_ROOT = RUN_DIR / "wcmc_ecgsex_from_scratch"

INTERP_ROOT = (
    WCMC_ROOT
    / "21_vcg_geometry_resnet_interpretability"
)

FINAL_DIR = (
    INTERP_ROOT
    / "04_final_summary"
)

OVERALL_LEAD_CSV = (
    FINAL_DIR
    / "LeadAblation_Ranked_By_AUROC_Drop.csv"
)

MECH_CSV = (
    FINAL_DIR
    / "VCG_Mechanistic_Ablation_Ranked.csv"
)

# CURRENT reviewer-added age-stratified REGIONAL analysis.
AGE_REGIONAL_CSV = (
    PROJECT
    / "Results"
    / "WCMC_AgeStratified_RegionalLeadAblation"
    / "05_PRIMARY_AUROC_regional_importance_by_age.csv"
)

OUT_DIR = (
    RUN_DIR
    / "publication_figures"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

STEM = (
    OUT_DIR
    / "Figure5_FINAL_VCG_Interpretability_CURRENT_RegionalAge_v2"
)

PNG_OUT = Path(str(STEM) + ".png")
PDF_OUT = Path(str(STEM) + ".pdf")
SVG_OUT = Path(str(STEM) + ".svg")
SOURCE_OUT = Path(str(STEM) + "_SourceData.csv")


# ==========================================================================================
# CONSTANTS
# ==========================================================================================

LEADS = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]

PRECORDIAL = ["V1", "V2", "V3", "V4", "V5", "V6"]

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

def require_file(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing required current-run file:\n{path}")


def require_columns(df, cols, label):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"{label} missing columns: {missing}\n"
            f"Observed: {df.columns.tolist()}"
        )


def normalize_age_group(x):
    s = str(x).strip()
    replacements = {
        "18-39": "18–39",
        "40-49": "40–49",
        "50-59": "50–59",
        "60-69": "60–69",
        "70-79": "70–79",
        ">=80": "≥80",
        "80+": "≥80",
        "≥ 80": "≥80",
    }
    return replacements.get(s, s)


# ==========================================================================================
# FILE AUDIT
# ==========================================================================================

for p in [
    OVERALL_LEAD_CSV,
    MECH_CSV,
    AGE_REGIONAL_CSV,
]:
    require_file(p)

print("=" * 120)
print("CURRENT FIGURE 5 SOURCE AUDIT")
print("=" * 120)
print("Overall lead ablation :", OVERALL_LEAD_CSV)
print("VCG mechanistic      :", MECH_CSV)
print("Age-regional ablation:", AGE_REGIONAL_CSV)


# ==========================================================================================
# LOAD PANELS A-C
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
    "Overall single-lead ablation",
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
    lead[c] = pd.to_numeric(
        lead[c],
        errors="raise",
    )
    mech[c] = pd.to_numeric(
        mech[c],
        errors="raise",
    )

lead["region"] = np.where(
    lead["lead"].isin(PRECORDIAL),
    "Precordial",
    "Limb",
)

lead = (
    lead
    .sort_values(
        "AUROC_drop",
        ascending=False,
    )
    .reset_index(drop=True)
)

MECH_LABELS = {
    "vcg_branch_off": "VCG branch off",
    "vcg_no_direct_xyz": "No direct XYZ",
    "vcg_no_magnitude_direction": "No magnitude/\ndirection",
    "vcg_no_trajectory_turning": "No trajectory\nturning",
    "vcg_no_velocity_dynamics": "No velocity\ndynamics",
    "fixed_inverse_dower": "Fixed inverse\nDower",
}

mech["display"] = (
    mech["condition"]
    .map(MECH_LABELS)
    .fillna(mech["condition"])
)

mech = (
    mech
    .sort_values(
        "AUROC_drop",
        ascending=False,
    )
    .reset_index(drop=True)
)


# ==========================================================================================
# PANEL B REGIONAL SUMMARY
# ==========================================================================================

regional_rows = []

for region in [
    "Precordial",
    "Limb",
]:
    vals = (
        lead.loc[
            lead["region"].eq(region),
            "AUROC_drop",
        ]
        .to_numpy(dtype=float)
    )

    regional_rows.append(
        {
            "region": region,
            "N_leads": len(vals),
            "mean_AUROC_drop": float(np.mean(vals)),
            "median_AUROC_drop": float(np.median(vals)),
            "Q1": float(np.percentile(vals, 25)),
            "Q3": float(np.percentile(vals, 75)),
        }
    )

regional = pd.DataFrame(regional_rows)


# ==========================================================================================
# LOAD CURRENT PANEL D — AGE-STRATIFIED REGIONAL ABLATION
# ==========================================================================================

age_reg = pd.read_csv(
    AGE_REGIONAL_CSV,
    low_memory=False,
)

require_columns(
    age_reg,
    [
        "age_group",
        "region_removed",
        "N",
        "Female_N",
        "Male_N",
        "metric",
        "intact_metric",
        "ablated_metric",
        "metric_drop",
        "drop_CI_low",
        "drop_CI_high",
    ],
    "Current age-stratified regional lead ablation",
)

age_reg["age_group"] = (
    age_reg["age_group"]
    .map(normalize_age_group)
)

# Keep AUROC only.
age_reg = age_reg.loc[
    age_reg["metric"]
    .astype(str)
    .str.upper()
    .eq("AUROC")
].copy()

for c in [
    "N",
    "Female_N",
    "Male_N",
    "intact_metric",
    "ablated_metric",
    "metric_drop",
    "drop_CI_low",
    "drop_CI_high",
]:
    age_reg[c] = pd.to_numeric(
        age_reg[c],
        errors="coerce",
    )

# Accept common region naming variants.
region_map = {
    "Precordial": "Precordial",
    "All precordial": "Precordial",
    "V1-V6": "Precordial",
    "V1–V6": "Precordial",
    "Limb": "Limb",
    "All limb": "Limb",
    "Precordial-minus-limb contrast": "Contrast",
    "Precordial - limb contrast": "Contrast",
    "Contrast": "Contrast",
}

age_reg["region_display"] = (
    age_reg["region_removed"]
    .astype(str)
    .map(region_map)
    .fillna(age_reg["region_removed"].astype(str))
)

# We want two model-perturbation rows plus the direct contrast.
wanted_regions = [
    "Precordial",
    "Limb",
    "Contrast",
]

age_reg = age_reg.loc[
    age_reg["region_display"].isin(wanted_regions)
].copy()

# Strict current-run audit.
observed_ages = set(
    age_reg["age_group"]
    .dropna()
    .unique()
)

if observed_ages != set(AGE_LABELS):
    raise RuntimeError(
        "Current regional-age file does not contain the expected six age groups.\n"
        f"Expected: {AGE_LABELS}\n"
        f"Observed: {sorted(observed_ages)}"
    )

for age_group in AGE_LABELS:
    present = set(
        age_reg.loc[
            age_reg["age_group"].eq(age_group),
            "region_display",
        ]
        .dropna()
        .tolist()
    )

    required = {
        "Precordial",
        "Limb",
        "Contrast",
    }

    if not required.issubset(present):
        raise RuntimeError(
            f"{age_group}: missing regional rows.\n"
            f"Required: {sorted(required)}\n"
            f"Observed: {sorted(present)}"
        )

age_order = {
    x: i
    for i, x in enumerate(AGE_LABELS)
}

region_order = {
    "Precordial": 0,
    "Limb": 1,
    "Contrast": 2,
}

age_reg["_age_order"] = age_reg["age_group"].map(age_order)
age_reg["_region_order"] = age_reg["region_display"].map(region_order)

age_reg = (
    age_reg
    .sort_values(
        [
            "_age_order",
            "_region_order",
        ]
    )
    .reset_index(drop=True)
)


# ==========================================================================================
# COLORS / STYLE
# ==========================================================================================

CMAP = plt.get_cmap("plasma")

C_LIMB = CMAP(0.25)
C_PRE = CMAP(0.76)
C_CONTRAST = CMAP(0.48)

AGE_COLORS = {
    "18–39": CMAP(0.10),
    "40–49": CMAP(0.24),
    "50–59": CMAP(0.40),
    "60–69": CMAP(0.56),
    "70–79": CMAP(0.72),
    "≥80": CMAP(0.88),
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 12.5,

        "axes.titlesize": 15,
        "axes.titleweight": "bold",

        "axes.labelsize": 14,
        "axes.labelweight": "bold",

        "xtick.labelsize": 12,
        "ytick.labelsize": 12,

        "legend.fontsize": 11,

        "axes.spines.top": False,
        "axes.spines.right": False,

        "figure.dpi": 120,
        "savefig.dpi": 600,

        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


# ==========================================================================================
# FIGURE
# ==========================================================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(15.8, 11.4),
    gridspec_kw={
        "width_ratios": [
            1.00,
            1.18,
        ],
        "height_ratios": [
            0.96,
            1.14,
        ],
    },
)

axA, axB = axes[0]
axC, axD = axes[1]


# ==========================================================================================
# A — OVERALL SINGLE-LEAD ABLATION
# ==========================================================================================

plotA = (
    lead.iloc[::-1]
    .reset_index(drop=True)
)

yA = np.arange(len(plotA))

valA = plotA["AUROC_drop"].to_numpy(float)
loA = plotA["AUROC_drop_CI_low"].to_numpy(float)
hiA = plotA["AUROC_drop_CI_high"].to_numpy(float)

colorsA = [
    C_PRE
    if region == "Precordial"
    else C_LIMB
    for region in plotA["region"]
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
            valA - loA,
            hiA - valA,
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

axA.set_yticks(yA)

axA.set_yticklabels(
    plotA["lead"],
    fontsize=12,
    fontweight="bold",
)

axA.set_xlabel(
    "AUROC decrease after lead removal"
)

axA.set_title(
    "Lead importance by single-lead ablation",
    pad=8,
)

axA.grid(
    axis="x",
    alpha=0.14,
)

axA.set_xlim(
    min(
        -0.0005,
        float(loA.min()) - 0.0003,
    ),
    float(hiA.max()) + 0.0037,
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
        fontsize=11,
    )


# ==========================================================================================
# B — PRECORDIAL VS LIMB SUMMARY
# ==========================================================================================

pre = (
    lead.loc[
        lead["region"].eq("Precordial"),
        "AUROC_drop",
    ]
    .to_numpy(float)
)

limb = (
    lead.loc[
        lead["region"].eq("Limb"),
        "AUROC_drop",
    ]
    .to_numpy(float)
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

bp["boxes"][0].set_facecolor(C_PRE)
bp["boxes"][1].set_facecolor(C_LIMB)

for box in bp["boxes"]:
    box.set_alpha(0.80)

rng = np.random.default_rng(42)

for xpos, vals, color in [
    (1, pre, C_PRE),
    (2, limb, C_LIMB),
]:
    jitter = rng.uniform(
        -0.075,
        0.075,
        size=len(vals),
    )

    axB.scatter(
        xpos + jitter,
        vals,
        s=46,
        color=color,
        edgecolor="black",
        linewidth=0.45,
        zorder=3,
    )

for tick in axB.get_xticklabels():
    tick.set_fontsize(12)
    tick.set_fontweight("bold")

axB.set_ylabel(
    "AUROC decrease\nafter lead removal"
)

axB.set_title(
    "Regional lead importance",
    pad=8,
)

axB.grid(
    axis="y",
    alpha=0.14,
)


# ==========================================================================================
# C — VCG MECHANISTIC ABLATION
# ==========================================================================================

plotC = (
    mech.iloc[::-1]
    .reset_index(drop=True)
)

yC = np.arange(len(plotC))

valC = plotC["AUROC_drop"].to_numpy(float)
loC = plotC["AUROC_drop_CI_low"].to_numpy(float)
hiC = plotC["AUROC_drop_CI_high"].to_numpy(float)

colorsC = [
    CMAP(x)
    for x in np.linspace(
        0.32,
        0.86,
        len(plotC),
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
            valC - loC,
            hiC - valC,
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

axC.set_yticks(yC)

axC.set_yticklabels(
    plotC["display"],
    fontsize=12,
    fontweight="bold",
)

for tick in axC.get_yticklabels():
    tick.set_linespacing(1.05)

axC.set_xlabel(
    "AUROC decrease after component ablation"
)

axC.set_title(
    "VCG component ablation",
    pad=8,
)

axC.grid(
    axis="x",
    alpha=0.14,
)

axC.set_xlim(
    min(
        -0.0005,
        float(loC.min()) - 0.00025,
    ),
    float(hiC.max()) + 0.0030,
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
            if abs(point) < 0.001
            else f"{point:.4f}"
        ),
        va="center",
        ha="left",
        fontsize=11,
    )


# ==========================================================================================
# D — CURRENT AGE-STRATIFIED REGIONAL LEAD ABLATION
# ==========================================================================================

# One y position per age group, with separate offsets for:
#   Precordial removal
#   Limb removal
#   Precordial-minus-limb contrast
base_y = {
    age_group: i
    for i, age_group in enumerate(AGE_LABELS)
}

offsets = {
    "Precordial": -0.22,
    "Limb": 0.00,
    "Contrast": +0.22,
}

region_colors = {
    "Precordial": C_PRE,
    "Limb": C_LIMB,
    "Contrast": C_CONTRAST,
}

region_labels = {
    "Precordial": "All precordial leads removed",
    "Limb": "All limb leads removed",
    "Contrast": "Precordial − limb contrast",
}

for region_name in [
    "Precordial",
    "Limb",
    "Contrast",
]:
    sub = age_reg.loc[
        age_reg["region_display"].eq(region_name)
    ].copy()

    sub["_age_order"] = (
        sub["age_group"]
        .map(age_order)
    )

    sub = (
        sub
        .sort_values("_age_order")
        .reset_index(drop=True)
    )

    x = sub["metric_drop"].to_numpy(float)
    lo = sub["drop_CI_low"].to_numpy(float)
    hi = sub["drop_CI_high"].to_numpy(float)

    y = np.array(
        [
            base_y[a] + offsets[region_name]
            for a in sub["age_group"]
        ],
        dtype=float,
    )

    axD.errorbar(
        x,
        y,
        xerr=np.vstack(
            [
                x - lo,
                hi - x,
            ]
        ),
        fmt="o",
        markersize=6.0,
        linewidth=1.15,
        capsize=2.5,
        color=region_colors[region_name],
        ecolor=region_colors[region_name],
        markeredgecolor="black",
        markeredgewidth=0.45,
        label=region_labels[region_name],
        zorder=3,
    )

axD.axvline(
    0,
    color="0.6",
    linewidth=0.8,
)

axD.set_yticks(
    np.arange(len(AGE_LABELS))
)

axD.set_yticklabels(
    AGE_LABELS,
    fontsize=12,
    fontweight="bold",
)

# Display age strata in natural manuscript order from youngest to oldest.
axD.invert_yaxis()

axD.set_xlabel(
    "AUROC decrease / precordial–limb contrast"
)

axD.set_title(
    "Age-stratified regional lead importance",
    pad=30,
)

axD.grid(
    axis="x",
    alpha=0.14,
)

# Use all current CI bounds to define a safe x range.
d_lo = float(
    age_reg["drop_CI_low"].min()
)

d_hi = float(
    age_reg["drop_CI_high"].max()
)

d_span = max(
    d_hi - d_lo,
    0.01,
)

axD.set_xlim(
    min(
        -0.01,
        d_lo - 0.04 * d_span,
    ),
    d_hi + 0.10 * d_span,
)

legendD = axD.legend(
    frameon=False,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.005),
    ncol=3,
    fontsize=9.5,
    handletextpad=0.45,
    columnspacing=1.05,
    borderaxespad=0.0,
)

# Keep the panel itself free of annotation text so no data points are obscured.


# ==========================================================================================
# PANEL LABELS
# ==========================================================================================

for label, ax in zip(
    ["A", "B", "C", "D"],
    [axA, axB, axC, axD],
):
    ax.text(
        -0.12,
        1.055,
        label,
        transform=ax.transAxes,
        fontsize=20,
        fontweight="bold",
        ha="left",
        va="top",
    )


# ==========================================================================================
# SOURCE DATA
# ==========================================================================================

def block(df, panel, source):
    out = df.copy()
    out.insert(0, "panel", panel)
    out.insert(1, "source", source)
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
            "Descriptive precordial-vs-limb single-lead summary",
        ),
        block(
            mech,
            "C",
            "Final VCG mechanistic ablation",
        ),
        block(
            age_reg.drop(
                columns=[
                    "_age_order",
                    "_region_order",
                ],
                errors="ignore",
            ),
            "D",
            "Current age-stratified grouped regional lead ablation",
        ),
    ],
    ignore_index=True,
    sort=False,
).to_csv(
    SOURCE_OUT,
    index=False,
)


# ==========================================================================================
# TITLE / SAVE
# ==========================================================================================

fig.suptitle(
    "Lead- and VCG-Level Interpretability of the VCG-Geometry ResNet",
    fontsize=18,
    fontweight="bold",
    y=0.985,
)

fig.subplots_adjust(
    left=0.16,
    right=0.988,
    bottom=0.08,
    top=0.90,
    wspace=0.30,
    hspace=0.36,
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

print("\nDisplaying Figure 5...")
plt.show()


# ==========================================================================================
# SUMMARY
# ==========================================================================================

print("\n" + "=" * 120)
print("FIGURE 5 CURRENT RUN COMPLETE")
print("=" * 120)

print("\nPanel A — overall lead importance")
print(
    lead[
        [
            "lead",
            "region",
            "AUROC_drop",
            "AUROC_drop_CI_low",
            "AUROC_drop_CI_high",
        ]
    ].to_string(index=False)
)

print("\nPanel B — regional summary")
print(
    regional.to_string(index=False)
)

print("\nPanel C — VCG mechanistic ablation")
print(
    mech[
        [
            "display",
            "AUROC_drop",
            "AUROC_drop_CI_low",
            "AUROC_drop_CI_high",
        ]
    ].to_string(index=False)
)

print("\nPanel D — CURRENT age-stratified grouped regional lead ablation")
print(
    age_reg[
        [
            "age_group",
            "region_display",
            "N",
            "Female_N",
            "Male_N",
            "intact_metric",
            "ablated_metric",
            "metric_drop",
            "drop_CI_low",
            "drop_CI_high",
        ]
    ].to_string(index=False)
)

print("\nSaved:")
print(PNG_OUT)
print(PDF_OUT)
print(SVG_OUT)
print(SOURCE_OUT)

print("\nDONE")
