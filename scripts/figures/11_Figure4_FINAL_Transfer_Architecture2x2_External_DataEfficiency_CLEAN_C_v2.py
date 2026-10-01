#!/usr/bin/env python3
# ==========================================================================================
# ECG-SEX — REVISED FIGURE 4
# TRANSFER LEARNING, ARCHITECTURE × PRETRAINING, EXTERNAL GENERALIZATION, DATA EFFICIENCY
# 3 × 2 layout
# ==========================================================================================
#
# PANELS
# ------
# A  WCMC/NYP locked-test ROC:
#      CNN-TCN-Transformer — MIMIC-only frozen
#      CNN-TCN-Transformer — WCMC-only
#      CNN-TCN-Transformer — MIMIC→WCMC
#      ResNet1D — MIMIC→WCMC
#      VCG-Geometry ResNet — MIMIC→WCMC
#
# B  WCMC/NYP locked-test precision–recall curves for the same five models.
#
# C  NEW reviewer analysis: matched 2 × 2 architecture × pretraining comparison
#      ResNet1D — WCMC-only
#      VCG-Geometry ResNet — WCMC-only
#      ResNet1D — MIMIC→WCMC
#      VCG-Geometry ResNet — MIMIC→WCMC
#    Shows AUROC and AUPRC with 95% paired-bootstrap CIs.
#    Annotates the VCG-minus-ResNet architecture effect both without and with pretraining.
#
# D  PTB-XL zero-shot matched ResNet1D vs VCG-Geometry ResNet.
# E  Chapman/Ningbo zero-shot matched ResNet1D vs VCG-Geometry ResNet.
# F  PTB-XL VCG-Geometry ResNet data-efficiency analysis.
#
# No model training or inference is performed. Figure only.
# ==========================================================================================

from pathlib import Path
import os

import numpy as np
import pandas as pd
import matplotlib

# ------------------------------------------------------------------------------------------
# NOTEBOOK / TERMINAL DISPLAY HANDLING
# ------------------------------------------------------------------------------------------
# In Jupyter/IPYNB, do NOT force the Agg backend. The inline notebook backend can render
# figures even when the remote SCU node has no graphical DISPLAY.
#
# In Slurm or a plain headless shell, use Agg so the script can still save PNG/PDF/SVG safely.

def _running_in_jupyter():
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is None:
            return False
        return (
            "IPKernelApp" in getattr(ip, "config", {})
            or ip.__class__.__name__ == "ZMQInteractiveShell"
        )
    except Exception:
        return False

IN_JUPYTER = _running_in_jupyter()

if (not IN_JUPYTER) and (
    os.environ.get("SLURM_JOB_ID")
    or not os.environ.get("DISPLAY")
):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score


# ==========================================================================================
# PATHS
# ==========================================================================================

RUN_DIR = Path(
    "/athena/madelab/scratch/iqh4001/ECG/ECG_Echo/Runs/"
    "Master_ECG_Sex_MIMIC_EchoNext/"
    "20260609_020154_MASTER_ECG_SEX_MIMIC_ECHONEXT"
)
WCMC_ROOT = RUN_DIR / "wcmc_ecgsex_from_scratch"

# Original transfer-learning benchmark predictions.
OLD_PRED = {
    "CNN-TCN-Transformer — MIMIC-only frozen":
        WCMC_ROOT / "04_wcmc_finetune_from_mimic/predictions/"
        "WCMC_ECGSex_initial_MIMICPretrain_locked_test_predictions.csv",

    "CNN-TCN-Transformer — WCMC-only":
        WCMC_ROOT / "05_wcmc_from_scratch/predictions/"
        "WCMC_ECGSex_from_scratch_locked_test_predictions.csv",

    "CNN-TCN-Transformer — MIMIC→WCMC":
        WCMC_ROOT / "04_wcmc_finetune_from_mimic/predictions/"
        "WCMC_ECGSex_finetuned_locked_test_predictions.csv",
}

# Final transferred ResNet / VCG ensemble.
WCMC_FINAL = (
    WCMC_ROOT
    / "19_vcg_geometry_resnet_5seed_locked_wcmc"
    / "final_5seed_ensemble_bootstrap"
)

WCMC_FINAL_PRED = (
    WCMC_FINAL / "WCMC_LockedTest_Aligned_5SeedEnsemble_Predictions.csv"
)
WCMC_FINAL_PERF = (
    WCMC_FINAL / "WCMC_VCG_vs_ResNet_5SeedEnsemble_Performance_With95CI.csv"
)

# NEW: completed WCMC-only vs transfer 2 × 2 reviewer analysis.
TWO_BY_TWO = (
    WCMC_ROOT
    / "21_wcmc_only_vcg_resnet_5seed_2x2"
    / "final_5seed_2x2_bootstrap"
)

TWO_BY_TWO_PERF = (
    TWO_BY_TWO / "WCMC_2x2_ArchitectureTransfer_Performance.csv"
)
TWO_BY_TWO_DIFF = (
    TWO_BY_TWO / "WCMC_2x2_PairedDifferences.csv"
)
TWO_BY_TWO_BOOT = (
    TWO_BY_TWO / "WCMC_2x2_PairedBootstrap_Distributions.csv"
)

# External zero-shot confirmation.
EXT_ROOT = WCMC_ROOT / "20_vcg_geometry_resnet_external_confirmation"
EXT_FINAL = EXT_ROOT / "final_paired_bootstrap"

EXT_PERF = (
    EXT_FINAL / "VCGGeometryResNet_ExternalConfirmation_Combined_Performance.csv"
)
EXT_DIFF = (
    EXT_FINAL / "VCGGeometryResNet_ExternalConfirmation_Combined_PairedDifferences.csv"
)
PTB_BOOT = EXT_FINAL / "ptbxl_VCG_vs_ResNet_bootstrap_2000.csv"
CHAPMAN_BOOT = EXT_FINAL / "chapman_ningbo_VCG_vs_ResNet_bootstrap_2000.csv"

# PTB-XL VCG data efficiency.
DE_FINAL = (
    RUN_DIR
    / "ptbxl_ecgsex_from_scratch"
    / "12_vcg_geometry_data_efficiency"
    / "final_seedensemble_bootstrap"
)

DE_PERF = (
    DE_FINAL / "PTBXL_VCG_DataEfficiency_SeedEnsemble_Performance_With95CI.csv"
)
DE_COMP = (
    DE_FINAL / "PTBXL_VCG_DataEfficiency_PairedComparisons.csv"
)

# Output.
OUTDIR = RUN_DIR / "publication_figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

STEM = OUTDIR / "Figure4_REVISED_Transfer_Architecture2x2_External_DataEfficiency_CLEAN_C_v2"
PNG_OUT = Path(str(STEM) + ".png")
PDF_OUT = Path(str(STEM) + ".pdf")
SVG_OUT = Path(str(STEM) + ".svg")
CSV_OUT = Path(str(STEM) + "_SourceData.csv")


# ==========================================================================================
# FILE AUDIT
# ==========================================================================================

required = (
    list(OLD_PRED.values())
    + [
        WCMC_FINAL_PRED,
        WCMC_FINAL_PERF,
        TWO_BY_TWO_PERF,
        TWO_BY_TWO_DIFF,
        TWO_BY_TWO_BOOT,
        EXT_PERF,
        EXT_DIFF,
        PTB_BOOT,
        CHAPMAN_BOOT,
        DE_PERF,
        DE_COMP,
    ]
)

missing = [p for p in required if not p.exists()]
if missing:
    raise FileNotFoundError(
        "Missing required files:\n" + "\n".join(str(p) for p in missing)
    )


# ==========================================================================================
# HELPERS
# ==========================================================================================

def find_col(cols, candidates):
    lower = {str(c).lower(): c for c in cols}

    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]

    for c in cols:
        lc = str(c).lower()
        for cand in candidates:
            if cand.lower() in lc:
                return c

    return None


def load_pred(path):
    df = pd.read_csv(path, low_memory=False)

    ycol = find_col(
        df.columns,
        ["sex_true_binary", "y_true", "true_label", "sex_binary", "label", "target"],
    )
    pcol = find_col(
        df.columns,
        ["prob_male", "pred_prob", "y_prob", "probability", "prob", "score"],
    )

    if ycol is None or pcol is None:
        raise RuntimeError(
            f"Cannot identify label/probability columns in {path}\n"
            f"Columns: {list(df.columns)}"
        )

    out = df[[ycol, pcol]].copy()
    out.columns = ["y_true", "prob"]
    out["y_true"] = pd.to_numeric(out["y_true"], errors="raise").astype(int)
    out["prob"] = pd.to_numeric(out["prob"], errors="raise")
    out = out.dropna().reset_index(drop=True)

    if set(out["y_true"].unique()) != {0, 1}:
        raise RuntimeError(f"Invalid binary labels in {path}")

    return out


def canon_final_model(name):
    s = str(name).lower()

    if "vcg" in s and "resnet" in s:
        return "VCG-Geometry ResNet"

    if "resnet" in s and "vcg" not in s:
        return "ResNet1D"

    return str(name)


def paired_delta_ci_text(row, delta_col, label):
    delta = float(row[delta_col])

    ci_pairs = [
        ("CI_low", "CI_high"),
        ("ci_low", "ci_high"),
        ("delta_CI_low", "delta_CI_high"),
        ("delta_ci_low", "delta_ci_high"),
        ("CI95_low", "CI95_high"),
        ("lower", "upper"),
    ]

    for lo_col, hi_col in ci_pairs:
        if lo_col in row.index and hi_col in row.index:
            lo = float(row[lo_col])
            hi = float(row[hi_col])
            return f"{label} {delta:+.4f} [{lo:+.4f}, {hi:+.4f}]"

    raise RuntimeError(
        "Could not identify CI columns. "
        f"Available: {list(row.index)}"
    )


def require_bootstrap_vector(df, column, label):
    if column not in df.columns:
        raise RuntimeError(
            f"Missing real bootstrap column '{column}' for {label}."
        )

    vals = pd.to_numeric(df[column], errors="coerce").dropna().to_numpy(float)

    if vals.size < 1000:
        raise RuntimeError(
            f"Insufficient bootstrap replicates for {label}: {vals.size:,}"
        )

    if not np.isfinite(vals).all():
        raise RuntimeError(f"Non-finite bootstrap values for {label}.")

    return vals


def draw_violin(ax, values, xpos, color, width):
    vp = ax.violinplot(
        [values],
        positions=[xpos],
        widths=width,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        points=200,
        bw_method="scott",
    )

    for body in vp["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_linewidth(0.70)
        body.set_alpha(0.46)

    return vp


def block(df, panel, source):
    out = df.copy()
    out.insert(0, "panel", panel)
    out.insert(1, "source", source)
    return out


# ==========================================================================================
# PANEL A/B — WCMC TRANSFER BENCHMARK CURVES
# ==========================================================================================

preds = {name: load_pred(path) for name, path in OLD_PRED.items()}

fraw = pd.read_csv(WCMC_FINAL_PRED, low_memory=False)
for c in ["sex_true_binary", "prob_resnet", "prob_vcg"]:
    if c not in fraw.columns:
        raise RuntimeError(f"Missing {c} in {WCMC_FINAL_PRED}")

preds["ResNet1D — MIMIC→WCMC"] = pd.DataFrame(
    {
        "y_true": pd.to_numeric(
            fraw["sex_true_binary"], errors="raise"
        ).astype(int),
        "prob": pd.to_numeric(fraw["prob_resnet"], errors="raise"),
    }
)

preds["VCG-Geometry ResNet — MIMIC→WCMC"] = pd.DataFrame(
    {
        "y_true": pd.to_numeric(
            fraw["sex_true_binary"], errors="raise"
        ).astype(int),
        "prob": pd.to_numeric(fraw["prob_vcg"], errors="raise"),
    }
)

# Historical transfer benchmark values retained exactly as in the manuscript.
old_perf = pd.DataFrame(
    [
        dict(
            model="CNN-TCN-Transformer — MIMIC-only frozen",
            AUROC=0.904,
            AUROC_CI_low=0.901,
            AUROC_CI_high=0.908,
            AUPRC=0.896,
            AUPRC_CI_low=0.891,
            AUPRC_CI_high=0.902,
        ),
        dict(
            model="CNN-TCN-Transformer — WCMC-only",
            AUROC=0.929,
            AUROC_CI_low=0.926,
            AUROC_CI_high=0.933,
            AUPRC=0.923,
            AUPRC_CI_low=0.919,
            AUPRC_CI_high=0.927,
        ),
        dict(
            model="CNN-TCN-Transformer — MIMIC→WCMC",
            AUROC=0.944,
            AUROC_CI_low=0.942,
            AUROC_CI_high=0.947,
            AUPRC=0.940,
            AUPRC_CI_low=0.936,
            AUPRC_CI_high=0.943,
        ),
    ]
)

fperf = pd.read_csv(WCMC_FINAL_PERF, low_memory=False)
fperf["canonical_model"] = fperf["model"].map(canon_final_model)

final_rows = []

for canonical, display in [
    ("ResNet1D", "ResNet1D — MIMIC→WCMC"),
    ("VCG-Geometry ResNet", "VCG-Geometry ResNet — MIMIC→WCMC"),
]:
    sub = fperf.loc[fperf["canonical_model"].eq(canonical)]

    if len(sub) != 1:
        raise RuntimeError(
            f"Expected one final WCMC row for {canonical}; found {len(sub)}"
        )

    r = sub.iloc[0]

    final_rows.append(
        dict(
            model=display,
            AUROC=float(r["AUROC"]),
            AUROC_CI_low=float(r["AUROC_CI_low"]),
            AUROC_CI_high=float(r["AUROC_CI_high"]),
            AUPRC=float(r["AUPRC"]),
            AUPRC_CI_low=float(r["AUPRC_CI_low"]),
            AUPRC_CI_high=float(r["AUPRC_CI_high"]),
        )
    )

perf5 = pd.concat(
    [old_perf, pd.DataFrame(final_rows)],
    ignore_index=True,
)

MODEL_ORDER = [
    "CNN-TCN-Transformer — MIMIC-only frozen",
    "CNN-TCN-Transformer — WCMC-only",
    "CNN-TCN-Transformer — MIMIC→WCMC",
    "ResNet1D — MIMIC→WCMC",
    "VCG-Geometry ResNet — MIMIC→WCMC",
]

audit_rows = []

for model in MODEL_ORDER:
    df = preds[model]

    if len(df) != 21374:
        raise RuntimeError(
            f"{model}: expected N=21,374; got {len(df):,}"
        )

    auc = roc_auc_score(df["y_true"], df["prob"])
    ap = average_precision_score(df["y_true"], df["prob"])

    saved = perf5.loc[perf5["model"].eq(model)].iloc[0]
    tol = 0.0025 if model.startswith("CNN-TCN") else 5e-5

    if (
        abs(auc - float(saved["AUROC"])) > tol
        or abs(ap - float(saved["AUPRC"])) > tol
    ):
        raise RuntimeError(
            f"Metric audit failed for {model}: "
            f"observed={auc:.6f}/{ap:.6f}; "
            f"saved={float(saved['AUROC']):.6f}/{float(saved['AUPRC']):.6f}"
        )

    audit_rows.append(
        {
            "model": model,
            "N": len(df),
            "observed_AUROC": auc,
            "observed_AUPRC": ap,
        }
    )

audit_df = pd.DataFrame(audit_rows)


# ==========================================================================================
# PANEL C — NEW 2 × 2 ARCHITECTURE × PRETRAINING ANALYSIS
# ==========================================================================================

two_perf = pd.read_csv(TWO_BY_TWO_PERF, low_memory=False)
two_diff = pd.read_csv(TWO_BY_TWO_DIFF, low_memory=False)
two_boot = pd.read_csv(TWO_BY_TWO_BOOT, low_memory=False)

TWO_KEYS = [
    "Local_ResNet",
    "Local_VCG",
    "Transfer_ResNet",
    "Transfer_VCG",
]

TWO_LABELS = {
    "Local_ResNet": "ResNet1D\nWCMC-only",
    "Local_VCG": "VCG-Geometry\nWCMC-only",
    "Transfer_ResNet": "ResNet1D\nMIMIC→WCMC",
    "Transfer_VCG": "VCG-Geometry\nMIMIC→WCMC",
}

TWO_SHORT = {
    "Local_ResNet": "ResNet local",
    "Local_VCG": "VCG local",
    "Transfer_ResNet": "ResNet transfer",
    "Transfer_VCG": "VCG transfer",
}

if set(TWO_KEYS) - set(two_perf["model_key"]):
    raise RuntimeError(
        "2x2 performance table is missing one or more required model keys."
    )

two_perf = (
    two_perf
    .set_index("model_key")
    .loc[TWO_KEYS]
    .reset_index()
)

# Architecture contrasts.
local_arch = two_diff.loc[
    two_diff["family"].eq("Architecture effect without pretraining")
    & two_diff["metric"].eq("AUROC")
]

transfer_arch = two_diff.loc[
    two_diff["family"].eq("Architecture effect after MIMIC pretraining")
    & two_diff["metric"].eq("AUROC")
]

if len(local_arch) != 1 or len(transfer_arch) != 1:
    raise RuntimeError(
        "Could not identify the two primary architecture-effect AUROC rows."
    )

local_arch = local_arch.iloc[0]
transfer_arch = transfer_arch.iloc[0]


# ==========================================================================================
# EXTERNAL / DATA EFFICIENCY
# ==========================================================================================

ext_perf = pd.read_csv(EXT_PERF, low_memory=False)
ext_diff = pd.read_csv(EXT_DIFF, low_memory=False)
ptb_boot = pd.read_csv(PTB_BOOT, low_memory=False)
chapman_boot = pd.read_csv(CHAPMAN_BOOT, low_memory=False)

de_perf = pd.read_csv(DE_PERF, low_memory=False)
de_comp = pd.read_csv(DE_COMP, low_memory=False)

ext_perf["canonical_model"] = ext_perf["model"].map(canon_final_model)


def external_boot_values(dataset, model, metric):
    boot = ptb_boot if dataset == "PTB-XL" else chapman_boot
    prefix = "VCG" if model == "VCG-Geometry ResNet" else "ResNet"

    return require_bootstrap_vector(
        boot,
        f"{prefix}__{metric}",
        f"{dataset} {model} {metric}",
    )


# ==========================================================================================
# STYLE
# ==========================================================================================

cmap = plt.get_cmap("plasma")

COLORS = {
    "CNN-TCN-Transformer — MIMIC-only frozen": cmap(0.08),
    "CNN-TCN-Transformer — WCMC-only": cmap(0.24),
    "CNN-TCN-Transformer — MIMIC→WCMC": cmap(0.43),
    "ResNet1D — MIMIC→WCMC": cmap(0.68),
    "VCG-Geometry ResNet — MIMIC→WCMC": cmap(0.90),

    "Local_ResNet": cmap(0.20),
    "Local_VCG": cmap(0.43),
    "Transfer_ResNet": cmap(0.68),
    "Transfer_VCG": cmap(0.90),
}

LINESTYLES = {
    "CNN-TCN-Transformer — MIMIC-only frozen": ":",
    "CNN-TCN-Transformer — WCMC-only": "--",
    "CNN-TCN-Transformer — MIMIC→WCMC": "-.",
    "ResNet1D — MIMIC→WCMC": "--",
    "VCG-Geometry ResNet — MIMIC→WCMC": "-",
}

LINEWIDTHS = {
    "CNN-TCN-Transformer — MIMIC-only frozen": 1.45,
    "CNN-TCN-Transformer — WCMC-only": 1.55,
    "CNN-TCN-Transformer — MIMIC→WCMC": 1.70,
    "ResNet1D — MIMIC→WCMC": 2.00,
    "VCG-Geometry ResNet — MIMIC→WCMC": 2.60,
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 12.5,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.labelsize": 14,
        "axes.labelweight": "bold",
        "xtick.labelsize": 11.5,
        "ytick.labelsize": 12,
        "legend.fontsize": 10.2,
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

fig, axes = plt.subplots(3, 2, figsize=(14.2, 15.8))
axA, axB = axes[0]
axC, axD = axes[1]
axE, axF = axes[2]


# ------------------------------------------------------------------------------------------
# A — WCMC ROC
# ------------------------------------------------------------------------------------------

roc_handles = {}

for model in MODEL_ORDER:
    df = preds[model]
    fpr, tpr, _ = roc_curve(df["y_true"], df["prob"])
    r = perf5.loc[perf5["model"].eq(model)].iloc[0]

    h, = axA.plot(
        fpr,
        tpr,
        color=COLORS[model],
        linestyle=LINESTYLES[model],
        linewidth=LINEWIDTHS[model],
        label=(
            f"{model}: "
            f"{float(r['AUROC']):.3f} "
            f"({float(r['AUROC_CI_low']):.3f}–{float(r['AUROC_CI_high']):.3f})"
        ),
    )

    roc_handles[model] = h

axA.plot([0, 1], [0, 1], "--", color="0.76", linewidth=0.9)
axA.set(
    xlim=(0, 1),
    ylim=(0, 1.015),
    xlabel="False-positive rate",
    ylabel="True-positive rate",
)
axA.set_title("WCMC/NYP locked-test ROC")
axA.grid(alpha=0.14)
axA.legend(
    [roc_handles[m] for m in MODEL_ORDER],
    [roc_handles[m].get_label() for m in MODEL_ORDER],
    frameon=False,
    loc="lower right",
    fontsize=8.8,
)


# ------------------------------------------------------------------------------------------
# B — WCMC PR
# ------------------------------------------------------------------------------------------

pr_handles = {}
prevalence = float(preds[MODEL_ORDER[0]]["y_true"].mean())

for model in MODEL_ORDER:
    df = preds[model]
    precision, recall, _ = precision_recall_curve(
        df["y_true"],
        df["prob"],
    )
    r = perf5.loc[perf5["model"].eq(model)].iloc[0]

    h, = axB.plot(
        recall,
        precision,
        color=COLORS[model],
        linestyle=LINESTYLES[model],
        linewidth=LINEWIDTHS[model],
        label=(
            f"{model}: "
            f"{float(r['AUPRC']):.3f} "
            f"({float(r['AUPRC_CI_low']):.3f}–{float(r['AUPRC_CI_high']):.3f})"
        ),
    )

    pr_handles[model] = h

axB.axhline(
    prevalence,
    linestyle=":",
    color="0.72",
    linewidth=1.0,
)
axB.set(
    xlim=(0, 1),
    ylim=(0.45, 1.01),
    xlabel="Recall",
    ylabel="Precision",
)
axB.set_title("WCMC/NYP locked-test precision–recall")
axB.grid(alpha=0.14)
axB.legend(
    [pr_handles[m] for m in MODEL_ORDER],
    [pr_handles[m].get_label() for m in MODEL_ORDER],
    frameon=False,
    loc="lower left",
    fontsize=8.8,
)


# ------------------------------------------------------------------------------------------
# C — NEW MATCHED 2 × 2 ARCHITECTURE × PRETRAINING COMPARISON
#     CLEAN PRIMARY-AUROC PRESENTATION
# ------------------------------------------------------------------------------------------
#
# The previous violin version was visually crowded and the paired-difference annotations
# could extend outside the panel. This panel now shows the primary reviewer comparison
# directly: AUROC for ResNet1D versus VCG-Geometry ResNet under WCMC-only training and
# after MIMIC-IV pretraining. Point estimates and 95% bootstrap CIs are shown, with the
# paired VCG-minus-ResNet ΔAUROC printed above each training strategy.
# v2 visual fix: both comparison annotations are placed in a shared upper band so they
# do not overlap point estimates or confidence-interval labels.
#
# AUPRC remains reported in Panels B / source data and in the supplementary 2×2 table.

x_centers = np.array([0.0, 1.0])
x_labels = ["WCMC-only", "MIMIC→WCMC"]
arch_offsets = {
    "ResNet1D": -0.12,
    "VCG-Geometry ResNet": +0.12,
}
arch_colors = {
    "ResNet1D": COLORS["Local_ResNet"],
    "VCG-Geometry ResNet": COLORS["Transfer_VCG"],
}

pair_keys = {
    ("WCMC-only", "ResNet1D"): "Local_ResNet",
    ("WCMC-only", "VCG-Geometry ResNet"): "Local_VCG",
    ("MIMIC→WCMC", "ResNet1D"): "Transfer_ResNet",
    ("MIMIC→WCMC", "VCG-Geometry ResNet"): "Transfer_VCG",
}

panel_c_rows = []

for xi, strategy in zip(x_centers, x_labels):
    for architecture in ["ResNet1D", "VCG-Geometry ResNet"]:
        key = pair_keys[(strategy, architecture)]
        r = two_perf.loc[two_perf["model_key"].eq(key)].iloc[0]

        point = float(r["AUROC"])
        lo = float(r["AUROC_CI_low"])
        hi = float(r["AUROC_CI_high"])
        xpos = xi + arch_offsets[architecture]

        axC.errorbar(
            xpos,
            point,
            yerr=np.array([[point - lo], [hi - point]]),
            fmt="o",
            markersize=8.0,
            capsize=4.0,
            linewidth=1.6,
            color=arch_colors[architecture],
            markeredgecolor="black",
            markeredgewidth=0.75,
            zorder=5,
        )

        axC.text(
            xpos,
            hi + 0.00035,
            f"{point:.3f}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            zorder=6,
        )

        panel_c_rows.append(
            {
                "strategy": strategy,
                "architecture": architecture,
                "model_key": key,
                "AUROC": point,
                "AUROC_CI_low": lo,
                "AUROC_CI_high": hi,
            }
        )

# Light connector within each matched architecture comparison.
for xi, strategy in zip(x_centers, x_labels):
    rr = pd.DataFrame(panel_c_rows)
    sub = rr.loc[rr["strategy"].eq(strategy)]

    resnet_y = float(
        sub.loc[sub["architecture"].eq("ResNet1D"), "AUROC"].iloc[0]
    )
    vcg_y = float(
        sub.loc[sub["architecture"].eq("VCG-Geometry ResNet"), "AUROC"].iloc[0]
    )

    axC.plot(
        [
            xi + arch_offsets["ResNet1D"],
            xi + arch_offsets["VCG-Geometry ResNet"],
        ],
        [resnet_y, vcg_y],
        color="0.55",
        linewidth=1.0,
        zorder=2,
    )


def add_pair_delta(ax, row, center_x, y):
    delta = float(row["point_delta"])
    lo = float(row["CI_low"])
    hi = float(row["CI_high"])

    x1 = center_x + arch_offsets["ResNet1D"]
    x2 = center_x + arch_offsets["VCG-Geometry ResNet"]

    ax.plot(
        [x1, x1, x2, x2],
        [y, y + 0.00028, y + 0.00028, y],
        color="black",
        linewidth=0.85,
        zorder=7,
    )

    ax.text(
        center_x,
        y + 0.00050,
        f"ΔAUROC {delta:+.4f}\n95% CI {lo:+.4f} to {hi:+.4f}",
        ha="center",
        va="bottom",
        fontsize=8.8,
        zorder=8,
    )


# Keep both architecture-effect annotations in the same upper band.
# This prevents the MIMIC→WCMC annotation from colliding with the VCG point/CI label.
PAIR_ANNOTATION_Y = 0.9580

add_pair_delta(
    axC,
    local_arch,
    x_centers[0],
    PAIR_ANNOTATION_Y,
)

add_pair_delta(
    axC,
    transfer_arch,
    x_centers[1],
    PAIR_ANNOTATION_Y,
)

axC.set_xticks(x_centers)
axC.set_xticklabels(
    x_labels,
    fontsize=11.5,
    fontweight="bold",
)
axC.set_xlim(-0.42, 1.42)
axC.set_ylim(0.944, 0.9612)
axC.set_ylabel("AUROC")
axC.set_title("Architecture effect with and without MIMIC pretraining")
axC.grid(axis="y", alpha=0.14)

panel_c_handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="-",
        linewidth=1.0,
        markersize=6.5,
        color=arch_colors["ResNet1D"],
        markeredgecolor="black",
        markeredgewidth=0.5,
        label="ResNet1D",
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="-",
        linewidth=1.0,
        markersize=6.5,
        color=arch_colors["VCG-Geometry ResNet"],
        markeredgecolor="black",
        markeredgewidth=0.5,
        label="VCG-Geometry ResNet",
    ),
]

axC.legend(
    handles=panel_c_handles,
    frameon=False,
    loc="lower right",
    fontsize=9.2,
    ncol=1,
)

# ------------------------------------------------------------------------------------------
# D/E — ZERO-SHOT EXTERNAL MATCHED COMPARISON
# ------------------------------------------------------------------------------------------

def plot_external(ax, dataset):
    sub = ext_perf.loc[ext_perf["dataset"].eq(dataset)].copy()

    models = [
        "ResNet1D",
        "VCG-Geometry ResNet",
    ]

    centers = np.arange(2)
    external_offsets = {
        "ResNet1D": -0.16,
        "VCG-Geometry ResNet": +0.16,
    }

    all_values = []

    for metric_idx, metric_name in enumerate(["AUROC", "AUPRC"]):
        for model in models:
            rr = sub.loc[sub["canonical_model"].eq(model)]

            if len(rr) != 1:
                raise RuntimeError(
                    f"{dataset}: expected one row for {model}; found {len(rr)}."
                )

            r = rr.iloc[0]

            point = float(r[metric_name])
            lo = float(r[f"{metric_name}_CI_low"])
            hi = float(r[f"{metric_name}_CI_high"])

            vals = external_boot_values(
                dataset,
                model,
                metric_name,
            )
            all_values.append(vals)

            xpos = centers[metric_idx] + external_offsets[model]
            color_key = (
                "Transfer_VCG"
                if model == "VCG-Geometry ResNet"
                else "Transfer_ResNet"
            )

            draw_violin(
                ax,
                vals,
                xpos,
                COLORS[color_key],
                width=0.245,
            )

            ax.vlines(
                xpos,
                lo,
                hi,
                color="black",
                linewidth=1.55,
                zorder=4,
            )
            ax.scatter(
                [xpos],
                [point],
                s=32,
                facecolor="white",
                edgecolor="black",
                linewidth=0.9,
                zorder=5,
            )

            ax.text(
                xpos,
                hi + 0.0012,
                f"{point:.3f}",
                ha="center",
                va="bottom",
                fontsize=10.0,
            )

    concat = np.concatenate(all_values)

    ax.set_xticks(centers)
    ax.set_xticklabels(
        ["AUROC", "AUPRC"],
        fontweight="bold",
    )
    ax.set_ylim(
        max(0.875, float(np.nanmin(concat)) - 0.006),
        min(0.970, float(np.nanmax(concat)) + 0.010),
    )
    ax.set_ylabel("Bootstrap performance")
    ax.set_title(f"{dataset} zero-shot performance")
    ax.grid(axis="y", alpha=0.14)

    da = ext_diff.loc[
        ext_diff["dataset"].eq(dataset)
        & ext_diff["metric"].eq("AUROC")
    ].iloc[0]

    dp = ext_diff.loc[
        ext_diff["dataset"].eq(dataset)
        & ext_diff["metric"].eq("AUPRC")
    ].iloc[0]

    ax.text(
        0.02,
        0.97,
        "VCG − ResNet\n"
        + paired_delta_ci_text(
            da,
            "delta_VCG_minus_ResNet",
            "ΔAUROC",
        )
        + "\n"
        + paired_delta_ci_text(
            dp,
            "delta_VCG_minus_ResNet",
            "ΔAUPRC",
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.6,
        bbox=dict(
            boxstyle="round,pad=0.24",
            facecolor="white",
            edgecolor="none",
            alpha=0.90,
        ),
    )


plot_external(axD, "PTB-XL")
plot_external(axE, "Chapman/Ningbo")

external_handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        markersize=7.0,
        markerfacecolor=COLORS["Transfer_ResNet"],
        markeredgecolor="black",
        markeredgewidth=0.5,
        label="ResNet1D — MIMIC→WCMC",
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        markersize=7.0,
        markerfacecolor=COLORS["Transfer_VCG"],
        markeredgecolor="black",
        markeredgewidth=0.5,
        label="VCG-Geometry ResNet — MIMIC→WCMC",
    ),
]

axD.legend(
    handles=external_handles,
    frameon=False,
    loc="lower left",
    fontsize=8.7,
)
axE.legend(
    handles=external_handles,
    frameon=False,
    loc="lower left",
    fontsize=8.7,
)


# ------------------------------------------------------------------------------------------
# F — PTB-XL DATA EFFICIENCY
# ------------------------------------------------------------------------------------------

LOCAL = "PTB-XL-only VCG-Geometry ResNet"
TRANSFER = "MIMIC→WCMC→PTB-XL VCG-Geometry ResNet"

for strategy, color, ls, label in [
    (
        LOCAL,
        cmap(0.20),
        "--",
        "PTB-XL-only VCG",
    ),
    (
        TRANSFER,
        COLORS["Transfer_VCG"],
        "-",
        "MIMIC→WCMC→PTB-XL VCG",
    ),
]:
    sub = (
        de_perf.loc[de_perf["strategy"].eq(strategy)]
        .sort_values("train_percent")
    )

    if len(sub) != 4:
        raise RuntimeError(
            f"Expected four data-efficiency rows for {strategy}; found {len(sub)}"
        )

    xp = sub["train_percent"].to_numpy(float)
    vals = sub["AUROC"].to_numpy(float)
    lows = sub["AUROC_CI_low"].to_numpy(float)
    highs = sub["AUROC_CI_high"].to_numpy(float)

    axF.errorbar(
        xp,
        vals,
        yerr=np.vstack([vals - lows, highs - vals]),
        marker="o",
        markersize=6.2,
        linewidth=2.0,
        linestyle=ls,
        capsize=3.4,
        color=color,
        label=label,
    )

primary = de_comp.loc[
    de_comp["comparison_type"].eq("PRIMARY data-efficiency")
    & de_comp["metric"].eq("AUROC")
]

if len(primary) != 1:
    raise RuntimeError(
        "Expected exactly one PRIMARY data-efficiency AUROC comparison."
    )

r = primary.iloc[0]

axF.axhline(
    float(r["B_point"]),
    linestyle=":",
    color="0.58",
    linewidth=1.0,
)

axF.text(
    0.03,
    0.96,
    "10% transfer vs 100% local-only\n"
    f"{float(r['A_point']):.3f} vs {float(r['B_point']):.3f}\n"
    f"ΔAUROC {float(r['delta_A_minus_B']):+.4f} "
    f"[{float(r['CI_low']):+.4f}, {float(r['CI_high']):+.4f}]\n"
    "90% fewer local labeled ECGs",
    transform=axF.transAxes,
    ha="left",
    va="top",
    fontsize=10.0,
    bbox=dict(
        boxstyle="round,pad=0.27",
        facecolor="white",
        edgecolor="none",
        alpha=0.90,
    ),
)

axF.set_xticks([10, 25, 50, 100])
axF.set_xticklabels(
    ["10", "25", "50", "100"],
    fontweight="bold",
)
axF.set_ylim(0.80, 1.05)
axF.set_xlabel("PTB-XL local training data (%)")
axF.set_ylabel("AUROC")
axF.set_title("PTB-XL data efficiency — VCG-Geometry ResNet")
axF.grid(alpha=0.14)
axF.legend(
    frameon=False,
    loc="lower right",
    fontsize=9.2,
)


# ==========================================================================================
# PANEL LETTERS / TITLE / LAYOUT
# ==========================================================================================

for label, ax in zip("ABCDEF", axes.ravel()):
    ax.text(
        -0.14,
        1.065,
        label,
        transform=ax.transAxes,
        fontsize=20,
        fontweight="bold",
        ha="left",
        va="top",
    )

fig.suptitle(
    "Transfer Learning, Architecture Sensitivity, External Generalization, and Data Efficiency",
    fontsize=17.5,
    fontweight="bold",
    y=0.992,
)

fig.subplots_adjust(
    left=0.085,
    right=0.985,
    bottom=0.055,
    top=0.945,
    wspace=0.30,
    hspace=0.43,
)


# ==========================================================================================
# SOURCE DATA
# ==========================================================================================

source_blocks = [
    block(perf5, "A-B", "WCMC transfer benchmark performance"),
    block(audit_df, "A-B", "WCMC prediction audit"),
    block(two_perf, "C", "WCMC architecture x pretraining 2x2 performance"),
    block(two_diff, "C", "WCMC architecture x pretraining paired differences"),
    block(ext_perf, "D-E", "External zero-shot performance"),
    block(ext_diff, "D-E", "External paired differences"),
    block(de_perf, "F", "PTB-XL VCG data efficiency"),
    block(de_comp, "F", "PTB-XL data-efficiency paired comparisons"),
]

# Compact Panel C source data used in the final point-and-CI display.
source_blocks.append(
    block(
        pd.DataFrame(panel_c_rows),
        "C",
        "Matched architecture x pretraining AUROC point estimates",
    )
)


pd.concat(
    source_blocks,
    ignore_index=True,
    sort=False,
).to_csv(
    CSV_OUT,
    index=False,
)


# ==========================================================================================
# SAVE
# ==========================================================================================

fig.savefig(
    PNG_OUT,
    dpi=500,
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

print("\n" + "=" * 120)
print("REVISED FIGURE 4 CLEAN_C_v2 COMPLETE")
print("=" * 120)
print("A  WCMC ROC — explicit architecture/training labels")
print("B  WCMC PR — explicit architecture/training labels")
print("C  NEW matched 2x2 architecture × pretraining sensitivity")
print("D  PTB-XL zero-shot — matched ResNet vs VCG")
print("E  Chapman/Ningbo zero-shot — matched ResNet vs VCG")
print("F  PTB-XL VCG data efficiency")
print("\nSaved:")
print(PNG_OUT)
print(PDF_OUT)
print(SVG_OUT)
print(CSV_OUT)

# ==========================================================================================
# DISPLAY
# ==========================================================================================
# This explicitly displays the Matplotlib Figure object inside a Jupyter/IPYNB notebook.
# It does NOT depend on an X11 DISPLAY variable.
if IN_JUPYTER:
    # Most robust notebook display: render the already-saved high-resolution PNG.
    # This avoids back-end/front-end issues where Jupyter only prints
    # "<Figure size ... with 6 Axes>" instead of rendering the image.
    from IPython.display import Image, display
    display(Image(filename=str(PNG_OUT)))
elif os.environ.get("DISPLAY") and not os.environ.get("SLURM_JOB_ID"):
    plt.show()
else:
    print(
        "\nHeadless terminal/Slurm session detected: figure was saved but not displayed. "
        "Open the PNG/PDF, or run this same file from an IPYNB notebook to display it inline."
    )

plt.close(fig)
