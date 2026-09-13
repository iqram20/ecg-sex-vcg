#!/usr/bin/env python
# ======================================================================================
# ECG-SEX — FIGURE 4 — TRANSFER, EXTERNAL GENERALIZATION, AND DATA EFFICIENCY
# 3 × 2 | Plasma
#
# A–C: WCMC/NYP locked test, including transfer-learning benchmarks + final architecture
#   1) MIMIC-only frozen (TRiSM)
#   2) WCMC-only (TRiSM)
#   3) MIMIC→WCMC fine-tuned (TRiSM)
#   4) Transferred ResNet1D (matched final baseline)
#   5) VCG-Geometry ResNet (final primary model)
#
# D–E: final matched ResNet1D vs VCG-Geometry ResNet zero-shot external validation
# F: final VCG PTB-XL data-efficiency experiment
#
# No training/inference. Figure only.
# ======================================================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

# --------------------------------------------------------------------------------------
# PATHS
# --------------------------------------------------------------------------------------
RUN_DIR = Path(
    "/athena/madelab/scratch/iqh4001/ECG/ECG_Echo/Runs/"
    "Master_ECG_Sex_MIMIC_EchoNext/"
    "20260609_020154_MASTER_ECG_SEX_MIMIC_ECHONEXT"
)
WCMC_ROOT = RUN_DIR / "wcmc_ecgsex_from_scratch"

# Original transfer-learning benchmarks
OLD_PRED = {
    "MIMIC-only frozen":
        WCMC_ROOT / "04_wcmc_finetune_from_mimic/predictions/"
        "WCMC_ECGSex_initial_MIMICPretrain_locked_test_predictions.csv",
    "WCMC-only":
        WCMC_ROOT / "05_wcmc_from_scratch/predictions/"
        "WCMC_ECGSex_from_scratch_locked_test_predictions.csv",
    "MIMIC→WCMC fine-tuned":
        WCMC_ROOT / "04_wcmc_finetune_from_mimic/predictions/"
        "WCMC_ECGSex_finetuned_locked_test_predictions.csv",
}

# Final 5-seed WCMC comparison
WCMC_FINAL = (
    WCMC_ROOT / "19_vcg_geometry_resnet_5seed_locked_wcmc/"
    "final_5seed_ensemble_bootstrap"
)
WCMC_FINAL_PRED = WCMC_FINAL / "WCMC_LockedTest_Aligned_5SeedEnsemble_Predictions.csv"
WCMC_FINAL_PERF = WCMC_FINAL / "WCMC_VCG_vs_ResNet_5SeedEnsemble_Performance_With95CI.csv"
WCMC_FINAL_DIFF = WCMC_FINAL / "WCMC_VCG_vs_ResNet_5SeedEnsemble_PairedDifferences.csv"
WCMC_FINAL_BOOT = WCMC_FINAL / "WCMC_VCG_vs_ResNet_5SeedEnsemble_PairedBootstrap_Distributions.csv"

WCMC_OLD_BOOT = (
    WCMC_ROOT / "06_paired_bootstrap_locked_test/"
    "WCMC_ECGSex_ThreeModel_Paired_Bootstrap_Distributions.csv"
)

# External confirmation
EXT_ROOT = WCMC_ROOT / "20_vcg_geometry_resnet_external_confirmation"
EXT_FINAL = EXT_ROOT / "final_paired_bootstrap"
EXT_PERF = EXT_FINAL / "VCGGeometryResNet_ExternalConfirmation_Combined_Performance.csv"
EXT_DIFF = EXT_FINAL / "VCGGeometryResNet_ExternalConfirmation_Combined_PairedDifferences.csv"
PTB_BOOT = EXT_FINAL / "ptbxl_VCG_vs_ResNet_bootstrap_2000.csv"
CHAPMAN_BOOT = EXT_FINAL / "chapman_ningbo_VCG_vs_ResNet_bootstrap_2000.csv"

# Final VCG PTB-XL data efficiency
DE_FINAL = (
    RUN_DIR / "ptbxl_ecgsex_from_scratch/12_vcg_geometry_data_efficiency/"
    "final_seedensemble_bootstrap"
)
DE_PERF = DE_FINAL / "PTBXL_VCG_DataEfficiency_SeedEnsemble_Performance_With95CI.csv"
DE_COMP = DE_FINAL / "PTBXL_VCG_DataEfficiency_PairedComparisons.csv"

# Output
OUTDIR = RUN_DIR / "publication_figures"
OUTDIR.mkdir(parents=True, exist_ok=True)
STEM = OUTDIR / "Figure4_FINAL_Transfer_External_DataEfficiency_v3_NoPValues"
PNG_OUT = Path(str(STEM) + ".png")
PDF_OUT = Path(str(STEM) + ".pdf")
SVG_OUT = Path(str(STEM) + ".svg")
CSV_OUT = Path(str(STEM) + "_SourceData.csv")

# --------------------------------------------------------------------------------------
# FILE AUDIT
# --------------------------------------------------------------------------------------
required = list(OLD_PRED.values()) + [
    WCMC_FINAL_PRED, WCMC_FINAL_PERF, WCMC_FINAL_DIFF,
    WCMC_FINAL_BOOT, WCMC_OLD_BOOT,
    EXT_PERF, EXT_DIFF, PTB_BOOT, CHAPMAN_BOOT,
    DE_PERF, DE_COMP,
]
missing = [p for p in required if not p.exists()]
if missing:
    raise FileNotFoundError("Missing required files:\n" + "\n".join(map(str, missing)))

# --------------------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------------------
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
    ycol = find_col(df.columns, [
        "sex_true_binary", "y_true", "true_label", "sex_binary", "label", "target"
    ])
    pcol = find_col(df.columns, [
        "prob_male", "pred_prob", "y_prob", "probability", "prob", "score"
    ])
    if ycol is None or pcol is None:
        raise RuntimeError(f"Cannot identify label/probability columns in {path}\n{list(df.columns)}")
    out = df[[ycol, pcol]].copy()
    out.columns = ["y_true", "prob"]
    out["y_true"] = pd.to_numeric(out["y_true"], errors="raise").astype(int)
    out["prob"] = pd.to_numeric(out["prob"], errors="raise")
    out = out.dropna().reset_index(drop=True)
    if set(out.y_true.unique()) != {0, 1}:
        raise RuntimeError(f"Invalid binary labels in {path}")
    return out


def canon_final_model(name):
    s = str(name).lower()
    if "vcg" in s and "resnet" in s:
        return "VCG-Geometry ResNet"
    if "resnet" in s and "vcg" not in s:
        return "Transferred ResNet1D"
    return str(name)


def paired_delta_ci_text(row, delta_col="delta_VCG_minus_ResNet", label="Δ"):
    """
    Format a paired model difference using the point estimate and 95% bootstrap CI.
    No conventional P-value label is shown in Panels C-E.

    This helper accepts several common CI column naming conventions used in the
    saved paired-difference CSVs and hard-stops if the CI cannot be identified.
    """
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
        "Could not identify 95% CI columns in paired-difference row. "
        f"Available columns: {list(row.index)}"
    )


# --------------------------------------------------------------------------------------
# ORIGINAL THREE WCMC PREDICTIONS + FINALIZED BOOTSTRAP VALUES
# --------------------------------------------------------------------------------------
preds = {name: load_pred(path) for name, path in OLD_PRED.items()}

# Finalized 2,000-paired-bootstrap results from the original three-model WCMC analysis.
# These values are intentionally kept fixed because they are the manuscript benchmark values.
old_perf = pd.DataFrame([
    dict(model="MIMIC-only frozen",
         AUROC=0.904, AUROC_CI_low=0.901, AUROC_CI_high=0.908,
         AUPRC=0.896, AUPRC_CI_low=0.891, AUPRC_CI_high=0.902,
         family="Original TRiSM transfer benchmark"),
    dict(model="WCMC-only",
         AUROC=0.929, AUROC_CI_low=0.926, AUROC_CI_high=0.933,
         AUPRC=0.923, AUPRC_CI_low=0.919, AUPRC_CI_high=0.927,
         family="Original TRiSM transfer benchmark"),
    dict(model="MIMIC→WCMC fine-tuned",
         AUROC=0.944, AUROC_CI_low=0.942, AUROC_CI_high=0.947,
         AUPRC=0.940, AUPRC_CI_low=0.936, AUPRC_CI_high=0.943,
         family="Original TRiSM transfer benchmark"),
])

# --------------------------------------------------------------------------------------
# FINAL WCMC PREDICTIONS / PERFORMANCE
# --------------------------------------------------------------------------------------
fraw = pd.read_csv(WCMC_FINAL_PRED, low_memory=False)
for c in ["sex_true_binary", "prob_resnet", "prob_vcg"]:
    if c not in fraw.columns:
        raise RuntimeError(f"Missing {c} in {WCMC_FINAL_PRED}")

preds["Transferred ResNet1D"] = pd.DataFrame({
    "y_true": pd.to_numeric(fraw["sex_true_binary"], errors="raise").astype(int),
    "prob": pd.to_numeric(fraw["prob_resnet"], errors="raise"),
})
preds["VCG-Geometry ResNet"] = pd.DataFrame({
    "y_true": pd.to_numeric(fraw["sex_true_binary"], errors="raise").astype(int),
    "prob": pd.to_numeric(fraw["prob_vcg"], errors="raise"),
})

fperf = pd.read_csv(WCMC_FINAL_PERF, low_memory=False)
fperf["canonical_model"] = fperf["model"].map(canon_final_model)

final_rows = []
for model in ["Transferred ResNet1D", "VCG-Geometry ResNet"]:
    sub = fperf.loc[fperf.canonical_model == model]
    if len(sub) != 1:
        raise RuntimeError(f"Expected one final WCMC row for {model}; found {len(sub)}")
    r = sub.iloc[0]
    final_rows.append(dict(
        model=model,
        AUROC=float(r.AUROC), AUROC_CI_low=float(r.AUROC_CI_low), AUROC_CI_high=float(r.AUROC_CI_high),
        AUPRC=float(r.AUPRC), AUPRC_CI_low=float(r.AUPRC_CI_low), AUPRC_CI_high=float(r.AUPRC_CI_high),
        family="Final matched architecture comparison",
    ))

perf5 = pd.concat([old_perf, pd.DataFrame(final_rows)], ignore_index=True)

MODEL_ORDER = [
    "MIMIC-only frozen",
    "WCMC-only",
    "MIMIC→WCMC fine-tuned",
    "Transferred ResNet1D",
    "VCG-Geometry ResNet",
]

# --------------------------------------------------------------------------------------
# AUDIT ALL FIVE WCMC PREDICTION SETS
# --------------------------------------------------------------------------------------
audit = []
for model in MODEL_ORDER:
    df = preds[model]
    if len(df) != 21374:
        raise RuntimeError(f"{model}: expected N=21,374; got {len(df):,}")
    auc = roc_auc_score(df.y_true, df.prob)
    ap = average_precision_score(df.y_true, df.prob)
    saved = perf5.loc[perf5.model == model].iloc[0]
    # Old benchmark values are rounded to 3 decimals; final rows are more precise.
    tol = 0.0025 if model in old_perf.model.values else 5e-5
    if abs(auc - saved.AUROC) > tol or abs(ap - saved.AUPRC) > tol:
        raise RuntimeError(
            f"Metric audit failed for {model}: observed AUROC/AUPRC={auc:.6f}/{ap:.6f}; "
            f"saved={saved.AUROC:.6f}/{saved.AUPRC:.6f}"
        )
    audit.append(dict(model=model, N=len(df), observed_AUROC=auc, observed_AUPRC=ap))
audit_df = pd.DataFrame(audit)

print("=" * 120)
print("WCMC FIVE-MODEL PREDICTION AUDIT")
print("=" * 120)
print(audit_df.to_string(index=False))

# --------------------------------------------------------------------------------------
# LOAD FINAL PAIRED / EXTERNAL / DATA-EFFICIENCY TABLES
# --------------------------------------------------------------------------------------
wcmc_diff = pd.read_csv(WCMC_FINAL_DIFF, low_memory=False)
ext_perf = pd.read_csv(EXT_PERF, low_memory=False)
ext_diff = pd.read_csv(EXT_DIFF, low_memory=False)
de_perf = pd.read_csv(DE_PERF, low_memory=False)
de_comp = pd.read_csv(DE_COMP, low_memory=False)

# Real bootstrap distributions for violin panels C-E.
wcmc_old_boot = pd.read_csv(WCMC_OLD_BOOT, low_memory=False)
wcmc_final_boot = pd.read_csv(WCMC_FINAL_BOOT, low_memory=False)
ptb_boot = pd.read_csv(PTB_BOOT, low_memory=False)
chapman_boot = pd.read_csv(CHAPMAN_BOOT, low_memory=False)

OLD_BOOT_SAFE = {
    "MIMIC-only frozen": "mimic_pretrained_frozen",
    "WCMC-only": "wcmc_from_scratch",
    "MIMIC→WCMC fine-tuned": "mimic_to_wcmc_fine_tuned",
}

FINAL_BOOT_SAFE = {
    "Transferred ResNet1D": "ResNet",
    "VCG-Geometry ResNet": "VCG",
}

def require_bootstrap_vector(df, column, label):
    if column not in df.columns:
        raise RuntimeError(
            f"Missing real bootstrap column '{column}' for {label}. "
            f"No fallback distribution will be generated."
        )
    vals = pd.to_numeric(
        df[column],
        errors="coerce",
    ).dropna().to_numpy(dtype=float)

    if vals.size < 1000:
        raise RuntimeError(
            f"Insufficient real bootstrap replicates for {label}: "
            f"{vals.size:,} found; expected approximately 2,000."
        )

    if not np.isfinite(vals).all():
        raise RuntimeError(f"Non-finite bootstrap values for {label}.")

    return vals


def wcmc_boot_values(model, metric):
    if model in OLD_BOOT_SAFE:
        return require_bootstrap_vector(
            wcmc_old_boot,
            f"{OLD_BOOT_SAFE[model]}__{metric}",
            f"WCMC {model} {metric}",
        )

    if model in FINAL_BOOT_SAFE:
        return require_bootstrap_vector(
            wcmc_final_boot,
            f"{FINAL_BOOT_SAFE[model]}__{metric}",
            f"WCMC {model} {metric}",
        )

    raise KeyError(model)


def external_boot_values(dataset, model, metric):
    boot = ptb_boot if dataset == "PTB-XL" else chapman_boot
    prefix = "VCG" if model == "VCG-Geometry ResNet" else "ResNet"

    return require_bootstrap_vector(
        boot,
        f"{prefix}__{metric}",
        f"{dataset} {model} {metric}",
    )


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
ext_perf["canonical_model"] = ext_perf["model"].map(canon_final_model)

# --------------------------------------------------------------------------------------
# COLORS / STYLES
# --------------------------------------------------------------------------------------
cmap = plt.get_cmap("plasma")
COLORS = {
    "MIMIC-only frozen": cmap(0.08),
    "WCMC-only": cmap(0.28),
    "MIMIC→WCMC fine-tuned": cmap(0.48),
    "Transferred ResNet1D": cmap(0.68),
    "VCG-Geometry ResNet": cmap(0.90),
}
LINESTYLES = {
    "MIMIC-only frozen": ":",
    "WCMC-only": "--",
    "MIMIC→WCMC fine-tuned": "-.",
    "Transferred ResNet1D": "--",
    "VCG-Geometry ResNet": "-",
}
LINEWIDTHS = {
    "MIMIC-only frozen": 1.55,
    "WCMC-only": 1.65,
    "MIMIC→WCMC fine-tuned": 1.80,
    "Transferred ResNet1D": 2.05,
    "VCG-Geometry ResNet": 2.70,
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10.2,
    "axes.titlesize": 11.4,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.4,
    "xtick.labelsize": 9.2,
    "ytick.labelsize": 9.2,
    "legend.fontsize": 7.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
})

# --------------------------------------------------------------------------------------
# FIGURE
# --------------------------------------------------------------------------------------
fig, axes = plt.subplots(3, 2, figsize=(13.8, 15.5))
axA, axB = axes[0]
axC, axD = axes[1]
axE, axF = axes[2]

# ======================================================================================
# A — WCMC ROC, ALL FIVE
# ======================================================================================
roc_handles = {}
for model in MODEL_ORDER:
    df = preds[model]
    fpr, tpr, _ = roc_curve(df.y_true, df.prob)
    r = perf5.loc[perf5.model == model].iloc[0]
    h, = axA.plot(
        fpr, tpr,
        color=COLORS[model], linestyle=LINESTYLES[model], linewidth=LINEWIDTHS[model],
        label=f"{model}: {r.AUROC:.3f} ({r.AUROC_CI_low:.3f}–{r.AUROC_CI_high:.3f})"
    )
    roc_handles[model] = h
axA.plot([0, 1], [0, 1], "--", color="0.76", linewidth=0.9)
axA.set(xlim=(0, 1), ylim=(0, 1.015), xlabel="False-positive rate", ylabel="True-positive rate")
axA.set_title("WCMC/NYP locked-test ROC")
axA.grid(alpha=0.14)
axA.legend([roc_handles[m] for m in MODEL_ORDER], [roc_handles[m].get_label() for m in MODEL_ORDER],
           frameon=False, loc="lower right")

# ======================================================================================
# B — WCMC PR, ALL FIVE
# ======================================================================================
pr_handles = {}
prevalence = float(preds[MODEL_ORDER[0]].y_true.mean())
for model in MODEL_ORDER:
    df = preds[model]
    precision, recall, _ = precision_recall_curve(df.y_true, df.prob)
    r = perf5.loc[perf5.model == model].iloc[0]
    h, = axB.plot(
        recall, precision,
        color=COLORS[model], linestyle=LINESTYLES[model], linewidth=LINEWIDTHS[model],
        label=f"{model}: {r.AUPRC:.3f} ({r.AUPRC_CI_low:.3f}–{r.AUPRC_CI_high:.3f})"
    )
    pr_handles[model] = h
axB.axhline(prevalence, linestyle=":", color="0.72", linewidth=1.0)
axB.set(xlim=(0, 1), ylim=(0.45, 1.01), xlabel="Recall", ylabel="Precision")
axB.set_title("WCMC/NYP locked-test precision–recall")
axB.grid(alpha=0.14)
axB.legend([pr_handles[m] for m in MODEL_ORDER], [pr_handles[m].get_label() for m in MODEL_ORDER],
           frameon=False, loc="lower left")

# ======================================================================================
# C — WCMC REAL PAIRED-BOOTSTRAP VIOLINS, ALL FIVE
#     CLEAN LEGEND + PAIRED Δ WITH 95% BOOTSTRAP CI (NO P-VALUE LABEL)
# ======================================================================================

metrics = [
    "AUROC",
    "AUPRC",
]

x = np.arange(
    2
)

offsets = np.linspace(
    -0.30,
    0.30,
    len(
        MODEL_ORDER
    ),
)

violin_width = 0.115

panel_c_all = []
panel_c_positions = {}

for metric_idx, metric in enumerate(
    metrics
):
    panel_c_positions[
        metric
    ] = {}

    for model, off in zip(
        MODEL_ORDER,
        offsets,
    ):
        r = perf5.loc[
            perf5.model
            ==
            model
        ].iloc[
            0
        ]

        point = float(
            r[
                metric
            ]
        )

        lo = float(
            r[
                f"{metric}_CI_low"
            ]
        )

        hi = float(
            r[
                f"{metric}_CI_high"
            ]
        )

        vals = wcmc_boot_values(
            model,
            metric,
        )

        panel_c_all.append(
            vals
        )

        xpos = (
            x[
                metric_idx
            ]
            +
            off
        )

        panel_c_positions[
            metric
        ][
            model
        ] = {
            "x":
                xpos,

            "point":
                point,

            "low":
                lo,

            "high":
                hi,
        }

        draw_violin(
            axC,
            vals,
            xpos,
            COLORS[
                model
            ],
            violin_width,
        )

        # 95% bootstrap CI.
        axC.vlines(
            xpos,
            lo,
            hi,
            color="black",
            linewidth=1.55,
            zorder=4,
        )

        # Saved point estimate.
        axC.scatter(
            [
                xpos
            ],
            [
                point
            ],
            s=27,
            facecolor="white",
            edgecolor="black",
            linewidth=0.85,
            zorder=5,
        )

        # Keep point labels compact and close to their own violin.
        axC.text(
            xpos,
            hi + 0.00075,
            f"{point:.3f}",
            ha="center",
            va="bottom",
            fontsize=6.9,
            zorder=6,
        )


# --------------------------------------------------------------------------------------
# PAIRWISE VCG-vs-RESNET COMPARISON:
# small bracket above the final two violins for each metric.
# --------------------------------------------------------------------------------------

for metric in metrics:

    if metric == "AUROC":
        diff_row = wcmc_diff.loc[
            wcmc_diff.metric
            ==
            "AUROC"
        ].iloc[
            0
        ]

    else:
        diff_row = wcmc_diff.loc[
            wcmc_diff.metric
            ==
            "AUPRC"
        ].iloc[
            0
        ]

    resnet_info = panel_c_positions[
        metric
    ][
        "Transferred ResNet1D"
    ]

    vcg_info = panel_c_positions[
        metric
    ][
        "VCG-Geometry ResNet"
    ]

    x1 = float(
        resnet_info[
            "x"
        ]
    )

    x2 = float(
        vcg_info[
            "x"
        ]
    )

    # Put comparison well above numeric value labels.
    y_bracket = (
        max(
            float(
                resnet_info[
                    "high"
                ]
            ),
            float(
                vcg_info[
                    "high"
                ]
            ),
        )
        +
        0.0048
    )

    bracket_height = 0.0008

    axC.plot(
        [
            x1,
            x1,
            x2,
            x2,
        ],
        [
            y_bracket,
            y_bracket
            +
            bracket_height,
            y_bracket
            +
            bracket_height,
            y_bracket,
        ],
        color="black",
        linewidth=0.85,
        clip_on=False,
        zorder=7,
    )

    axC.text(
        (
            x1
            +
            x2
        )
        /
        2.0,
        y_bracket
        +
        bracket_height
        +
        0.00045,
        paired_delta_ci_text(
            diff_row,
            delta_col="delta_VCG_minus_ResNet",
            label="Δ",
        ),
        ha="center",
        va="bottom",
        fontsize=6.7,
        zorder=8,
    )


# --------------------------------------------------------------------------------------
# AXES
# --------------------------------------------------------------------------------------

for xc in x:
    axC.axvline(
        xc,
        color="0.93",
        linewidth=0.8,
        zorder=0,
    )

panel_c_concat = np.concatenate(
    panel_c_all
)

panel_c_ymin = max(
    0.88,
    float(
        np.nanmin(
            panel_c_concat
        )
    )
    -
    0.005,
)

# Explicit headroom for paired-comparison brackets.
highest_ci = max(
    float(
        perf5[
            "AUROC_CI_high"
        ].max()
    ),
    float(
        perf5[
            "AUPRC_CI_high"
        ].max()
    ),
)

panel_c_ymax = min(
    0.980,
    highest_ci
    +
    0.0125,
)

axC.set_xticks(
    x
)

axC.set_xticklabels(
    metrics
)

axC.set_ylim(
    panel_c_ymin,
    panel_c_ymax,
)

axC.set_ylabel(
    "Bootstrap performance"
)

axC.set_title(
    "WCMC/NYP paired-bootstrap distributions"
)

axC.grid(
    axis="y",
    alpha=0.14,
)


# --------------------------------------------------------------------------------------
# CLEAN, ABBREVIATED LEGEND
# --------------------------------------------------------------------------------------

PANEL_C_LEGEND_LABELS = {
    "MIMIC-only frozen":
        "MIMIC-only",

    "WCMC-only":
        "WCMC-only",

    "MIMIC→WCMC fine-tuned":
        "MIMIC→WCMC",

    "Transferred ResNet1D":
        "ResNet1D",

    "VCG-Geometry ResNet":
        "VCG-ResNet",
}

panel_c_handles = [
    Line2D(
        [
            0
        ],
        [
            0
        ],
        marker="o",
        linestyle="none",
        markersize=5.7,
        markerfacecolor=COLORS[
            model
        ],
        markeredgecolor="black",
        markeredgewidth=0.45,
        label=PANEL_C_LEGEND_LABELS[
            model
        ],
    )
    for model in MODEL_ORDER
]

axC.legend(
    handles=panel_c_handles,
    frameon=False,
    loc="lower left",
    bbox_to_anchor=(
        0.005,
        0.005,
    ),
    fontsize=6.35,
    ncol=2,
    handletextpad=0.30,
    columnspacing=0.75,
    borderaxespad=0.0,
)


# ======================================================================================
# D/E — FINAL MATCHED EXTERNAL COMPARISON — REAL BOOTSTRAP VIOLINS
#       Paired Δ values are shown with 95% bootstrap CIs; no P-value labels.
# ======================================================================================
def plot_external(ax, dataset):
    sub = ext_perf.loc[ext_perf.dataset == dataset].copy()

    models = [
        "Transferred ResNet1D",
        "VCG-Geometry ResNet",
    ]

    centers = np.arange(2)

    offsets = {
        "Transferred ResNet1D": -0.16,
        "VCG-Geometry ResNet": +0.16,
    }

    violin_width = 0.245
    all_values = []

    for metric_idx, metric in enumerate(["AUROC", "AUPRC"]):
        for model in models:
            rr = sub.loc[sub.canonical_model == model]

            if len(rr) != 1:
                raise RuntimeError(
                    f"{dataset}: expected one performance row for {model}; "
                    f"found {len(rr)}."
                )

            r = rr.iloc[0]

            point = float(r[metric])
            lo = float(r[f"{metric}_CI_low"])
            hi = float(r[f"{metric}_CI_high"])

            vals = external_boot_values(
                dataset,
                model,
                metric,
            )

            all_values.append(vals)

            xpos = centers[metric_idx] + offsets[model]

            draw_violin(
                ax,
                vals,
                xpos,
                COLORS[model],
                violin_width,
            )

            ax.vlines(
                xpos,
                lo,
                hi,
                color="black",
                linewidth=1.65,
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
                hi + 0.0014,
                f"{point:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.9,
            )

    for xc in centers:
        ax.axvline(
            xc,
            color="0.93",
            linewidth=0.8,
            zorder=0,
        )

    concat = np.concatenate(all_values)

    ymin = max(
        0.875,
        float(np.nanmin(concat)) - 0.006,
    )

    ymax = min(
        0.970,
        float(np.nanmax(concat)) + 0.010,
    )

    ax.set_xticks(centers)
    ax.set_xticklabels(["AUROC", "AUPRC"])
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel("Bootstrap performance")
    ax.set_title(f"{dataset} zero-shot paired-bootstrap distributions")
    ax.grid(axis="y", alpha=0.14)

    da = ext_diff.loc[
        (ext_diff.dataset == dataset)
        &
        (ext_diff.metric == "AUROC")
    ].iloc[0]

    dp = ext_diff.loc[
        (ext_diff.dataset == dataset)
        &
        (ext_diff.metric == "AUPRC")
    ].iloc[0]

    ax.text(
        0.02,
        0.97,
        "VCG − ResNet\n"
        + paired_delta_ci_text(
            da,
            delta_col="delta_VCG_minus_ResNet",
            label="ΔAUROC",
        )
        + "\n"
        + paired_delta_ci_text(
            dp,
            delta_col="delta_VCG_minus_ResNet",
            label="ΔAUPRC",
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.6,
        bbox=dict(
            boxstyle="round,pad=0.24",
            facecolor="white",
            edgecolor="none",
            alpha=0.90,
        ),
    )


plot_external(axD, "PTB-XL")
plot_external(axE, "Chapman/Ningbo")

fig.legend(
    handles=[
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=7.5,
            markerfacecolor=COLORS["Transferred ResNet1D"],
            markeredgecolor="black",
            markeredgewidth=0.5,
            label="Transferred ResNet1D",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=7.5,
            markerfacecolor=COLORS["VCG-Geometry ResNet"],
            markeredgecolor="black",
            markeredgewidth=0.5,
            label="VCG-Geometry ResNet",
        ),
    ],
    loc="lower center",
    bbox_to_anchor=(0.50, 0.337),
    ncol=2,
    frameon=False,
    fontsize=9.0,
    columnspacing=2.5,
)


# ======================================================================================
# F — FINAL VCG PTB-XL DATA EFFICIENCY
#     Primary comparison shown as ΔAUROC with 95% paired-bootstrap CI; no P-value label.
# ======================================================================================
LOCAL = "PTB-XL-only VCG-Geometry ResNet"
TRANSFER = "MIMIC→WCMC→PTB-XL VCG-Geometry ResNet"

for strategy, color, ls, label in [
    (LOCAL, cmap(0.08), "--", "PTB-XL-only VCG"),
    (TRANSFER, COLORS["VCG-Geometry ResNet"], "-", "MIMIC→WCMC→PTB-XL VCG"),
]:
    sub = de_perf.loc[de_perf.strategy == strategy].sort_values("train_percent")
    if len(sub) != 4:
        raise RuntimeError(f"Expected four data-efficiency rows for {strategy}; found {len(sub)}")
    xp = sub.train_percent.to_numpy(float)
    vals = sub.AUROC.to_numpy(float)
    lows = sub.AUROC_CI_low.to_numpy(float)
    highs = sub.AUROC_CI_high.to_numpy(float)
    axF.errorbar(
        xp, vals, yerr=np.vstack([vals-lows, highs-vals]),
        marker="o", markersize=6.2, linewidth=2.0, linestyle=ls, capsize=3.4,
        color=color, label=label
    )
    for xi, yi, hi in zip(xp, vals, highs):
        axF.text(xi, hi + 0.0020, f"{yi:.3f}", ha="center", va="bottom", fontsize=7.8)

primary = de_comp.loc[(de_comp.comparison_type == "PRIMARY data-efficiency") & (de_comp.metric == "AUROC")]
if len(primary) != 1:
    raise RuntimeError("Expected exactly one PRIMARY data-efficiency AUROC comparison")
r = primary.iloc[0]
axF.axhline(float(r.B_point), linestyle=":", color="0.58", linewidth=1.0)
axF.text(
    0.03, 0.96,
    "10% transfer vs 100% local-only\n"
    f"{float(r.A_point):.3f} vs {float(r.B_point):.3f}\n"
    f"ΔAUROC {float(r.delta_A_minus_B):+.4f} "
    f"[{float(r.CI_low):+.4f}, {float(r.CI_high):+.4f}]\n"
    "90% fewer local labeled ECGs",
    transform=axF.transAxes, ha="left", va="top", fontsize=7.8,
    bbox=dict(boxstyle="round,pad=0.27", facecolor="white", edgecolor="none", alpha=0.90)
)
axF.set_xticks([10, 25, 50, 100])
axF.set_ylim(0.80, 0.99)
axF.set_xlabel("PTB-XL local training data (%)")
axF.set_ylabel("AUROC")
axF.set_title("PTB-XL data efficiency — VCG-Geometry ResNet")
axF.grid(alpha=0.14)
axF.legend(frameon=False, loc="lower right")

# --------------------------------------------------------------------------------------
# PANEL LETTERS / LAYOUT
# --------------------------------------------------------------------------------------
for label, ax in zip("ABCDEF", axes.ravel()):
    ax.text(-0.105, 1.065, label, transform=ax.transAxes, fontsize=16,
            fontweight="bold", ha="left", va="top")

fig.subplots_adjust(left=0.078, right=0.985, bottom=0.055, top=0.98, wspace=0.27, hspace=0.39)

# --------------------------------------------------------------------------------------
# SOURCE DATA
# --------------------------------------------------------------------------------------
def block(df, panel, source):
    out = df.copy()
    out.insert(0, "panel", panel)
    out.insert(1, "source", source)
    return out

# Real bootstrap source data used for Panels C-E.
violin_rows = []

for metric in ["AUROC", "AUPRC"]:
    for model in MODEL_ORDER:
        vals = wcmc_boot_values(model, metric)

        violin_rows.extend(
            {
                "panel": "C",
                "source": "Real paired-bootstrap distribution",
                "dataset": "WCMC/NYP",
                "model": model,
                "metric": metric,
                "bootstrap_id": i + 1,
                "bootstrap_value": float(v),
            }
            for i, v in enumerate(vals)
        )

for dataset in ["PTB-XL", "Chapman/Ningbo"]:
    for metric in ["AUROC", "AUPRC"]:
        for model in ["Transferred ResNet1D", "VCG-Geometry ResNet"]:
            vals = external_boot_values(dataset, model, metric)

            violin_rows.extend(
                {
                    "panel": "D" if dataset == "PTB-XL" else "E",
                    "source": "Real paired-bootstrap distribution",
                    "dataset": dataset,
                    "model": model,
                    "metric": metric,
                    "bootstrap_id": i + 1,
                    "bootstrap_value": float(v),
                }
                for i, v in enumerate(vals)
            )

violin_source_df = pd.DataFrame(violin_rows)

pd.concat([
    block(perf5, "A-C", "WCMC five-model performance"),
    block(audit_df, "A-B", "WCMC prediction audit"),
    block(wcmc_diff, "C", "Final VCG vs matched ResNet paired differences"),
    block(ext_perf, "D-E", "External zero-shot performance"),
    block(ext_diff, "D-E", "External paired differences"),
    block(de_perf, "F", "Final VCG PTB-XL data efficiency"),
    block(de_comp, "F", "Final VCG data-efficiency paired comparisons"),
    violin_source_df,
], ignore_index=True, sort=False).to_csv(CSV_OUT, index=False)

# --------------------------------------------------------------------------------------
# SAVE
# --------------------------------------------------------------------------------------
fig.savefig(PNG_OUT, dpi=500, bbox_inches="tight", facecolor="white")
fig.savefig(PDF_OUT, bbox_inches="tight", facecolor="white")
fig.savefig(SVG_OUT, bbox_inches="tight", facecolor="white")

print("\n" + "=" * 120)
print("FINAL MAIN PERFORMANCE FIGURE WITH MIMIC/WCMC TRANSFER BENCHMARKS")
print("=" * 120)
print("A  WCMC ROC — 5 models")
print("B  WCMC PR — 5 models")
print("C  WCMC real paired-bootstrap AUROC/AUPRC violins — 5 models")
print("D  PTB-XL zero-shot — real paired-bootstrap violins, ResNet vs VCG")
print("E  Chapman/Ningbo zero-shot — real paired-bootstrap violins, ResNet vs VCG")
print("Panels C-E: paired differences shown as Δ with 95% bootstrap CI; no P-value labels")
print("F  PTB-XL data efficiency — ΔAUROC with 95% bootstrap CI; no P-value label")
print("\nSaved:")
print(PNG_OUT)
print(PDF_OUT)
print(SVG_OUT)
print(CSV_OUT)
print("\nDisplaying figure...")
plt.show()
print("DONE")
