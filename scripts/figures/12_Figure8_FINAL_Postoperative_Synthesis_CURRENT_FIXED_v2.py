#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import os, re, warnings
import numpy as np
import pandas as pd
import matplotlib

def _running_in_jupyter():
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and (
            "IPKernelApp" in getattr(ip, "config", {})
            or ip.__class__.__name__ == "ZMQInteractiveShell"
        )
    except Exception:
        return False

IN_JUPYTER = _running_in_jupyter()
if (not IN_JUPYTER) and (os.environ.get("SLURM_JOB_ID") or not os.environ.get("DISPLAY")):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------------------
# CURRENT FINAL RUN
# ------------------------------------------------------------------------------------------

RUN_DIR = Path(
    "/athena/madelab/scratch/iqh4001/ECG/ECG_Echo/Runs/"
    "Master_ECG_Sex_MIMIC_EchoNext/"
    "20260609_020154_MASTER_ECG_SEX_MIMIC_ECHONEXT"
)

FINAL_ROOT = Path(
    "/athena/madelab/scratch/iqh4001/ECG/ECG-gender/Results/"
    "Postop_FinalSynthesis_v2"
)

ASSOC_CSV = FINAL_ROOT / "02_FINAL_combined_postoperative_associations.csv"
LOOKBACK_CSV = FINAL_ROOT / "03_FINAL_WCMC_mortality_lookback_sensitivity.csv"
INCREMENTAL_CSV = FINAL_ROOT / "04_FINAL_incremental_performance.csv"

OUT_DIR = RUN_DIR / "publication_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STEM = OUT_DIR / "Figure8_FINAL_Postoperative_Synthesis_CURRENT_FIXED_v2"
PNG_OUT = Path(str(STEM) + ".png")
PDF_OUT = Path(str(STEM) + ".pdf")
SVG_OUT = Path(str(STEM) + ".svg")
SOURCE_OUT = Path(str(STEM) + "_SourceData.csv")

for p in [ASSOC_CSV, LOOKBACK_CSV, INCREMENTAL_CSV]:
    if not p.exists():
        raise FileNotFoundError(f"Missing CURRENT final synthesis file:\n{p}")

# ------------------------------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------------------------------

def norm(x):
    return re.sub(r"[^a-z0-9]+", "", str(x).lower())

def pick_col(df, aliases, label, required=True):
    """
    Safely resolve a column name.

    IMPORTANT:
    Do NOT allow short aliases such as "OR" or "N" to fuzzy-match arbitrary
    columns (e.g., "OR" can match "cohort"; "N" can match many names).
    Exact normalized matches are always preferred. Fuzzy matching is allowed
    only for aliases with >= 4 normalized characters.
    """
    cols = list(df.columns)
    exact = {norm(c): c for c in cols}

    # 1) Exact normalized match.
    for alias in aliases:
        key = norm(alias)
        if key in exact:
            return exact[key]

    # 2) Safe fuzzy match only for sufficiently specific aliases.
    candidates = []
    for alias in aliases:
        key = norm(alias)
        if len(key) < 4:
            continue
        for c in cols:
            nc = norm(c)
            if key in nc or nc in key:
                candidates.append(c)

    # Preserve order while removing duplicates.
    candidates = list(dict.fromkeys(candidates))

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) > 1:
        raise RuntimeError(
            f"Ambiguous {label} column.\n"
            f"Aliases tried: {aliases}\n"
            f"Possible matches: {candidates}\n"
            f"Observed columns: {cols}"
        )

    if required:
        raise RuntimeError(
            f"Cannot identify {label} column.\n"
            f"Aliases tried: {aliases}\n"
            f"Observed columns: {cols}"
        )

    return None

def cohort_name(x):
    s = str(x).lower()
    if "wcmc" in s or "nyp" in s:
        return "WCMC/NYP"
    if "mimic" in s:
        return "MIMIC-IV"
    return str(x)

def outcome_name(x):
    s = str(x).lower()
    if "mort" in s:
        return "30-day mortality"
    if "aki" in s or "kidney" in s:
        return "Postoperative AKI"
    if "icu" in s:
        return "Postoperative ICU admission"
    return str(x)

def num(x):
    return pd.to_numeric(x, errors="coerce")

def add_block(df, panel, source):
    out = df.copy()
    out.insert(0, "panel", panel)
    out.insert(1, "source", source)
    return out

# ------------------------------------------------------------------------------------------
# LOAD
# ------------------------------------------------------------------------------------------

assoc_raw = pd.read_csv(ASSOC_CSV, low_memory=False)
look_raw = pd.read_csv(LOOKBACK_CSV, low_memory=False)
inc_raw = pd.read_csv(INCREMENTAL_CSV, low_memory=False)

print("=" * 120)
print("FIGURE 8 — CURRENT FINAL POSTOPERATIVE SYNTHESIS")
print("=" * 120)
print("Final result root:", FINAL_ROOT)
print("\nAssociation columns:", assoc_raw.columns.tolist())
print("Lookback columns:", look_raw.columns.tolist())
print("Incremental columns:", inc_raw.columns.tolist())

print("\nSchema audit:")
print("  Association file :", ASSOC_CSV)
print("  Lookback file    :", LOOKBACK_CSV)
print("  Incremental file :", INCREMENTAL_CSV)

# ------------------------------------------------------------------------------------------
# PANEL B DATA — FINAL WCMC PRIMARY + MIMIC SECONDARY REPLICATION
# ------------------------------------------------------------------------------------------

c_cohort = pick_col(assoc_raw, ["cohort", "dataset", "source"], "cohort")
c_out = pick_col(assoc_raw, ["outcome", "endpoint"], "outcome")
c_or = pick_col(
    assoc_raw,
    ["OR", "adjusted_OR", "OR_per_1SD", "odds_ratio", "adjusted_or"],
    "OR",
)
c_lo = pick_col(assoc_raw, ["OR_CI95_low", "CI95_low", "CI_low", "lower"], "CI low")
c_hi = pick_col(assoc_raw, ["OR_CI95_high", "CI95_high", "CI_high", "upper"], "CI high")
c_n = pick_col(
    assoc_raw,
    ["N", "analysis_N", "sample_size", "n_patients"],
    "N",
    required=False,
)
c_ev = pick_col(assoc_raw, ["events", "n_events", "deaths"], "events", required=False)
c_win = pick_col(
    assoc_raw,
    ["window_days", "ecg_window_days", "lookback_days", "window"],
    "window",
    required=False,
)

assoc = pd.DataFrame({
    "cohort": assoc_raw[c_cohort].map(cohort_name),
    "outcome": assoc_raw[c_out].map(outcome_name),
    "OR": num(assoc_raw[c_or]),
    "CI_low": num(assoc_raw[c_lo]),
    "CI_high": num(assoc_raw[c_hi]),
    "N": num(assoc_raw[c_n]) if c_n else np.nan,
    "events": num(assoc_raw[c_ev]) if c_ev else np.nan,
    "window": assoc_raw[c_win] if c_win else np.nan,
})

OUTCOMES = [
    "30-day mortality",
    "Postoperative AKI",
    "Postoperative ICU admission",
]

wcmc = assoc.loc[assoc["cohort"].eq("WCMC/NYP")].copy()
if c_win:
    wnum = num(wcmc["window"])
    z30 = wcmc.loc[wnum.eq(30)].copy()
    if len(z30) >= 3:
        wcmc = z30

mimic = assoc.loc[assoc["cohort"].eq("MIMIC-IV")].copy()

def exact_three(df, label):
    rows = []
    for out in OUTCOMES:
        z = df.loc[df["outcome"].eq(out)]
        if len(z) != 1:
            raise RuntimeError(
                f"{label}: expected exactly one row for {out}; found {len(z)}.\n"
                f"{z.to_string(index=False)}"
            )
        rows.append(z.iloc[0])
    return pd.DataFrame(rows)

wcmc3 = exact_three(wcmc, "WCMC/NYP primary")
mimic3 = exact_three(mimic, "MIMIC-IV secondary replication")

panelB = pd.concat([
    wcmc3.assign(series="WCMC/NYP primary"),
    mimic3.assign(series="MIMIC-IV secondary replication"),
], ignore_index=True)

# ------------------------------------------------------------------------------------------
# PANEL A DATA — FINAL WCMC MORTALITY LOOKBACK
# ------------------------------------------------------------------------------------------

l_win = pick_col(look_raw, ["window_days", "lookback_days", "window", "lookback"], "lookback")
l_or = pick_col(
    look_raw,
    ["OR", "Firth_OR", "adjusted_OR", "OR_per_1SD", "odds_ratio", "adjusted_or", "firth_or"],
    "lookback OR",
)
l_lo = pick_col(look_raw, ["OR_CI95_low", "CI95_low", "CI_low", "lower"], "lookback CI low")
l_hi = pick_col(look_raw, ["OR_CI95_high", "CI95_high", "CI_high", "upper"], "lookback CI high")
l_n = pick_col(
    look_raw,
    ["N", "analysis_N", "sample_size", "n_patients"],
    "lookback N",
    required=False,
)
l_ev = pick_col(look_raw, ["events", "deaths", "n_events"], "lookback events", required=False)

def window_name(x):
    s = str(x).strip().lower()
    if (
        "unbound" in s
        or "unrestrict" in s
        or "unlimit" in s
        or "all preop" in s
        or "all_preop" in s
    ):
        return "Unrestricted"
    m = re.search(r"(\d+)", s)
    return f"{int(m.group(1))} days" if m else str(x)

look = pd.DataFrame({
    "window": look_raw[l_win].map(window_name),
    "OR": num(look_raw[l_or]),
    "CI_low": num(look_raw[l_lo]),
    "CI_high": num(look_raw[l_hi]),
    "N": num(look_raw[l_n]) if l_n else np.nan,
    "events": num(look_raw[l_ev]) if l_ev else np.nan,
})

LOOK_ORDER = ["30 days", "90 days", "180 days", "365 days", "Unrestricted"]
panelA = look.set_index("window").reindex(LOOK_ORDER).reset_index()

if panelA[["OR", "CI_low", "CI_high"]].isna().any().any():
    raise RuntimeError(
        "Could not resolve all five final WCMC lookback windows:\n"
        + panelA.to_string(index=False)
    )

# ------------------------------------------------------------------------------------------
# PANEL C DATA — FINAL INCREMENTAL AUROC
# ------------------------------------------------------------------------------------------

# The CURRENT final synthesis file has separate AUROC and AUPRC columns:
#   delta_AUROC
#   delta_AUROC_CI_low
#   delta_AUROC_CI_high
#   delta_AUPRC
#   delta_AUPRC_CI_low
#   delta_AUPRC_CI_high
#
# Therefore, do NOT fuzzy-match generic CI_low / CI_high names here.
# Use the exact AUROC columns so AUPRC intervals can never be selected accidentally.

required_inc_cols = [
    "cohort",
    "outcome",
    "delta_AUROC",
    "delta_AUROC_CI_low",
    "delta_AUROC_CI_high",
]

missing_inc = [c for c in required_inc_cols if c not in inc_raw.columns]
if missing_inc:
    raise RuntimeError(
        "CURRENT incremental-performance table is missing required AUROC columns:\n"
        + "\n".join(missing_inc)
        + f"\nObserved columns: {inc_raw.columns.tolist()}"
    )

inc = pd.DataFrame({
    "cohort": inc_raw["cohort"].map(cohort_name),
    "outcome": inc_raw["outcome"].map(outcome_name),
    "delta_AUROC": num(inc_raw["delta_AUROC"]),
    "CI_low": num(inc_raw["delta_AUROC_CI_low"]),
    "CI_high": num(inc_raw["delta_AUROC_CI_high"]),
})

rows = []
for cohort in ["WCMC/NYP", "MIMIC-IV"]:
    for out in OUTCOMES:
        z = inc.loc[(inc["cohort"] == cohort) & (inc["outcome"] == out)]
        if len(z) != 1:
            raise RuntimeError(
                f"Incremental AUROC: expected one row for {cohort} / {out}; found {len(z)}."
            )
        r = z.iloc[0].copy()
        r["series"] = (
            "WCMC/NYP primary"
            if cohort == "WCMC/NYP"
            else "MIMIC-IV secondary replication"
        )
        rows.append(r)

panelC = pd.DataFrame(rows)

# ------------------------------------------------------------------------------------------
# CURRENT-RUN GUARD — blocks accidental use of old postoperative files
# ------------------------------------------------------------------------------------------

wcmc_mort_or = float(
    wcmc3.loc[wcmc3["outcome"] == "30-day mortality", "OR"].iloc[0]
)
mimic_mort_or = float(
    mimic3.loc[mimic3["outcome"] == "30-day mortality", "OR"].iloc[0]
)
unbounded_or = float(
    panelA.loc[panelA["window"] == "Unrestricted", "OR"].iloc[0]
)
wcmc_delta = float(
    panelC.loc[
        (panelC["cohort"] == "WCMC/NYP")
        & (panelC["outcome"] == "30-day mortality"),
        "delta_AUROC",
    ].iloc[0]
)

if not (1.75 <= wcmc_mort_or <= 1.85):
    raise RuntimeError(
        f"CURRENT-RUN GUARD FAILED: WCMC mortality OR={wcmc_mort_or:.4f}; expected ~1.80."
    )
if not (1.30 <= mimic_mort_or <= 1.45):
    raise RuntimeError(
        f"CURRENT-RUN GUARD FAILED: MIMIC mortality OR={mimic_mort_or:.4f}; expected ~1.38."
    )
if not (1.65 <= unbounded_or <= 1.85):
    raise RuntimeError(
        f"CURRENT-RUN GUARD FAILED: unrestricted WCMC OR={unbounded_or:.4f}; expected ~1.76."
    )
if not (0.015 <= wcmc_delta <= 0.040):
    raise RuntimeError(
        f"CURRENT-RUN GUARD FAILED: WCMC mortality ΔAUROC={wcmc_delta:.4f}; expected ~0.028."
    )

print("\nCURRENT-RUN GUARD PASSED")
print(f"WCMC mortality OR = {wcmc_mort_or:.4f}")
print(f"MIMIC mortality OR = {mimic_mort_or:.4f}")
print(f"WCMC unrestricted mortality OR = {unbounded_or:.4f}")
print(f"WCMC mortality ΔAUROC = {wcmc_delta:.4f}")

# ------------------------------------------------------------------------------------------
# STYLE
# ------------------------------------------------------------------------------------------

CMAP = plt.get_cmap("plasma")
C_WCMC = CMAP(0.25)
C_MIMIC = CMAP(0.78)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12.5,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 14,
    "axes.labelweight": "bold",
    "xtick.labelsize": 11.5,
    "ytick.labelsize": 11.5,
    "legend.fontsize": 10.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

# ------------------------------------------------------------------------------------------
# FIGURE
# ------------------------------------------------------------------------------------------

fig, axes = plt.subplots(
    1, 3, figsize=(17.4, 6.2),
    gridspec_kw={"width_ratios": [1.05, 1.35, 1.25]},
)
axA, axB, axC = axes

# A — WCMC lookback sensitivity
yA = np.arange(len(LOOK_ORDER))
xA = panelA["OR"].to_numpy(float)
loA = panelA["CI_low"].to_numpy(float)
hiA = panelA["CI_high"].to_numpy(float)

axA.errorbar(
    xA, yA,
    xerr=np.vstack([xA-loA, hiA-xA]),
    fmt="o", markersize=7.2, linewidth=1.25, elinewidth=1.25,
    capsize=3.2, color=C_WCMC, ecolor=C_WCMC,
    markeredgecolor="black", markeredgewidth=0.55, zorder=3,
)
axA.axvline(1.0, ls="--", lw=0.9, color="0.62")
axA.set_yticks(yA)
axA.set_yticklabels(panelA["window"], fontweight="bold")
axA.invert_yaxis()
axA.set_xlabel("Adjusted OR per 1-SD greater\nECG-sex discordance")
axA.set_title("WCMC/NYP mortality lookback sensitivity")
axA.grid(axis="x", alpha=0.14)

for yy, r in zip(yA, panelA.itertuples(index=False)):
    axA.text(
        r.CI_high + 0.025, yy,
        f"{r.OR:.2f} ({r.CI_low:.2f}–{r.CI_high:.2f})",
        va="center", ha="left", fontsize=9.5,
    )

axA.set_xlim(
    min(0.95, float(loA.min()) - 0.06),
    float(hiA.max()) + 0.48,
)

# B — WCMC vs MIMIC associations
base_y = np.arange(len(OUTCOMES))
offsets = {
    "WCMC/NYP primary": -0.13,
    "MIMIC-IV secondary replication": +0.13,
}
colors = {
    "WCMC/NYP primary": C_WCMC,
    "MIMIC-IV secondary replication": C_MIMIC,
}
markers = {
    "WCMC/NYP primary": "o",
    "MIMIC-IV secondary replication": "s",
}

for series in offsets:
    z = (
        panelB.loc[panelB["series"] == series]
        .set_index("outcome").loc[OUTCOMES].reset_index()
    )
    x = z["OR"].to_numpy(float)
    lo = z["CI_low"].to_numpy(float)
    hi = z["CI_high"].to_numpy(float)
    yy = base_y + offsets[series]

    axB.errorbar(
        x, yy,
        xerr=np.vstack([x-lo, hi-x]),
        fmt=markers[series], markersize=6.8, linewidth=1.2,
        elinewidth=1.2, capsize=3.0, color=colors[series],
        ecolor=colors[series], markeredgecolor="black",
        markeredgewidth=0.5, label=series, zorder=3,
    )

axB.axvline(1.0, ls="--", lw=0.9, color="0.62")
axB.set_yticks(base_y)
axB.set_yticklabels(
    ["30-day mortality", "Postoperative AKI", "Postoperative ICU\nadmission"],
    fontweight="bold",
)
axB.invert_yaxis()
axB.set_xlabel("Adjusted OR per 1-SD greater\nECG-sex discordance")
axB.set_title("Adjusted postoperative associations")
axB.grid(axis="x", alpha=0.14)
axB.legend(frameon=False, loc="lower right")

all_lo = panelB["CI_low"].to_numpy(float)
all_hi = panelB["CI_high"].to_numpy(float)
axB.set_xlim(
    min(0.78, float(all_lo.min()) - 0.05),
    float(all_hi.max()) + 0.20,
)

# C — incremental AUROC
for series in offsets:
    z = (
        panelC.loc[panelC["series"] == series]
        .set_index("outcome").loc[OUTCOMES].reset_index()
    )
    x = z["delta_AUROC"].to_numpy(float)
    lo = z["CI_low"].to_numpy(float)
    hi = z["CI_high"].to_numpy(float)
    yy = base_y + offsets[series]

    axC.errorbar(
        x, yy,
        xerr=np.vstack([x-lo, hi-x]),
        fmt=markers[series], markersize=6.8, linewidth=1.2,
        elinewidth=1.2, capsize=3.0, color=colors[series],
        ecolor=colors[series], markeredgecolor="black",
        markeredgewidth=0.5, label=series, zorder=3,
    )

axC.axvline(0.0, ls="--", lw=0.9, color="0.62")
axC.set_yticks(base_y)
axC.set_yticklabels(
    ["30-day mortality", "Postoperative AKI", "Postoperative ICU\nadmission"],
    fontweight="bold",
)
axC.invert_yaxis()
axC.set_xlabel("ΔAUROC after adding ECG-sex discordance")
axC.set_title("Incremental discrimination")
axC.grid(axis="x", alpha=0.14)
axC.legend(frameon=False, loc="lower right")

c_lo = panelC["CI_low"].to_numpy(float)
c_hi = panelC["CI_high"].to_numpy(float)
span = max(float(c_hi.max() - c_lo.min()), 0.02)
axC.set_xlim(
    float(c_lo.min()) - 0.12*span,
    float(c_hi.max()) + 0.18*span,
)

for label, ax in zip("ABC", axes):
    ax.text(
        -0.15, 1.075, label,
        transform=ax.transAxes,
        fontsize=20, fontweight="bold",
        ha="left", va="top",
    )

fig.suptitle(
    "Postoperative Relevance of ECG-Sex Discordance",
    fontsize=18, fontweight="bold", y=0.985,
)

fig.subplots_adjust(
    left=0.085, right=0.985, bottom=0.19,
    top=0.82, wspace=0.48,
)

# ------------------------------------------------------------------------------------------
# SOURCE DATA / SAVE
# ------------------------------------------------------------------------------------------

pd.concat([
    add_block(panelA, "A", "WCMC/NYP mortality lookback sensitivity"),
    add_block(panelB, "B", "WCMC/NYP primary + MIMIC-IV secondary replication"),
    add_block(panelC, "C", "Incremental AUROC after ECG-sex discordance"),
], ignore_index=True, sort=False).to_csv(SOURCE_OUT, index=False)

fig.savefig(PNG_OUT, dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(PDF_OUT, bbox_inches="tight", facecolor="white")
fig.savefig(SVG_OUT, bbox_inches="tight", facecolor="white")

print("\n" + "=" * 120)
print("FIGURE 8 CURRENT FINAL SYNTHESIS COMPLETE")
print("=" * 120)

print("\nPanel A")
print(panelA.to_string(index=False))

print("\nPanel B")
print(panelB[["series","outcome","N","events","OR","CI_low","CI_high"]].to_string(index=False))

print("\nPanel C")
print(panelC[["series","outcome","delta_AUROC","CI_low","CI_high"]].to_string(index=False))

print("\nSaved:")
print(PNG_OUT)
print(PDF_OUT)
print(SVG_OUT)
print(SOURCE_OUT)

if IN_JUPYTER:
    from IPython.display import Image, display
    display(Image(filename=str(PNG_OUT)))
elif os.environ.get("DISPLAY") and not os.environ.get("SLURM_JOB_ID"):
    plt.show()
else:
    print("\nHeadless session: figure saved. Run this file in IPYNB to display inline.")

plt.close(fig)
print("\nDONE")
