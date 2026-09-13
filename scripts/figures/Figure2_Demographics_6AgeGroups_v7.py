#!/usr/bin/env python
# ==========================================================================================
# ECG-SEX — DEMOGRAPHICS ACROSS FINAL ANALYSIS COHORTS
# REVISED: SIX AGE STRATA TO MATCH FIGURE S3
# Age groups: 18–39, 40–49, 50–59, 60–69, 70–79, >=80 years
# ==========================================================================================
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch

# ============================================================
# PATHS
# ============================================================

RUN = Path(
    "/athena/madelab/scratch/iqh4001/ECG/ECG_Echo/Runs/"
    "Master_ECG_Sex_MIMIC_EchoNext/"
    "20260609_020154_MASTER_ECG_SEX_MIMIC_ECHONEXT"
)
WCMC = RUN / "wcmc_ecgsex_from_scratch"

MIMIC_CSV = Path(
    "/athena/madelab/scratch/iqh4001/ECG/ECG_mimic_IV/"
    "Data/Output/master_ecg_full_dataset_20260228_155252.csv"
)

WCMC_SPLIT = WCMC / "04_wcmc_finetune_from_mimic" / "splits"
WCMC_TRAIN = WCMC_SPLIT / "WCMC_ECGSex_train.csv"
WCMC_VAL   = WCMC_SPLIT / "WCMC_ECGSex_validation.csv"
WCMC_TEST  = WCMC_SPLIT / "WCMC_ECGSex_test.csv"

PTB_COHORT = (
    RUN / "ptbxl_ecgsex_from_scratch" / "01_cohort" /
    "PTBXL_ECGSex_Locked_OneECGPerPatient_AgeQC.csv"
)

CHAPMAN_COHORT = (
    RUN / "chapman_ningbo_ecgsex_from_scratch" / "01_cohort" /
    "ChapmanNingbo_ECGSex_Locked_Adult_KnownSex.csv"
)

OUT = RUN / "publication_figures" / "Supplementary"
OUT.mkdir(parents=True, exist_ok=True)
STEM = OUT / "Figure2_Comprehensive_Demographics_FinalAnalysisCohorts_6AgeGroups_v7"

EXPECTED_N = {
    "MIMIC-IV": 155626,
    "WCMC/NYP": 106868,
    "PTB-XL": 18752,
    "Chapman/Ningbo": 42997,
}
ORDER = list(EXPECTED_N)
SEX_NAMES = {0: "Female", 1: "Male"}
AGE_LABELS = ["18–39", "40–49", "50–59", "60–69", "70–79", "≥80"]

# ============================================================
# HELPERS
# ============================================================

def req(path):
    if not path.exists():
        raise FileNotFoundError(path)

def sex01(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in {"f","female","woman","0","false"}:
        return 0
    if s in {"m","male","man","1","true"}:
        return 1
    return np.nan

def standardize(dataset, d, subject, ecg, sex, age, ptb=False):
    out = pd.DataFrame({
        "dataset": dataset,
        "subject_id": d[subject].astype(str),
        "ecg_id": d[ecg].astype(str) if ecg is not None else d[subject].astype(str),
        "sex_binary": d[sex].map(sex01),
        "age_raw": pd.to_numeric(d[age], errors="coerce"),
    })
    if out["subject_id"].duplicated().any():
        raise RuntimeError(f"{dataset}: duplicate subject IDs")
    if out["ecg_id"].duplicated().any():
        raise RuntimeError(f"{dataset}: duplicate ECG IDs")
    if out["sex_binary"].isna().any():
        raise RuntimeError(f"{dataset}: unexpected missing/unknown sex")

    out["sex_binary"] = out["sex_binary"].astype(int)
    out["sex"] = out["sex_binary"].map(SEX_NAMES)

    # Exact ages for numerical summaries.
    out["age_exact"] = out["age_raw"].copy()
    out["ptb_privacy_ge90"] = False

    if ptb:
        privacy = out["age_raw"].eq(300)
        out.loc[privacy, "ptb_privacy_ge90"] = True
        out.loc[privacy, "age_exact"] = np.nan

    bad = out["age_exact"].notna() & (
        out["age_exact"].lt(18) | out["age_exact"].ge(120)
    )
    out.loc[bad, "age_exact"] = np.nan

    # age_plot is visualization-only.
    out["age_plot"] = out["age_exact"]
    if ptb:
        out.loc[out["ptb_privacy_ge90"], "age_plot"] = 90.0

    out["age_group"] = pd.cut(
        out["age_exact"],
        bins=[18,40,50,60,70,80,np.inf],
        labels=AGE_LABELS,
        right=False,
        include_lowest=True,
    ).astype(object)
    if ptb:
        out.loc[out["ptb_privacy_ge90"], "age_group"] = "≥80"

    return out.reset_index(drop=True)

# ============================================================
# LOAD EXACT FINAL COHORTS
# ============================================================

# MIMIC — match original ECG-sex loader: path + patient + known sex.
req(MIMIC_CSV)
m = pd.read_csv(MIMIC_CSV, low_memory=False)
for c in ["subject_id","study_id","gender","age","path"]:
    if c not in m.columns:
        raise RuntimeError(f"MIMIC missing {c}")
m["_sex"] = m["gender"].map(sex01)
m = m.dropna(subset=["subject_id","study_id","path","_sex"]).copy()
m["_sex"] = m["_sex"].astype(int)
if m["subject_id"].astype(str).duplicated().any():
    m = m.drop_duplicates("subject_id", keep="first")
MIMIC = standardize("MIMIC-IV", m, "subject_id", "study_id", "_sex", "age")

# WCMC — exact successful train + val + locked test; NOT the 933k-row master.
for p in [WCMC_TRAIN,WCMC_VAL,WCMC_TEST]:
    req(p)
wparts = []
for split, p in [("train",WCMC_TRAIN),("validation",WCMC_VAL),("test",WCMC_TEST)]:
    x = pd.read_csv(p, low_memory=False)
    for c in ["patient_key","ecg_key","sex_binary","age_at_ecg"]:
        if c not in x.columns:
            raise RuntimeError(f"{p.name} missing {c}")
    x["_split"] = split
    wparts.append(x)
w = pd.concat(wparts, ignore_index=True)
if w["patient_key"].astype(str).duplicated().any():
    raise RuntimeError("WCMC patient overlap across splits")
if w["ecg_key"].astype(str).duplicated().any():
    raise RuntimeError("WCMC ECG overlap across splits")
WCMC_DF = standardize(
    "WCMC/NYP", w, "patient_key", "ecg_key", "sex_binary", "age_at_ecg"
)

# PTB-XL — exact locked one-ECG-per-patient cohort.
req(PTB_COHORT)
p = pd.read_csv(PTB_COHORT, low_memory=False)
for c in ["patient_id","ecg_id","sex_binary","age"]:
    if c not in p.columns:
        raise RuntimeError(f"PTB-XL missing {c}")
PTB = standardize("PTB-XL", p, "patient_id", "ecg_id", "sex_binary", "age", ptb=True)

# Chapman/Ningbo — exact locked adult known-sex cohort.
req(CHAPMAN_COHORT)
c = pd.read_csv(CHAPMAN_COHORT, low_memory=False)
record_col = "record_id" if "record_id" in c.columns else "record_base"
for col in [record_col,"sex_binary","age"]:
    if col not in c.columns:
        raise RuntimeError(f"Chapman/Ningbo missing {col}")
CHAP = standardize(
    "Chapman/Ningbo", c, record_col, record_col, "sex_binary", "age"
)

cohorts = [MIMIC,WCMC_DF,PTB,CHAP]

# Hard-stop on wrong cohort.
for d in cohorts:
    name = d["dataset"].iloc[0]
    if len(d) != EXPECTED_N[name]:
        raise RuntimeError(
            f"{name}: expected N={EXPECTED_N[name]:,}, observed N={len(d):,}"
        )

all_df = pd.concat(cohorts, ignore_index=True)

# ============================================================
# SUMMARIES
# ============================================================

summary = []
age_groups = []

for name in ORDER:
    d = all_df[all_df["dataset"].eq(name)].copy()
    ages = d["age_exact"].dropna()
    fn = int(d["sex_binary"].eq(0).sum())
    mn = int(d["sex_binary"].eq(1).sum())
    privacy = int(d["ptb_privacy_ge90"].sum())
    agecat = d["age_group"].notna()

    summary.append({
        "dataset": name,
        "N_analysis": len(d),
        "Female_N": fn,
        "Male_N": mn,
        "Female_pct": 100*fn/len(d),
        "Male_pct": 100*mn/len(d),
        "Age_exact_N": len(ages),
        "Age_exact_pct": 100*len(ages)/len(d),
        "Age_categorizable_N": int(agecat.sum()),
        "Age_categorizable_pct": 100*agecat.sum()/len(d),
        "PTB_privacy_age_ge90_N": privacy,
        "Age_mean": ages.mean(),
        "Age_SD": ages.std(),
        "Age_median": ages.median(),
        "Age_Q1": ages.quantile(.25),
        "Age_Q3": ages.quantile(.75),
        "Age_min": ages.min(),
        "Age_max": ages.max(),
    })

    dc = d[agecat].copy()
    denom = len(dc)
    for ag in AGE_LABELS:
        n = int(dc["age_group"].eq(ag).sum())
        age_groups.append({
            "dataset": name, "age_group": ag, "N": n,
            "denominator": denom, "Percent": 100*n/denom if denom else np.nan
        })

summary = pd.DataFrame(summary)
age_groups = pd.DataFrame(age_groups)
summary.to_csv(str(STEM) + "_Summary.csv", index=False)
age_groups.to_csv(str(STEM) + "_AgeGroups.csv", index=False)
all_df.to_csv(str(STEM) + "_LongDemographics.csv", index=False)

audit = pd.DataFrame([{
    "dataset": n,
    "expected_N": EXPECTED_N[n],
    "observed_N": int((all_df["dataset"] == n).sum()),
    "unique_subjects": all_df.loc[all_df["dataset"] == n, "subject_id"].nunique(),
    "unique_ECGs": all_df.loc[all_df["dataset"] == n, "ecg_id"].nunique(),
    "exact_age_N": all_df.loc[all_df["dataset"] == n, "age_exact"].notna().sum(),
    "age_band_N": all_df.loc[all_df["dataset"] == n, "age_group"].notna().sum(),
    "PTB_privacy_ge90_N": all_df.loc[
        all_df["dataset"] == n, "ptb_privacy_ge90"
    ].sum(),
} for n in ORDER])
audit.to_csv(str(STEM) + "_CohortAudit.csv", index=False)

print("="*125)
print("FINAL ANALYSIS-COHORT DEMOGRAPHICS")
print("="*125)
print("\nCohort audit:")
print(audit.to_string(index=False))
print("\nSummary:")
print(summary.round(2).to_string(index=False))

# ============================================================
# STYLE
# ============================================================

cmap = plt.get_cmap("plasma")
DCLR = dict(zip(ORDER, [cmap(.14),cmap(.36),cmap(.62),cmap(.86)]))
SCLR = {"Female": cmap(.22), "Male": cmap(.82)}
ACLR = dict(zip(AGE_LABELS, [cmap(.10), cmap(.24), cmap(.40), cmap(.56), cmap(.72), cmap(.88)]))

plt.rcParams.update({
    "font.family":"DejaVu Sans",
    "font.size":10,
    "axes.titlesize":11.5,
    "axes.labelsize":10,
    "xtick.labelsize":8.6,
    "ytick.labelsize":8.6,
    "legend.fontsize":8.2,
    "axes.spines.top":False,
    "axes.spines.right":False,
    "figure.dpi":120,
    "savefig.dpi":600,
})

fig, axs = plt.subplots(2,3,figsize=(18.0,10.8))
A,B,C,D,E,F = axs.ravel()
x = np.arange(len(ORDER))
S = summary.set_index("dataset").loc[ORDER]

# A cohort size
vals = S["N_analysis"].to_numpy(float)
bars = A.bar(x, vals, color=[DCLR[z] for z in ORDER], edgecolor="black", lw=.5)
A.set_xticks(x); A.set_xticklabels(ORDER,rotation=22,ha="right")
A.set_ylabel("Participants"); A.set_title("Analysis cohort size")
A.grid(axis="y",alpha=.14); A.set_ylim(0,vals.max()*1.18)
for b,v in zip(bars,vals):
    A.text(b.get_x()+b.get_width()/2,v+vals.max()*.015,f"{int(v):,}",
           ha="center",fontsize=8.2,fontweight="bold")

# B sex
fp = S["Female_pct"].to_numpy(float); mp = S["Male_pct"].to_numpy(float)
B.bar(x,fp,color=SCLR["Female"],edgecolor="black",lw=.5,label="Female")
B.bar(x,mp,bottom=fp,color=SCLR["Male"],edgecolor="black",lw=.5,label="Male")
B.set_xticks(x); B.set_xticklabels(ORDER,rotation=22,ha="right")
B.set_ylim(0,100); B.set_ylabel("Composition (%)"); B.set_title("Sex composition")
B.grid(axis="y",alpha=.12)
for i,n in enumerate(ORDER):
    B.text(i,fp[i]/2,f"{int(S.loc[n,'Female_N']):,}\n{fp[i]:.1f}%",
           ha="center",va="center",fontsize=7)
    B.text(i,fp[i]+mp[i]/2,f"{int(S.loc[n,'Male_N']):,}\n{mp[i]:.1f}%",
           ha="center",va="center",fontsize=7)

# C age distribution. PTB age=300 appears at 90 only for visualization.
arrs = [all_df.loc[
    all_df["dataset"].eq(n) & all_df["age_plot"].notna(),"age_plot"
].to_numpy(float) for n in ORDER]
bp = C.boxplot(arrs,patch_artist=True,widths=.62,showfliers=False,
               medianprops={"color":"black","linewidth":1.5})
for box,n in zip(bp["boxes"],ORDER):
    box.set_facecolor(DCLR[n]); box.set_alpha(.72); box.set_edgecolor("black")
C.set_xticks(range(1,len(ORDER)+1)); C.set_xticklabels(ORDER,rotation=22,ha="right")
C.set_ylabel("Age at ECG, years"); C.set_title("Age distribution"); C.grid(axis="y",alpha=.14)
C.set_ylim(15,116)
for i,n in enumerate(ORDER,1):
    C.text(i,108,f"{S.loc[n,'Age_median']:.0f} "
           f"[{S.loc[n,'Age_Q1']:.0f}–{S.loc[n,'Age_Q3']:.0f}]",
           ha="center",fontsize=7.2)

# D age by sex
data=[]; pos=[]; cols=[]; ticks=[]; cur=1.
for n in ORDER:
    for j,sx in enumerate(["Female","Male"]):
        data.append(all_df.loc[
            all_df["dataset"].eq(n) & all_df["sex"].eq(sx) &
            all_df["age_plot"].notna(),"age_plot"
        ].to_numpy(float))
        pos.append(cur + .72*j); cols.append(SCLR[sx])
    ticks.append(cur+.36); cur += 2.05
bp2 = D.boxplot(data,positions=pos,patch_artist=True,widths=.54,showfliers=False,
                medianprops={"color":"black","linewidth":1.4})
for box,col in zip(bp2["boxes"],cols):
    box.set_facecolor(col); box.set_alpha(.72); box.set_edgecolor("black")
D.set_xticks(ticks); D.set_xticklabels(ORDER,rotation=22,ha="right")
D.set_ylabel("Age at ECG, years"); D.set_title("Age distribution stratified by sex")
D.set_ylim(15,112); D.grid(axis="y",alpha=.14)

# E age bands
bottom=np.zeros(len(ORDER))
for ag in AGE_LABELS:
    vv=np.array([
        age_groups.loc[
            age_groups["dataset"].eq(n) & age_groups["age_group"].eq(ag),"Percent"
        ].iloc[0] for n in ORDER
    ],float)
    E.bar(x,vv,bottom=bottom,color=ACLR[ag],edgecolor="black",lw=.4,label=ag)
    for xx,v,base in zip(x,vv,bottom):
        if v>=7:
            E.text(xx,base+v/2,f"{v:.1f}%",ha="center",va="center",fontsize=6.9)
    bottom += vv
E.set_xticks(x); E.set_xticklabels(ORDER,rotation=22,ha="right")
E.set_ylim(0,100); E.set_ylabel("Age-categorizable cohort (%)")
E.set_title("Age-group composition (six strata)"); E.grid(axis="y",alpha=.12)

# F dataset allocation across study stages
# ------------------------------------------------------------
# This panel summarizes the role of each dataset in the final study.
#
# MIMIC-IV:
#   train = 132,282
#   validation = 23,344
#   no locked test used from MIMIC in the primary transfer experiment
#
# WCMC/NYP:
#   train = 74,807
#   validation = 10,687
#   locked test = 21,374
#
# PTB-XL and Chapman/Ningbo:
#   used as zero-shot external test cohorts in the final external
#   generalization analysis.
# ------------------------------------------------------------

SPLIT_COUNTS = {
    "MIMIC-IV": {
        "Training": 132282,
        "Validation": 23344,
        "Test / external": 0,
    },
    "WCMC/NYP": {
        "Training": 74807,
        "Validation": 10687,
        "Test / external": 21374,
    },
    "PTB-XL": {
        "Training": 0,
        "Validation": 0,
        "Test / external": 18752,
    },
    "Chapman/Ningbo": {
        "Training": 0,
        "Validation": 0,
        "Test / external": 42997,
    },
}

# Hard QC: allocations must sum to the displayed cohort size.
for dataset in ORDER:
    allocated_n = sum(SPLIT_COUNTS[dataset].values())
    if allocated_n != EXPECTED_N[dataset]:
        raise RuntimeError(
            f"{dataset}: Panel F allocation does not match cohort N. "
            f"Allocated {allocated_n:,}; expected {EXPECTED_N[dataset]:,}"
        )

stage_colors = {
    "Training": cmap(.20),
    "Validation": cmap(.52),
    "Test / external": cmap(.86),
}

split_bottom = np.zeros(len(ORDER), dtype=float)

for stage in ["Training", "Validation", "Test / external"]:
    values = np.array(
        [SPLIT_COUNTS[d][stage] for d in ORDER],
        dtype=float,
    )

    barsF = F.bar(
        x,
        values,
        bottom=split_bottom,
        width=.66,
        color=stage_colors[stage],
        edgecolor="black",
        linewidth=.45,
        label=stage,
    )

    for xx, value, base in zip(x, values, split_bottom):
        if value <= 0:
            continue

        # Label all visible study allocations.
        F.text(
            xx,
            base + value / 2,
            f"{int(value):,}",
            ha="center",
            va="center",
            fontsize=7.2,
        )

    split_bottom += values

F.set_xticks(x)
F.set_xticklabels(ORDER, rotation=22, ha="right")
F.set_ylabel("Participants / ECGs")
F.set_title("Dataset allocation across study stages")
F.grid(axis="y", alpha=.12)

F.legend(
    frameon=False,
    loc="upper right",
)


# ------------------------------------------------------------
# FIGURE-LEVEL SHARED LEGENDS
# ------------------------------------------------------------

# Shared sex legend for Panels B and D.
sex_handles = [
    Patch(
        facecolor=SCLR["Female"],
        edgecolor="black",
        linewidth=.5,
        label="Female",
    ),
    Patch(
        facecolor=SCLR["Male"],
        edgecolor="black",
        linewidth=.5,
        label="Male",
    ),
]

sex_legend = fig.legend(
    handles=sex_handles,
    loc="upper center",
    bbox_to_anchor=(0.50, 0.948),
    ncol=2,
    frameon=False,
    columnspacing=1.8,
    handlelength=1.6,
    borderaxespad=0.0,
)

# Shared age-group legend for Panel E.
age_handles = [
    Patch(
        facecolor=ACLR[ag],
        edgecolor="black",
        linewidth=.4,
        label=ag,
    )
    for ag in AGE_LABELS
]

age_legend = fig.legend(
    handles=age_handles,
    title="Age, years",
    loc="lower center",
    bbox_to_anchor=(0.50, 0.012),
    ncol=6,
    frameon=False,
    columnspacing=1.4,
    handlelength=1.5,
)

# Panel labels
for lab,ax in zip("ABCDEF",[A,B,C,D,E,F]):
    ax.text(-.12,1.06,lab,transform=ax.transAxes,fontsize=15.5,
            fontweight="bold",ha="left",va="top")

fig.suptitle(
    "Demographic characteristics across final ECG analysis cohorts",
    fontsize=16,
    fontweight="bold",
    y=.985
)

fig.subplots_adjust(left=.070,right=.988,bottom=.115,top=.895,wspace=.30,hspace=.34)

fig.savefig(str(STEM)+".png",dpi=600,bbox_inches="tight",facecolor="white")
fig.savefig(str(STEM)+".pdf",bbox_inches="tight",facecolor="white")
fig.savefig(str(STEM)+".svg",bbox_inches="tight",facecolor="white")
plt.show()

print("\nSaved:")
print(str(STEM)+".png")
print(str(STEM)+".pdf")
print(str(STEM)+".svg")
print(str(STEM)+"_Summary.csv")
print(str(STEM)+"_AgeGroups.csv")
print(str(STEM)+"_LongDemographics.csv")
print(str(STEM)+"_CohortAudit.csv")
print("\nDONE")
