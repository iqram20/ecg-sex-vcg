#!/usr/bin/env python3
# ==========================================================================================
# WCMC ECG-SEX — POSTOPERATIVE OUTCOME EXTENSION
# ALL PREOPERATIVE ECGs FROM LOCKED-TEST PATIENTS + FROZEN FINAL 5-SEED VCG ENSEMBLE
# ==========================================================================================
#
# Patient universe: original WCMC/NYP locked-test patients only.
# For each patient/window: first surgery with a qualifying ECG, then closest ECG before surgery.
# Primary window: <=30 days. Sensitivity window: <=90 days.
# Exact final VCG model/preprocessing is imported from the original training script.
# No retraining, recalibration, or threshold adjustment.
# ==========================================================================================

from __future__ import annotations

import os
os.environ["VCGFINAL_MODE"] = "FINAL"
os.environ.setdefault("VCGFINAL_BATCH_SIZE", "128")
os.environ.setdefault("VCGFINAL_NUM_WORKERS", "8")

import argparse
import gc
import importlib.util
import json
import math
import sys
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import statsmodels.api as sm
from scipy.stats import chi2
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ==========================================================================================
# PATHS / CONSTANTS
# ==========================================================================================

CODE_DIR = Path("/athena/madelab/scratch/iqh4001/ECG/ECG-gender/Code")
FINAL_MODEL_SOURCE = CODE_DIR / "WCMC_ECGSex_VCGGeometry_ResNet_5Seed_LockedTest.py"

RUN_DIR = Path(
    "/athena/madelab/scratch/iqh4001/ECG/ECG_Echo/Runs/"
    "Master_ECG_Sex_MIMIC_EchoNext/20260609_020154_MASTER_ECG_SEX_MIMIC_ECHONEXT"
)
WCMC_ROOT = RUN_DIR / "wcmc_ecgsex_from_scratch"
LOCKED_SPLIT = WCMC_ROOT / "04_wcmc_finetune_from_mimic/splits/WCMC_ECGSex_test.csv"
LOCKED_PRED = (
    WCMC_ROOT / "19_vcg_geometry_resnet_5seed_locked_wcmc/"
    "final_5seed_ensemble_bootstrap/WCMC_LockedTest_Aligned_5SeedEnsemble_Predictions.csv"
)
CHECKPOINT_ROOT = WCMC_ROOT / "19_vcg_geometry_resnet_5seed_locked_wcmc/seeds"

MASTER_PARQUET = Path(
    "/athena/madelab/scratch/iqh4001/Azure_Cradle_ECG/Data/"
    "ECG-Echo_MetaData/ecg_echo_master_cohort_v1.parquet"
)
RAW_ECG_DIR = Path(
    "/athena/madelab/scratch/iqh4001/Azure_Cradle_ECG/Data/cradle-ecg/raw_ecg"
)
DB_PATH = Path(
    "/athena/madelab/scratch/iqh4001/Azure_Cradle_ECG/Data/"
    "ECG-Echo_MetaData/WCMC_MPOG_TRUE_MASTER_2018_2023.duckdb"
)
OUT_DEFAULT = Path(
    "/athena/madelab/scratch/iqh4001/ECG/ECG-gender/Results/"
    "WCMC_PostopOutcome_AllPreopECG_Frozen5Seed"
)

SEEDS = [11, 22, 33, 42, 55]
PRIMARY_WINDOW = 30
SENSITIVITY_WINDOW = 90
EPS = 1e-6


def ckpt_path(seed: int) -> Path:
    return CHECKPOINT_ROOT / f"seed_{seed}/best_VCGGeometryResNet_WCMC.pt"


def norm_key(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


# ==========================================================================================
# IMPORT EXACT FINAL MODEL SOURCE
# ==========================================================================================

def import_final_module():
    spec = importlib.util.spec_from_file_location("final_vcg_source", FINAL_MODEL_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not import final model source.")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["final_vcg_source"] = mod
    spec.loader.exec_module(mod)
    for name in ["VCGGeometryResNet", "WCMCDataset", "collate_fn"]:
        if not hasattr(mod, name):
            raise RuntimeError(f"Final source missing required object: {name}")
    return mod


# ==========================================================================================
# DATA LOADERS
# ==========================================================================================

def load_locked_test():
    df = pd.read_csv(LOCKED_SPLIT, low_memory=False)
    need = ["patient_key", "ecg_key", "sex_binary"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise RuntimeError(f"Locked split missing {miss}; columns={list(df.columns)}")

    age_col = "age_at_ecg" if "age_at_ecg" in df.columns else (
        "ecg_age_at_ecg" if "ecg_age_at_ecg" in df.columns else None
    )

    out = df[["patient_key", "ecg_key", "sex_binary"] + ([age_col] if age_col else [])].copy()
    out["patient_key"] = norm_key(out["patient_key"])
    out["locked_ecg_key"] = norm_key(out["ecg_key"])
    out["sex_binary"] = pd.to_numeric(out["sex_binary"], errors="coerce")
    out = out[out["sex_binary"].isin([0, 1])].copy()
    out["sex_binary"] = out["sex_binary"].astype(int)
    out["locked_age"] = pd.to_numeric(out[age_col], errors="coerce") if age_col else np.nan
    out = out[["patient_key", "locked_ecg_key", "sex_binary", "locked_age"]]

    if out["patient_key"].duplicated().any():
        raise RuntimeError("Locked split is not one row per patient.")
    print(f"Locked-test patients: {len(out):,}")
    return out


def load_all_ecgs(locked):
    cols = [
        "patient_key", "ecg_key", "acquisition_datetime",
        "raw_ecg_blob_path", "ecg_age_at_ecg",
    ]
    ecg = pd.read_parquet(MASTER_PARQUET, columns=cols)
    ecg = ecg.rename(columns={
        "acquisition_datetime": "ecg_time",
        "raw_ecg_blob_path": "raw_path_metadata",
        "ecg_age_at_ecg": "ecg_age",
    })
    ecg["patient_key"] = norm_key(ecg["patient_key"])
    ecg["ecg_key"] = norm_key(ecg["ecg_key"])
    ecg["ecg_time"] = pd.to_datetime(ecg["ecg_time"], errors="coerce")
    ecg["ecg_age"] = pd.to_numeric(ecg["ecg_age"], errors="coerce")
    test_ids = set(locked["patient_key"])
    ecg = ecg[ecg["patient_key"].isin(test_ids)].copy()
    ecg = (
        ecg.sort_values(["patient_key", "ecg_time", "ecg_key"])
        .drop_duplicates(["patient_key", "ecg_key"], keep="first")
        .reset_index(drop=True)
    )
    print(f"All ECGs for locked patients: {len(ecg):,} / patients={ecg.patient_key.nunique():,}")
    return ecg


def load_surgeries(locked):
    con = duckdb.connect(str(DB_PATH), read_only=True)
    surg = con.execute("""
        SELECT
            CAST(patient_key AS VARCHAR) AS patient_key,
            TRY_CAST(AnesthesiaStart_Value AS TIMESTAMP) AS surgery_time,
            mortality_30d AS mortality_30d_raw,
            aki_raw,
            postop_icu AS postop_icu_raw,
            ASA_Class_Code AS asa_raw
        FROM main.surgeries
        WHERE patient_key IS NOT NULL
          AND AnesthesiaStart_Value IS NOT NULL
    """).df()
    con.close()
    surg["patient_key"] = norm_key(surg["patient_key"])
    surg["surgery_time"] = pd.to_datetime(surg["surgery_time"], errors="coerce")
    surg = surg[surg["patient_key"].isin(set(locked["patient_key"]))].copy()
    surg = surg.drop_duplicates().reset_index(drop=True)
    print(f"Surgery rows for locked patients: {len(surg):,} / patients={surg.patient_key.nunique():,}")
    return surg


def resolve_raw_path(meta_path, ecg_key):
    if pd.notna(meta_path):
        s = str(meta_path).strip().replace("\\", "/")
        if s:
            p = Path(s)
            if p.exists():
                return str(p)
            p = RAW_ECG_DIR / Path(s).name
            if p.exists():
                return str(p)
    p = RAW_ECG_DIR / f"{str(ecg_key).strip()}.npy"
    return str(p) if p.exists() else None


# ==========================================================================================
# COHORT SELECTION
# ==========================================================================================

def build_candidate_pairs(locked, ecg, surg):
    pair = surg.merge(ecg, on="patient_key", how="inner")
    pair["days_ecg_to_surgery"] = (
        pair["surgery_time"] - pair["ecg_time"]
    ).dt.total_seconds() / 86400.0
    pair = pair[
        pair["ecg_time"].notna()
        & pair["surgery_time"].notna()
        & pair["days_ecg_to_surgery"].gt(0)
        & pair["days_ecg_to_surgery"].le(SENSITIVITY_WINDOW)
    ].copy()

    unique_ecg = pair[["patient_key", "ecg_key", "raw_path_metadata"]].drop_duplicates()
    print(f"Resolving paths for {len(unique_ecg):,} candidate ECGs...")
    unique_ecg["raw_local_path"] = [
        resolve_raw_path(p, k)
        for p, k in zip(unique_ecg["raw_path_metadata"], unique_ecg["ecg_key"])
    ]
    pair = pair.merge(
        unique_ecg[["patient_key", "ecg_key", "raw_local_path"]],
        on=["patient_key", "ecg_key"], how="left", validate="many_to_one"
    )
    before = len(pair)
    pair = pair[pair["raw_local_path"].notna()].copy()
    print(f"Pairs <=90d: {before:,}; with waveform={len(pair):,}")

    pair = pair.merge(
        locked, on="patient_key", how="inner", validate="many_to_one"
    )
    pair["age"] = pd.to_numeric(pair["ecg_age"], errors="coerce")
    m = pair["age"].isna()
    pair.loc[m, "age"] = pd.to_numeric(pair.loc[m, "locked_age"], errors="coerce")
    return pair


def select_window(pair, days):
    d = pair[pair["days_ecg_to_surgery"].le(days)].copy()
    # First surgery with a qualifying ECG; within that surgery, closest prior ECG.
    d = d.sort_values(
        ["patient_key", "surgery_time", "days_ecg_to_surgery", "ecg_time"],
        ascending=[True, True, True, False],
    )
    d = d.drop_duplicates("patient_key", keep="first").copy()
    d["window_days"] = days
    print(
        f"{days}d selected cohort: N={len(d):,} | F={(d.sex_binary==0).sum():,} | "
        f"M={(d.sex_binary==1).sum():,} | median ECG->OR={d.days_ecg_to_surgery.median():.2f}d"
    )
    return d


# ==========================================================================================
# FROZEN 5-SEED INFERENCE
# ==========================================================================================

def make_loader(mod, df, batch_size, num_workers):
    ds = mod.WCMCDataset(df[["patient_key", "ecg_key", "sex_binary", "raw_local_path"]].copy())
    kwargs = dict(
        dataset=ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, drop_last=False, collate_fn=mod.collate_fn,
    )
    if num_workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(**kwargs)


@torch.no_grad()
def predict_seed(mod, model, loader, device):
    model.eval()
    rows, failures = [], []
    for bi, batch in enumerate(loader, 1):
        failures.extend(batch["invalid"])
        if batch["x_base"] is None:
            continue
        xb = batch["x_base"].to(device, non_blocking=True)
        xv = batch["x_vcg8"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=True):
            comp = model(xb, xv, return_components=True)
        p = torch.sigmoid(comp["logit"]).float().cpu().numpy()
        g = comp["gate"].float().cpu().numpy()
        dl = comp["delta"].float().cpu().numpy()
        for i, item in enumerate(batch["valid"]):
            rows.append({
                "patient_key": item["patient_key"], "ecg_key": item["ecg_key"],
                "prob_male": float(p[i]), "gate": float(g[i]), "delta": float(dl[i]),
            })
        if bi % 25 == 0:
            print(f"  batch {bi}/{len(loader)} | pred={len(rows):,} | fail={len(failures):,}")
    fail = pd.DataFrame([{
        "patient_key": x["patient_key"], "ecg_key": x["ecg_key"],
        "path": x["path"], "error": x["error"],
    } for x in failures])
    return pd.DataFrame(rows), fail


def infer_ensemble(mod, union, outdir, batch_size, num_workers):
    device = torch.device("cuda")
    loader = make_loader(mod, union, batch_size, num_workers)
    merged = union[["patient_key", "ecg_key", "sex_binary", "raw_local_path"]].copy()
    failures = []

    for seed in SEEDS:
        cp = ckpt_path(seed)
        print("\n" + "="*100)
        print(f"FROZEN SEED {seed}: {cp}")
        model = mod.VCGGeometryResNet().to(device)
        chk = torch.load(cp, map_location=device, weights_only=False)
        model.load_state_dict(chk["model_state_dict"], strict=True)
        pr, fl = predict_seed(mod, model, loader, device)
        pr = pr.rename(columns={
            "prob_male": f"prob_seed_{seed}",
            "gate": f"gate_seed_{seed}",
            "delta": f"delta_seed_{seed}",
        })
        merged = merged.merge(
            pr[["patient_key", "ecg_key", f"prob_seed_{seed}", f"gate_seed_{seed}", f"delta_seed_{seed}"]],
            on=["patient_key", "ecg_key"], how="left", validate="one_to_one"
        )
        if len(fl):
            fl["seed"] = seed
            failures.append(fl)
        del model, chk, pr
        gc.collect(); torch.cuda.empty_cache()

    pc = [f"prob_seed_{s}" for s in SEEDS]
    gcx = [f"gate_seed_{s}" for s in SEEDS]
    dc = [f"delta_seed_{s}" for s in SEEDS]
    merged["prob_vcg"] = merged[pc].mean(axis=1, skipna=False)
    merged["vcg_gate_mean"] = merged[gcx].mean(axis=1, skipna=False)
    merged["vcg_delta_mean"] = merged[dc].mean(axis=1, skipna=False)
    merged.to_csv(outdir / "02_selected_preopECG_5seed_predictions.csv", index=False)
    pd.concat(failures, ignore_index=True).to_csv(
        outdir / "02b_inference_failures.csv", index=False
    ) if failures else pd.DataFrame(columns=["patient_key","ecg_key","path","error","seed"]).to_csv(
        outdir / "02b_inference_failures.csv", index=False
    )
    return merged


# ==========================================================================================
# REPRODUCIBILITY QC + DISCORDANCE STANDARDIZATION
# ==========================================================================================

def qc_against_locked(new_pred, outdir):
    old = pd.read_csv(LOCKED_PRED, low_memory=False)[["ecg_key", "prob_vcg"]].copy()
    old["ecg_key"] = norm_key(old["ecg_key"])
    old = old.rename(columns={"prob_vcg": "prob_vcg_original"})
    q = new_pred[["ecg_key", "prob_vcg"]].merge(old, on="ecg_key", how="inner")
    if len(q):
        q["abs_diff"] = (q["prob_vcg"] - q["prob_vcg_original"]).abs()
        print(f"QC overlap={len(q):,} | mean diff={q.abs_diff.mean():.10f} | max diff={q.abs_diff.max():.10f}")
        q.to_csv(outdir / "03_inference_reproducibility_QC.csv", index=False)
        if q["abs_diff"].max() > 5e-4:
            raise RuntimeError("Inference reproducibility QC failed: max probability difference >5e-4.")
    else:
        print("WARNING: no selected ECG overlaps original locked ECG keys for QC.")


def add_discordance(pred):
    old = pd.read_csv(LOCKED_PRED, low_memory=False)
    old["sex_true_binary"] = pd.to_numeric(old["sex_true_binary"], errors="coerce")
    old["prob_vcg"] = pd.to_numeric(old["prob_vcg"], errors="coerce")
    old = old.dropna(subset=["sex_true_binary", "prob_vcg"])

    def raw(prob, sex):
        p = np.clip(np.asarray(prob, float), EPS, 1-EPS)
        logit = np.log(p/(1-p))
        sex = np.asarray(sex, int)
        return -np.where(sex == 1, logit, -logit)

    old["disc_raw"] = raw(old["prob_vcg"], old["sex_true_binary"])
    ref = {}
    for sex in [0, 1]:
        x = old.loc[old.sex_true_binary.astype(int).eq(sex), "disc_raw"]
        ref[sex] = {"mean": float(x.mean()), "sd": float(x.std(ddof=0)), "N": int(len(x))}

    out = pred.copy()
    out["ecg_sex_discordance_raw"] = raw(out["prob_vcg"], out["sex_binary"])
    out["ecg_sex_discordance_z"] = np.nan
    for sex in [0, 1]:
        m = out.sex_binary.eq(sex)
        out.loc[m, "ecg_sex_discordance_z"] = (
            out.loc[m, "ecg_sex_discordance_raw"] - ref[sex]["mean"]
        ) / ref[sex]["sd"]
    return out, ref


# ==========================================================================================
# OUTCOME CLEANING / ANALYSIS
# ==========================================================================================

def clean_binary(s):
    x = pd.to_numeric(s, errors="coerce")
    out = pd.Series(np.nan, index=s.index, dtype=float)
    out.loc[x.eq(0)] = 0.0; out.loc[x.eq(1)] = 1.0
    return out


def clean_aki(s):
    x = pd.to_numeric(s, errors="coerce")
    out = pd.Series(np.nan, index=s.index, dtype=float)
    out.loc[x.eq(0)] = 0.0; out.loc[x.isin([1,2,3])] = 1.0
    return out


def clean_asa(s):
    x = pd.to_numeric(s, errors="coerce")
    return x.where(x.between(1,5))


def prepare(selected, pred):
    d = selected.merge(
        pred[["patient_key","ecg_key","prob_vcg","vcg_gate_mean","vcg_delta_mean",
              "ecg_sex_discordance_raw","ecg_sex_discordance_z"]],
        on=["patient_key","ecg_key"], how="left", validate="one_to_one"
    )
    d = d[d["prob_vcg"].notna()].copy()
    d["mortality_30d"] = clean_binary(d["mortality_30d_raw"])
    d["aki"] = clean_aki(d["aki_raw"])
    d["postop_icu"] = clean_binary(d["postop_icu_raw"])
    d["asa"] = clean_asa(d["asa_raw"])
    d["age"] = pd.to_numeric(d["age"], errors="coerce").where(lambda x: x.between(18,110))
    age_center = float(d["age"].mean())
    d["age_10y_centered"] = (d["age"] - age_center) / 10.0
    return d, age_center


def fit_glm(df, outcome, covs, model_name, window, subgroup="All"):
    dat = df[[outcome] + covs].dropna().copy()
    if len(dat) == 0 or dat[outcome].nunique() < 2:
        return pd.DataFrame(), None
    y = dat[outcome].astype(float)
    X = sm.add_constant(dat[covs].astype(float), has_constant="add")
    fit = sm.GLM(y, X, family=sm.families.Binomial()).fit(cov_type="HC3")
    rows = []
    for term in covs:
        b, se = float(fit.params[term]), float(fit.bse[term])
        rows.append({
            "window_days": window, "outcome": outcome, "model": model_name,
            "subgroup": subgroup, "N": len(dat), "events": int(y.sum()), "term": term,
            "beta": b, "robust_SE_HC3": se, "OR": math.exp(b),
            "OR_CI95_low": math.exp(b-1.96*se), "OR_CI95_high": math.exp(b+1.96*se),
            "P_value": float(fit.pvalues[term]), "log_likelihood": float(fit.llf), "AIC": float(fit.aic),
        })
    return pd.DataFrame(rows), fit


def run_models(df, window):
    outcomes = ["mortality_30d", "aki", "postop_icu"]
    allr, comps = [], []
    m1 = ["ecg_sex_discordance_z", "age_10y_centered", "sex_binary"]
    m2 = m1 + ["asa"]

    for outcome in outcomes:
        for name, covs in [("Model1_age_sex", m1), ("Model2_age_sex_ASA", m2)]:
            r, _ = fit_glm(df, outcome, covs, name, window)
            allr.append(r)

        di = df.copy()
        di["discordance_x_sex"] = di["ecg_sex_discordance_z"] * di["sex_binary"]
        r, _ = fit_glm(di, outcome, m2+["discordance_x_sex"], "Model2_plus_sex_interaction", window)
        allr.append(r)

        for sex, label in [(0,"Female"),(1,"Male")]:
            sub = df[df.sex_binary.eq(sex)].copy()
            r, _ = fit_glm(sub, outcome,
                           ["ecg_sex_discordance_z","age_10y_centered","asa"],
                           "Model2_age_ASA_sex_stratified", window, label)
            allr.append(r)

        c = df[[outcome,"age_10y_centered","sex_binary","asa","ecg_sex_discordance_z"]].dropna().copy()
        if len(c) and c[outcome].nunique() == 2:
            y = c[outcome].astype(float)
            X0 = sm.add_constant(c[["age_10y_centered","sex_binary","asa"]], has_constant="add")
            X1 = sm.add_constant(c[["age_10y_centered","sex_binary","asa","ecg_sex_discordance_z"]], has_constant="add")
            f0 = sm.GLM(y, X0, family=sm.families.Binomial()).fit()
            f1 = sm.GLM(y, X1, family=sm.families.Binomial()).fit()
            p0, p1 = np.asarray(f0.predict(X0)), np.asarray(f1.predict(X1))
            lr = max(0.0, 2*(f1.llf-f0.llf))
            comps.append({
                "window_days": window, "outcome": outcome, "N": len(c), "events": int(y.sum()),
                "base_AUROC": roc_auc_score(y,p0), "extended_AUROC": roc_auc_score(y,p1),
                "delta_AUROC": roc_auc_score(y,p1)-roc_auc_score(y,p0),
                "base_AUPRC": average_precision_score(y,p0), "extended_AUPRC": average_precision_score(y,p1),
                "delta_AUPRC": average_precision_score(y,p1)-average_precision_score(y,p0),
                "base_Brier": brier_score_loss(y,p0), "extended_Brier": brier_score_loss(y,p1),
                "delta_Brier_extended_minus_base": brier_score_loss(y,p1)-brier_score_loss(y,p0),
                "LR_chi2_1df": lr, "LR_P_value": chi2.sf(lr,1),
            })

    reg = pd.concat([x for x in allr if len(x)], ignore_index=True)
    return reg, pd.DataFrame(comps)


def outcome_counts(df, window):
    rows=[]
    for o in ["mortality_30d","aki","postop_icu"]:
        x=df[o]; n=int(x.notna().sum()); e=int(x.eq(1).sum())
        rows.append({"window_days":window,"outcome":o,"N_nonmissing":n,"events":e,
                     "event_rate_pct":100*e/n if n else np.nan})
    return pd.DataFrame(rows)


def quartile_summary(df, window):
    d=df.copy(); d["discordance_quartile"]=pd.qcut(
        d["ecg_sex_discordance_z"],4,labels=["Q1 lowest","Q2","Q3","Q4 highest"],duplicates="drop")
    rows=[]
    for q,g in d.groupby("discordance_quartile",observed=True):
        for o in ["mortality_30d","aki","postop_icu"]:
            x=g[o]; n=int(x.notna().sum()); e=int(x.eq(1).sum())
            rows.append({"window_days":window,"discordance_quartile":str(q),"outcome":o,
                         "N_nonmissing":n,"events":e,"event_rate_pct":100*e/n if n else np.nan})
    return pd.DataFrame(rows)


def forest(reg, outdir):
    d=reg[(reg.window_days.eq(30))&(reg.model.eq("Model2_age_sex_ASA"))&
          (reg.subgroup.eq("All"))&(reg.term.eq("ecg_sex_discordance_z"))].copy()
    order=["mortality_30d","aki","postop_icu"]
    label={"mortality_30d":"30-day mortality","aki":"Postoperative AKI","postop_icu":"Postoperative ICU admission"}
    d["ord"]=d.outcome.map({x:i for i,x in enumerate(order)}); d=d.sort_values("ord")
    if d.empty: return
    x=d.OR.to_numpy(float); lo=d.OR_CI95_low.to_numpy(float); hi=d.OR_CI95_high.to_numpy(float); y=np.arange(len(d))
    fig,ax=plt.subplots(figsize=(7.4,3.9))
    ax.errorbar(x,y,xerr=np.vstack([x-lo,hi-x]),fmt="o",capsize=4)
    ax.axvline(1,linestyle="--",linewidth=1); ax.set_yticks(y); ax.set_yticklabels([label[o] for o in d.outcome]); ax.invert_yaxis()
    ax.set_xlabel("Adjusted OR per 1-SD greater ECG-sex discordance")
    ax.set_title("Closest preoperative ECG — frozen 5-seed ensemble"); ax.grid(axis="x",alpha=.25); fig.tight_layout()
    fig.savefig(outdir/"Postop_AllPreopECG_Frozen5Seed_ForestPlot.png",dpi=300,bbox_inches="tight")
    fig.savefig(outdir/"Postop_AllPreopECG_Frozen5Seed_ForestPlot.pdf",bbox_inches="tight")
    plt.show(); plt.close(fig)


# ==========================================================================================
# MAIN
# ==========================================================================================

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--outdir",type=Path,default=OUT_DEFAULT)
    ap.add_argument("--batch-size",type=int,default=128)
    ap.add_argument("--num-workers",type=int,default=8)
    args=ap.parse_args(); args.outdir.mkdir(parents=True,exist_ok=True)

    needed=[FINAL_MODEL_SOURCE,LOCKED_SPLIT,LOCKED_PRED,MASTER_PARQUET,RAW_ECG_DIR,DB_PATH]+[ckpt_path(s) for s in SEEDS]
    for p in needed:
        if not p.exists(): raise FileNotFoundError(p)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable.")

    print("="*120)
    print("WCMC ECG-SEX — ALL PREOPERATIVE ECGs / FROZEN FINAL 5-SEED ENSEMBLE")
    print("="*120)
    print("GPU:",torch.cuda.get_device_name(0)); print("Model source:",FINAL_MODEL_SOURCE); print("OUT:",args.outdir)

    mod=import_final_module()
    locked=load_locked_test(); ecg=load_all_ecgs(locked); surg=load_surgeries(locked)
    pair=build_candidate_pairs(locked,ecg,surg)
    sel30=select_window(pair,30); sel90=select_window(pair,90)
    sel30.to_csv(args.outdir/"01_selected_preopECG_cohort_30d_before_inference.csv",index=False)
    sel90.to_csv(args.outdir/"01_selected_preopECG_cohort_90d_before_inference.csv",index=False)

    union=pd.concat([
        sel30[["patient_key","ecg_key","sex_binary","raw_local_path"]],
        sel90[["patient_key","ecg_key","sex_binary","raw_local_path"]],
    ],ignore_index=True).drop_duplicates(["patient_key","ecg_key"]).reset_index(drop=True)
    print(f"Unique selected ECGs to infer: {len(union):,}")

    pred=infer_ensemble(mod,union,args.outdir,args.batch_size,args.num_workers)
    qc_against_locked(pred,args.outdir)
    pred,ref=add_discordance(pred)
    with open(args.outdir/"04_discordance_reference_locked_test.json","w") as f: json.dump(ref,f,indent=2)
    pred.to_csv(args.outdir/"04_selected_preopECG_predictions_with_discordance.csv",index=False)

    audits=[]; counts=[]; quarts=[]; regs=[]; comps=[]
    for window,sel in [(30,sel30),(90,sel90)]:
        d,age_center=prepare(sel,pred)
        d.to_csv(args.outdir/f"05_final_analysis_cohort_{window}d.csv",index=False)
        counts.append(outcome_counts(d,window)); quarts.append(quartile_summary(d,window))
        r,c=run_models(d,window); regs.append(r); comps.append(c)
        audits.append({
            "window_days":window,"locked_test_patient_universe_N":len(locked),
            "locked_test_patients_with_surgery_N":surg.patient_key.nunique(),
            "selected_preop_patients_before_inference_N":len(sel),"final_5seed_prediction_N":len(d),
            "female_N":int(d.sex_binary.eq(0).sum()),"male_N":int(d.sex_binary.eq(1).sum()),
            "age_available_N":int(d.age.notna().sum()),"ASA_available_N":int(d.asa.notna().sum()),
            "mean_age":float(d.age.mean()),"age_center":age_center,
            "median_days_ECG_to_surgery":float(d.days_ecg_to_surgery.median()),
            "Q1_days_ECG_to_surgery":float(d.days_ecg_to_surgery.quantile(.25)),
            "Q3_days_ECG_to_surgery":float(d.days_ecg_to_surgery.quantile(.75)),
            "selected_is_original_locked_ECG_N":int(d.ecg_key.eq(d.locked_ecg_key).sum()),
            "selected_is_original_locked_ECG_pct":float(100*d.ecg_key.eq(d.locked_ecg_key).mean()),
        })

    audit=pd.DataFrame(audits); count=pd.concat(counts,ignore_index=True); quart=pd.concat(quarts,ignore_index=True)
    reg=pd.concat(regs,ignore_index=True); comp=pd.concat(comps,ignore_index=True)
    audit.to_csv(args.outdir/"06_cohort_audit.csv",index=False); count.to_csv(args.outdir/"07_outcome_event_counts.csv",index=False)
    quart.to_csv(args.outdir/"08_discordance_quartile_outcomes.csv",index=False); reg.to_csv(args.outdir/"09_adjusted_logistic_regression.csv",index=False)
    comp.to_csv(args.outdir/"10_clinical_model_incremental_value.csv",index=False)

    primary=reg[(reg.window_days.eq(30))&(reg.model.eq("Model2_age_sex_ASA"))&(reg.subgroup.eq("All"))&(reg.term.eq("ecg_sex_discordance_z"))].copy()
    primary.to_csv(args.outdir/"11_PRIMARY_30d_ECGWindow_AdjustedAssociations.csv",index=False)
    mortality=reg[(reg.outcome.eq("mortality_30d"))&reg.term.isin(["ecg_sex_discordance_z","discordance_x_sex"])].copy()
    mortality.to_csv(args.outdir/"12_mortality_model_and_window_sensitivity.csv",index=False)
    forest(reg,args.outdir)

    print("\n"+"="*120); print("FINAL COHORT AUDIT"); print("="*120); print(audit.to_string(index=False))
    print("\n"+"="*120); print("OUTCOME COUNTS"); print("="*120); print(count.to_string(index=False))
    print("\n"+"="*120); print("PRIMARY 30-DAY WINDOW — AGE/SEX/ASA ADJUSTED"); print("="*120)
    print(primary[["outcome","N","events","OR","OR_CI95_low","OR_CI95_high","P_value"]].to_string(index=False) if len(primary) else "No rows")
    print("\n"+"="*120); print("MORTALITY MODEL/WINDOW/SEX-INTERACTION SENSITIVITY"); print("="*120)
    print(mortality[["window_days","model","subgroup","term","N","events","OR","OR_CI95_low","OR_CI95_high","P_value"]].to_string(index=False))
    print("\n"+"="*120); print("INCREMENTAL VALUE"); print("="*120); print(comp.to_string(index=False))
    print("\nSaved:"); [print(" ",p) for p in sorted(args.outdir.iterdir())]


if __name__ == "__main__":
    main()