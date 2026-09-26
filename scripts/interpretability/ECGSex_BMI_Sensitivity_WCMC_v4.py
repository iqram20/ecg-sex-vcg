#!/usr/bin/env python3
"""
WCMC/NYP ECG-sex BMI sensitivity analysis (v2)

Uses the FINAL locked-test 5-seed VCG ensemble predictions.
No model retraining or recalibration.

Analyses:
  1) BMI-stratified performance: <25, 25-29.9, >=30 kg/m^2
  2) 1:1 female/male age+BMI matched sensitivity cohort
  3) Sex x BMI interaction on frozen model score (HC3 robust OLS)

Robust alignment:
  - Uses ecg_key if available in predictions.
  - If the final aligned prediction CSV has no ecg_key, verifies:
      a) prediction row count == locked-test row count
      b) sex_true_binary exactly matches locked-test sex vector
    before attaching locked-test identifiers positionally.
"""

from __future__ import annotations

import argparse
import bisect
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score
)

MASTER_PARQUET = Path(
    "/athena/madelab/scratch/iqh4001/Azure_Cradle_ECG/Data/"
    "ECG-Echo_MetaData/ecg_echo_master_cohort_v1.parquet"
)

LOCKED_TEST_CSV = Path(
    "/athena/madelab/scratch/iqh4001/ECG/ECG_Echo/Runs/"
    "Master_ECG_Sex_MIMIC_EchoNext/"
    "20260609_020154_MASTER_ECG_SEX_MIMIC_ECHONEXT/"
    "wcmc_ecgsex_from_scratch/"
    "04_wcmc_finetune_from_mimic/splits/WCMC_ECGSex_test.csv"
)

FINAL_PREDICTIONS = Path(
    "/athena/madelab/scratch/iqh4001/ECG/ECG_Echo/Runs/"
    "Master_ECG_Sex_MIMIC_EchoNext/20260609_020154_MASTER_ECG_SEX_MIMIC_ECHONEXT/"
    "wcmc_ecgsex_from_scratch/19_vcg_geometry_resnet_5seed_locked_wcmc/"
    "final_5seed_ensemble_bootstrap/"
    "WCMC_LockedTest_Aligned_5SeedEnsemble_Predictions.csv"
)

THRESHOLD = 0.484619
N_BOOT = 2000
SEED = 20260925

KEY_CANDS = ["ecg_key", "ECG_KEY", "ecg_id", "ECG_ID"]
SEX_CANDS = ["sex_true_binary", "ecg_sex", "sex", "Sex", "gender", "Gender", "label", "target"]
AGE_CANDS = ["age_at_ecg", "ecg_age_at_ecg", "age", "Age"]
BMI_CANDS = ["bmi", "BMI", "body_mass_index", "BodyMassIndex", "bmi_kg_m2", "echo_bmi_value"]
PROB_CANDS = ["prob_vcg", "prob_male", "p_male", "ensemble_prob", "y_prob", "pred_prob", "probability"]

def first(cols, candidates):
    s = set(cols)
    for c in candidates:
        if c in s:
            return c
    return None

def norm_sex(s):
    if pd.api.types.is_numeric_dtype(s):
        x = pd.to_numeric(s, errors="coerce")
        if set(x.dropna().unique()).issubset({0, 1, 0.0, 1.0}):
            return x.astype("Float64")
    z = s.astype(str).str.strip().str.lower()
    return z.map({
        "female":0, "f":0, "0":0, "0.0":0,
        "male":1, "m":1, "1":1, "1.0":1
    }).astype("Float64")

def load_master(path):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    cols = pf.schema.names

    key = first(cols, KEY_CANDS)
    bmi = first(cols, BMI_CANDS)
    age = first(cols, AGE_CANDS)
    sex = first(cols, SEX_CANDS)

    height_cm = first(cols, ["height_cm", "Height_cm", "height_centimeters", "echo_height_value"])
    height_m  = first(cols, ["height_m", "Height_m", "height_meters"])
    weight_kg = first(cols, ["weight_kg", "Weight_kg", "weight_kilograms", "echo_weight_value"])
    weight_lb = first(cols, ["weight_lb", "weight_lbs", "Weight_lb", "weight_pounds"])
    height_units = "echo_height_units" if "echo_height_units" in cols else None
    weight_units = "echo_weight_units" if "echo_weight_units" in cols else None

    if key is None:
        raise RuntimeError("No ecg_key-like identifier in master parquet.")

    needed = [x for x in [key,bmi,age,sex,height_cm,height_m,weight_kg,weight_lb,height_units,weight_units] if x]
    needed = list(dict.fromkeys(needed))
    d = pd.read_parquet(path, columns=needed)

    info = dict(
        key=key, bmi=bmi, age=age, sex=sex,
        height_cm=height_cm, height_m=height_m,
        weight_kg=weight_kg, weight_lb=weight_lb,
        height_units=height_units, weight_units=weight_units,
        all_columns=cols
    )
    return d, info

def derive_bmi(d, info):
    # Preferred path: use the explicit echo-derived BMI already stored in the master cohort.
    if info["bmi"] is not None:
        bmi = pd.to_numeric(d[info["bmi"]], errors="coerce")
        print(f"[BMI] using direct column: {info['bmi']}")

        # Physiologic plausibility is enforced later (10-80 kg/m^2).
        return bmi

    # Fallback only if explicit BMI is absent.
    h = None
    w = None

    # WCMC echo metadata may store height/weight in generic value columns,
    # so inspect associated units if present.
    if info["height_m"]:
        h = pd.to_numeric(d[info["height_m"]], errors="coerce")
    elif info["height_cm"]:
        h_raw = pd.to_numeric(d[info["height_cm"]], errors="coerce")

        if info["height_cm"] == "echo_height_value" and "echo_height_units" in d.columns:
            u = d["echo_height_units"].astype(str).str.strip().str.lower()
            h = pd.Series(np.nan, index=d.index, dtype=float)
            cm_mask = u.str.contains("cm|centimeter", regex=True, na=False)
            m_mask = u.str.fullmatch(r"m|meter|meters", na=False)
            inch_mask = u.str.contains("in|inch", regex=True, na=False)
            h.loc[cm_mask] = h_raw.loc[cm_mask] / 100.0
            h.loc[m_mask] = h_raw.loc[m_mask]
            h.loc[inch_mask] = h_raw.loc[inch_mask] * 0.0254
        else:
            h = h_raw / 100.0

    if info["weight_kg"]:
        w_raw = pd.to_numeric(d[info["weight_kg"]], errors="coerce")

        if info["weight_kg"] == "echo_weight_value" and "echo_weight_units" in d.columns:
            u = d["echo_weight_units"].astype(str).str.strip().str.lower()
            w = pd.Series(np.nan, index=d.index, dtype=float)
            kg_mask = u.str.contains("kg|kilogram", regex=True, na=False)
            lb_mask = u.str.contains("lb|pound", regex=True, na=False)
            w.loc[kg_mask] = w_raw.loc[kg_mask]
            w.loc[lb_mask] = w_raw.loc[lb_mask] * 0.45359237
        else:
            w = w_raw

    elif info["weight_lb"]:
        w = pd.to_numeric(d[info["weight_lb"]], errors="coerce") * 0.45359237

    if h is not None and w is not None:
        print("[BMI] explicit BMI absent; computing from height and weight")
        return w / (h * h)

    bmi_like = [
        c for c in info["all_columns"]
        if any(k in c.lower() for k in ("bmi", "height", "weight"))
    ]
    raise RuntimeError(
        "BMI cannot be found or derived in ecg_echo_master_cohort_v1.parquet.\n"
        "BMI/height/weight-like columns:\n  " + "\n  ".join(bmi_like)
    )

def metrics(y, p, threshold):
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    yh = (p >= threshold).astype(int)
    return {
        "N": len(y),
        "Female_N": int((y==0).sum()),
        "Male_N": int((y==1).sum()),
        "AUROC": roc_auc_score(y,p),
        "AUPRC": average_precision_score(y,p),
        "Accuracy": accuracy_score(y,yh),
        "Balanced_Accuracy": balanced_accuracy_score(y,yh),
        "Precision": precision_score(y,yh,zero_division=0),
        "Sensitivity": recall_score(y,yh,zero_division=0),
        "Specificity": recall_score(y,yh,pos_label=0,zero_division=0),
        "F1": f1_score(y,yh,zero_division=0),
        "Brier": brier_score_loss(y,p),
    }

BOOT_KEYS = ["AUROC","AUPRC","Accuracy","Balanced_Accuracy","Precision",
             "Sensitivity","Specificity","F1","Brier"]

def metric_row(d, prob_col, label, n_boot, seed):
    y = d["sex_binary"].astype(int).to_numpy()
    p = d[prob_col].astype(float).to_numpy()
    if np.unique(y).size != 2:
        raise RuntimeError(f"{label}: only one sex class.")

    m = metrics(y,p,THRESHOLD)
    rng = np.random.default_rng(seed)
    store = {k:[] for k in BOOT_KEYS}
    n = len(d)

    for b in range(n_boot):
        idx = rng.integers(0,n,n)
        yy, pp = y[idx], p[idx]
        if np.unique(yy).size != 2:
            continue
        mm = metrics(yy,pp,THRESHOLD)
        for k in BOOT_KEYS:
            store[k].append(mm[k])
        if (b+1) % 250 == 0:
            print(f"    bootstrap {b+1}/{n_boot}")

    row = {"Analysis":label, **m}
    for k in BOOT_KEYS:
        a = np.asarray(store[k])
        row[f"{k}_CI_low"] = np.quantile(a,0.025)
        row[f"{k}_CI_high"] = np.quantile(a,0.975)

    print(f"{label}: N={m['N']:,}, AUROC={m['AUROC']:.6f}, "
          f"AUPRC={m['AUPRC']:.6f}, F1={m['F1']:.6f}")
    return row

def match_age_bmi(d, caliper):
    x = d.copy()
    x["age_int"] = np.floor(x["age"] + 0.5).astype(int)

    f = x[x.sex_binary==0]
    m = x[x.sex_binary==1]
    pairs = []
    pid = 0

    for age in sorted(set(f.age_int) & set(m.age_int)):
        fa = f[f.age_int==age].sort_values(["BMI","ecg_key"])
        ma = m[m.age_int==age].sort_values(["BMI","ecg_key"])

        if len(fa) <= len(ma):
            anchor, pool, anchor_f = fa, ma, True
        else:
            anchor, pool, anchor_f = ma, fa, False

        pool_idx = list(pool.index)
        pool_bmi = [float(pool.loc[i,"BMI"]) for i in pool_idx]

        for ai, ar in anchor.iterrows():
            if not pool_idx:
                break
            b = float(ar.BMI)
            pos = bisect.bisect_left(pool_bmi,b)
            cand = []
            if pos < len(pool_bmi): cand.append(pos)
            if pos > 0: cand.append(pos-1)
            if not cand: continue
            j = min(cand,key=lambda z: abs(pool_bmi[z]-b))
            if abs(pool_bmi[j]-b) > caliper:
                continue
            pi = pool_idx[j]
            pid += 1
            fi, mi = (ai,pi) if anchor_f else (pi,ai)
            pairs.append((pid,fi,mi))
            del pool_idx[j]
            del pool_bmi[j]

    rows = []
    for pid,fi,mi in pairs:
        rf = x.loc[fi].copy(); rf["match_pair_id"]=pid
        rm = x.loc[mi].copy(); rm["match_pair_id"]=pid
        rows += [rf,rm]

    if not rows:
        raise RuntimeError("No age+BMI matched pairs.")
    return pd.DataFrame(rows).reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default=str(FINAL_PREDICTIONS))
    ap.add_argument("--master", default=str(MASTER_PARQUET))
    ap.add_argument("--split", default=str(LOCKED_TEST_CSV))
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--bmi-caliper", type=float, default=1.0)
    ap.add_argument("--outdir", default=(
        "/athena/madelab/scratch/iqh4001/ECG/ECG-gender/Results/"
        "BMI_sensitivity_WCMC_locked_test"
    ))
    args = ap.parse_args()

    pred_path = Path(args.predictions)
    master_path = Path(args.master)
    split_path = Path(args.split)
    out = Path(args.outdir)
    out.mkdir(parents=True,exist_ok=True)

    print("="*110)
    print("FINAL WCMC/NYP ECG-SEX BMI SENSITIVITY ANALYSIS")
    print("="*110)
    print("Predictions:",pred_path)
    print("Master     :",master_path)
    print("Split      :",split_path)
    print("Threshold  :",THRESHOLD)
    print("="*110)

    if not pred_path.exists():
        raise FileNotFoundError(pred_path)

    pred = pd.read_csv(pred_path)
    split = pd.read_csv(split_path)
    master, info = load_master(master_path)

    prob_col = first(pred.columns, PROB_CANDS)
    if prob_col is None:
        raise RuntimeError(f"No VCG probability column. Prediction columns: {list(pred.columns)}")
    print("[prediction probability]",prob_col)

    split_key = first(split.columns, KEY_CANDS)
    if split_key is None:
        raise RuntimeError(f"No ecg_key in locked split. Columns: {list(split.columns)}")
    if split_key != "ecg_key":
        split = split.rename(columns={split_key:"ecg_key"})

    split_sex = first(split.columns, SEX_CANDS)
    split_age = first(split.columns, AGE_CANDS)
    if split_sex is None or split_age is None:
        raise RuntimeError("Locked split must contain sex and age.")

    split = split.copy()
    split["sex_locked"] = norm_sex(split[split_sex])
    split["age_locked"] = pd.to_numeric(split[split_age],errors="coerce")

    pred_key = first(pred.columns, KEY_CANDS)

    # Robust prediction alignment.
    if pred_key is not None:
        print(f"[alignment] key-based using {pred_key}")
        if pred_key != "ecg_key":
            pred = pred.rename(columns={pred_key:"ecg_key"})
        d = split[["ecg_key","sex_locked","age_locked"]].merge(
            pred, on="ecg_key", how="inner", validate="one_to_one"
        )
    else:
        print("[alignment] prediction CSV has no ECG key; validating aligned row order")
        if len(pred) != len(split):
            raise RuntimeError(
                f"Cannot positional-align: predictions N={len(pred)} vs split N={len(split)}"
            )
        pred_sex_col = first(pred.columns, ["sex_true_binary","y_true","label","target"])
        if pred_sex_col is None:
            raise RuntimeError("No prediction sex_true_binary available for alignment verification.")
        psex = norm_sex(pred[pred_sex_col])
        if psex.isna().any() or split["sex_locked"].isna().any():
            raise RuntimeError("Missing sex labels prevent positional alignment verification.")
        if not np.array_equal(psex.astype(int).to_numpy(),
                              split["sex_locked"].astype(int).to_numpy()):
            raise RuntimeError(
                "Prediction sex_true_binary does NOT exactly match locked-test sex vector; "
                "refusing positional alignment."
            )
        print("[alignment] PASS: N matches and sex_true_binary exactly matches locked split.")
        d = pd.concat(
            [
                split[["ecg_key","sex_locked","age_locked"]].reset_index(drop=True),
                pred.reset_index(drop=True)
            ],
            axis=1
        )

    print(f"[alignment] locked rows retained: {len(d):,}")

    if d["ecg_key"].duplicated().any():
        raise RuntimeError("Duplicate ecg_key after prediction alignment.")

    # BMI merge.
    if info["key"] != "ecg_key":
        master = master.rename(columns={info["key"]:"ecg_key"})
    master["BMI"] = derive_bmi(master,info)
    master = master[["ecg_key","BMI"]].drop_duplicates("ecg_key")

    d = d.merge(master,on="ecg_key",how="left",validate="one_to_one")
    d["sex_binary"] = d["sex_locked"].astype("Float64")
    d["age"] = d["age_locked"]

    d[prob_col] = pd.to_numeric(d[prob_col],errors="coerce")
    d["BMI"] = pd.to_numeric(d["BMI"],errors="coerce")

    n0 = len(d)
    d = d.dropna(subset=["sex_binary","age","BMI",prob_col]).copy()
    d = d[d.BMI.between(10,80)].copy()
    d["sex_binary"] = d["sex_binary"].astype(int)

    print(f"[BMI complete/QC] {len(d):,}/{n0:,}")
    print(f"Female={sum(d.sex_binary==0):,}, Male={sum(d.sex_binary==1):,}")
    print(f"BMI={d.BMI.mean():.2f} ± {d.BMI.std():.2f}")

    d["BMI_group"] = pd.cut(
        d.BMI,[-np.inf,25,30,np.inf],right=False,
        labels=["BMI <25","BMI 25-29.9","BMI >=30"]
    )

    results = []
    results.append(metric_row(d,prob_col,"BMI-complete locked test",args.n_boot,SEED))

    for j,g in enumerate(["BMI <25","BMI 25-29.9","BMI >=30"]):
        s = d[d.BMI_group.astype(str)==g].copy()
        print(f"\n[{g}] N={len(s):,}")
        if len(s)>=50 and s.sex_binary.nunique()==2:
            results.append(metric_row(s,prob_col,g,args.n_boot,SEED+100+j))

    matched = match_age_bmi(d,args.bmi_caliper)
    print(f"\n[matching] pairs={matched.match_pair_id.nunique():,}, N={len(matched):,}")
    results.append(metric_row(
        matched,prob_col,f"Age+BMI matched (N={len(matched):,})",
        args.n_boot,SEED+500
    ))

    # Sex x BMI interaction on frozen logit score.
    q = d.copy()
    p = np.clip(q[prob_col].astype(float),1e-6,1-1e-6)
    q["logit_score"] = np.log(p/(1-p))
    q["BMI_c"] = q.BMI-q.BMI.mean()
    q["Age_c"] = q.age-q.age.mean()
    q["Sex_x_BMI"] = q.sex_binary*q.BMI_c
    X = sm.add_constant(q[["sex_binary","BMI_c","Sex_x_BMI","Age_c"]].astype(float))
    fit = sm.OLS(q.logit_score.astype(float),X).fit(cov_type="HC3")
    ci = fit.conf_int()
    inter = pd.DataFrame({
        "Term":fit.params.index,
        "Beta":fit.params.values,
        "Robust_SE_HC3":fit.bse.values,
        "CI95_low":ci.iloc[:,0].values,
        "CI95_high":ci.iloc[:,1].values,
        "P_value":fit.pvalues.values,
        "N":int(fit.nobs),
        "R2":fit.rsquared
    })

    res = pd.DataFrame(results)
    res.to_csv(out/"BMI_sensitivity_metrics.csv",index=False)
    matched.to_csv(out/"BMI_age_matched_cohort.csv",index=False)
    inter.to_csv(out/"BMI_sex_interaction_HC3.csv",index=False)

    print("\n"+"="*110)
    print("FINAL")
    print("="*110)
    print(res[[
        "Analysis","N","Female_N","Male_N",
        "AUROC","AUROC_CI_low","AUROC_CI_high",
        "AUPRC","F1","Brier"
    ]].to_string(index=False))
    print("\nSex x BMI interaction")
    print(inter.to_string(index=False))
    print("\nOutputs:",out)

if __name__ == "__main__":
    main()