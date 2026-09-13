#!/usr/bin/env python
# ==========================================================================================
# ECG-SEX — MANUSCRIPT TABLE 3 — COMPARATIVE PERFORMANCE
# Final WCMC/NYP locked test + external PTB-XL + Chapman/Ningbo
# ==========================================================================================
#
# Outputs
# -------
# 1) Table3_ECGSex_Comparative_Performance_Long.csv
# 2) Table3_ECGSex_Comparative_Performance_Manuscript.csv
# 3) Table3_ECGSex_Comparative_Performance_Footnote.txt
#
# Metrics are the final audited values used in the current manuscript figures.
# Accuracy is completed from verified patient-level predictions and frozen thresholds.
# ==========================================================================================

from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    brier_score_loss,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)

RUN = Path(
    "/athena/madelab/scratch/iqh4001/ECG/ECG_Echo/Runs/"
    "Master_ECG_Sex_MIMIC_EchoNext/"
    "20260609_020154_MASTER_ECG_SEX_MIMIC_ECHONEXT"
)

OUT_DIR = RUN / "publication_tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LONG_OUT = OUT_DIR / "Table3_ECGSex_Comparative_Performance_Long.csv"
MANUSCRIPT_OUT = OUT_DIR / "Table3_ECGSex_Comparative_Performance_Manuscript.csv"
FOOTNOTE_OUT = OUT_DIR / "Table3_ECGSex_Comparative_Performance_Footnote.txt"

EXT_BOOT_DIR = (
    RUN / "wcmc_ecgsex_from_scratch"
    / "20_vcg_geometry_resnet_external_confirmation"
    / "final_paired_bootstrap"
)
PTB_BOOT = EXT_BOOT_DIR / "ptbxl_VCG_vs_ResNet_bootstrap_2000.csv"
CHAPMAN_BOOT = EXT_BOOT_DIR / "chapman_ningbo_VCG_vs_ResNet_bootstrap_2000.csv"


WCMC_ROOT = RUN / "wcmc_ecgsex_from_scratch"

WCMC_FINAL_PRED = (
    WCMC_ROOT
    / "19_vcg_geometry_resnet_5seed_locked_wcmc"
    / "final_5seed_ensemble_bootstrap"
    / "WCMC_LockedTest_Aligned_5SeedEnsemble_Predictions.csv"
)

WCMC_TEST_SPLIT = (
    WCMC_ROOT
    / "04_wcmc_finetune_from_mimic"
    / "splits"
    / "WCMC_ECGSex_test.csv"
)

EXT_ROOT = (
    WCMC_ROOT
    / "20_vcg_geometry_resnet_external_confirmation"
)

EXT_PERF = (
    EXT_BOOT_DIR
    / "VCGGeometryResNet_ExternalConfirmation_Combined_Performance.csv"
)

ACCURACY_AUDIT_OUT = (
    OUT_DIR
    / "Table3_ECGSex_Accuracy_SourceAudit.csv"
)

# Frozen WCMC validation thresholds for the final matched 5-seed ensembles.
VCG_THRESHOLD = 0.484619
RESNET_THRESHOLD = 0.453711

# Cohort sizes used as strict prediction-source checks.
EXPECTED_N = {
    "WCMC/NYP": 21374,
    "PTB-XL": 18752,
    "Chapman/Ningbo": 42997,
}


# ==========================================================================================
# FINAL AUDITED PERFORMANCE VALUES
# ==========================================================================================

rows = [

    # --------------------------------------------------------------------------------------
    # WCMC/NYP LOCKED TEST
    # --------------------------------------------------------------------------------------
    {
        "Dataset": "WCMC/NYP",
        "Evaluation": "Locked test",
        "N": 21374,
        "Model": "MIMIC-only frozen",
        "Training strategy": "MIMIC-IV pretraining; zero-shot WCMC/NYP",
        "AUROC": 0.904395,
        "AUROC_L": 0.901,
        "AUROC_U": 0.908,
        "AUPRC": 0.896241,
        "AUPRC_L": 0.891,
        "AUPRC_U": 0.902,
        "Accuracy": 0.821,
        "F1": 0.821,
        "Brier": 0.135,
    },
    {
        "Dataset": "WCMC/NYP",
        "Evaluation": "Locked test",
        "N": 21374,
        "Model": "CNN-TCN-Transformer",
        "Training strategy": "WCMC/NYP-only",
        "AUROC": 0.929352,
        "AUROC_L": 0.926,
        "AUROC_U": 0.933,
        "AUPRC": 0.922941,
        "AUPRC_L": 0.919,
        "AUPRC_U": 0.927,
        "Accuracy": 0.844,
        "F1": 0.846638,
        "Brier": 0.110256,
    },
    {
        "Dataset": "WCMC/NYP",
        "Evaluation": "Locked test",
        "N": 21374,
        "Model": "CNN-TCN-Transformer",
        "Training strategy": "MIMIC-IV → WCMC/NYP",
        "AUROC": 0.944491,
        "AUROC_L": 0.942,
        "AUROC_U": 0.947,
        "AUPRC": 0.939860,
        "AUPRC_L": 0.936,
        "AUPRC_U": 0.943,
        "Accuracy": 0.868,
        "F1": 0.868093,
        "Brier": 0.094134,
    },
    {
        "Dataset": "WCMC/NYP",
        "Evaluation": "Locked test",
        "N": 21374,
        "Model": "ResNet1D",
        "Training strategy": "WCMC/NYP-only",
        "AUROC": 0.938674,
        "AUROC_L": np.nan,
        "AUROC_U": np.nan,
        "AUPRC": 0.932386,
        "AUPRC_L": np.nan,
        "AUPRC_U": np.nan,
        "Accuracy": np.nan,
        "F1": 0.860929,
        "Brier": 0.099295,
    },
    {
        "Dataset": "WCMC/NYP",
        "Evaluation": "Locked test",
        "N": 21374,
        "Model": "CNN-LSTM",
        "Training strategy": "WCMC/NYP-only",
        "AUROC": 0.926467,
        "AUROC_L": np.nan,
        "AUROC_U": np.nan,
        "AUPRC": 0.920764,
        "AUPRC_L": np.nan,
        "AUPRC_U": np.nan,
        "Accuracy": np.nan,
        "F1": 0.843049,
        "Brier": 0.116783,
    },
    {
        "Dataset": "WCMC/NYP",
        "Evaluation": "Locked test",
        "N": 21374,
        "Model": "CNN",
        "Training strategy": "WCMC/NYP-only",
        "AUROC": 0.918361,
        "AUROC_L": np.nan,
        "AUROC_U": np.nan,
        "AUPRC": 0.912966,
        "AUPRC_L": np.nan,
        "AUPRC_U": np.nan,
        "Accuracy": np.nan,
        "F1": 0.835764,
        "Brier": 0.120049,
    },
    {
        "Dataset": "WCMC/NYP",
        "Evaluation": "Locked test",
        "N": 21374,
        "Model": "Transformer",
        "Training strategy": "WCMC/NYP-only",
        "AUROC": 0.910683,
        "AUROC_L": np.nan,
        "AUROC_U": np.nan,
        "AUPRC": 0.902892,
        "AUPRC_L": np.nan,
        "AUPRC_U": np.nan,
        "Accuracy": np.nan,
        "F1": 0.822756,
        "Brier": 0.122543,
    },
    {
        "Dataset": "WCMC/NYP",
        "Evaluation": "Locked test",
        "N": 21374,
        "Model": "ResNet1D",
        "Training strategy": "MIMIC-IV → WCMC/NYP",
        "AUROC": 0.950463,
        "AUROC_L": 0.948,
        "AUROC_U": 0.953,
        "AUPRC": 0.946670,
        "AUPRC_L": 0.943,
        "AUPRC_U": 0.950,
        "Accuracy": np.nan,
        "F1": 0.873242,
        "Brier": 0.088621,
    },
    {
        "Dataset": "WCMC/NYP",
        "Evaluation": "Locked test",
        "N": 21374,
        "Model": "VCG-Geometry ResNet",
        "Training strategy": "MIMIC-IV → WCMC/NYP",
        "AUROC": 0.954230,
        "AUROC_L": 0.951729,
        "AUROC_U": 0.956681,
        "AUPRC": 0.950889,
        "AUPRC_L": 0.947675,
        "AUPRC_U": 0.953874,
        "Accuracy": 0.880837,
        "F1": 0.878859,
        "Brier": 0.085006,
    },

    # --------------------------------------------------------------------------------------
    # PTB-XL ZERO-SHOT EXTERNAL
    # --------------------------------------------------------------------------------------
    {
        "Dataset": "PTB-XL",
        "Evaluation": "Zero-shot external",
        "N": 18752,
        "Model": "ResNet1D",
        "Training strategy": "MIMIC-IV → WCMC/NYP; frozen external evaluation",
        "AUROC": 0.918502,
        "AUROC_L": np.nan,
        "AUROC_U": np.nan,
        "AUPRC": 0.920998,
        "AUPRC_L": np.nan,
        "AUPRC_U": np.nan,
        "Accuracy": np.nan,
        "F1": 0.777063,
        "Brier": 0.201021,
    },
    {
        "Dataset": "PTB-XL",
        "Evaluation": "Zero-shot external",
        "N": 18752,
        "Model": "VCG-Geometry ResNet",
        "Training strategy": "MIMIC-IV → WCMC/NYP; frozen external evaluation",
        "AUROC": 0.926778,
        "AUROC_L": np.nan,
        "AUROC_U": np.nan,
        "AUPRC": 0.929112,
        "AUPRC_L": np.nan,
        "AUPRC_U": np.nan,
        "Accuracy": np.nan,
        "F1": 0.784935,
        "Brier": 0.195693,
    },

    # --------------------------------------------------------------------------------------
    # CHAPMAN/NINGBO ZERO-SHOT EXTERNAL
    # --------------------------------------------------------------------------------------
    {
        "Dataset": "Chapman/Ningbo",
        "Evaluation": "Zero-shot external",
        "N": 42997,
        "Model": "ResNet1D",
        "Training strategy": "MIMIC-IV → WCMC/NYP; frozen external evaluation",
        "AUROC": 0.910480,
        "AUROC_L": np.nan,
        "AUROC_U": np.nan,
        "AUPRC": 0.930031,
        "AUPRC_L": np.nan,
        "AUPRC_U": np.nan,
        "Accuracy": np.nan,
        "F1": 0.844181,
        "Brier": 0.131999,
    },
    {
        "Dataset": "Chapman/Ningbo",
        "Evaluation": "Zero-shot external",
        "N": 42997,
        "Model": "VCG-Geometry ResNet",
        "Training strategy": "MIMIC-IV → WCMC/NYP; frozen external evaluation",
        "AUROC": 0.916236,
        "AUROC_L": np.nan,
        "AUROC_U": np.nan,
        "AUPRC": 0.935149,
        "AUPRC_L": np.nan,
        "AUPRC_U": np.nan,
        "Accuracy": np.nan,
        "F1": 0.844364,
        "Brier": 0.134846,
    },
]

df = pd.DataFrame(rows)


# ==========================================================================================
# COMPLETE / AUDIT ACCURACY VALUES FROM REAL PATIENT-LEVEL PREDICTIONS
# ==========================================================================================
#
# Accuracy is threshold-dependent. This block therefore NEVER estimates accuracy
# from AUROC, AUPRC, F1, or Brier score.
#
# Final VCG/ResNet:
#   - Uses exact 5-seed patient-level probabilities.
#   - Uses frozen WCMC validation thresholds.
#
# External PTB-XL / Chapman-Ningbo:
#   - Uses the SAME frozen WCMC validation thresholds.
#   - No external threshold optimization is permitted.
#
# WCMC-only ResNet1D / CNN-LSTM / CNN / Transformer:
#   - Finds the exact locked-test prediction source by matching N + labels +
#     AUROC/AUPRC/Brier to the stored manuscript point estimates.
#   - Uses an existing binary-prediction/threshold field when available.
#   - Otherwise derives the threshold ONLY from a matched WCMC validation
#     prediction file by maximizing validation F1, then applies that frozen
#     threshold to the locked test.
#
# The script hard-stops if the source cannot be verified.
# ==========================================================================================

def require_file(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing required file:\n{path}")


def find_col(columns, candidates):
    lower = {str(c).lower(): c for c in columns}

    # Exact case-insensitive match first.
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]

    # Then conservative substring match.
    for c in columns:
        lc = str(c).lower()
        for cand in candidates:
            if cand.lower() in lc:
                return c

    return None


LABEL_CANDIDATES = [
    "sex_true_binary",
    "y_true",
    "true_label",
    "sex_binary",
    "label",
    "target",
]

PROB_CANDIDATES = [
    "prob_male",
    "y_prob",
    "pred_prob",
    "probability",
    "prob",
    "score",
]

BINARY_PRED_CANDIDATES = [
    "pred_binary",
    "y_pred",
    "pred_label",
    "predicted_label",
    "sex_pred_binary",
    "prediction",
]

THRESHOLD_CANDIDATES = [
    "threshold",
    "best_threshold",
    "decision_threshold",
    "operating_threshold",
    "f1_threshold",
]


def standardize_prediction(raw, prob_col=None):
    ycol = find_col(raw.columns, LABEL_CANDIDATES)

    if ycol is None:
        raise RuntimeError(
            f"Cannot identify binary ground-truth column.\n"
            f"Columns: {raw.columns.tolist()}"
        )

    if prob_col is None:
        prob_col = find_col(raw.columns, PROB_CANDIDATES)

    if prob_col is None:
        raise RuntimeError(
            f"Cannot identify probability column.\n"
            f"Columns: {raw.columns.tolist()}"
        )

    out = pd.DataFrame({
        "y": pd.to_numeric(raw[ycol], errors="raise").astype(int),
        "p": pd.to_numeric(raw[prob_col], errors="raise"),
    })

    if len(out) == 0:
        raise RuntimeError("Prediction dataframe is empty.")

    if out[["y", "p"]].isna().any().any():
        raise RuntimeError("Prediction dataframe contains missing labels/probabilities.")

    if set(out["y"].unique()) != {0, 1}:
        raise RuntimeError(
            f"Expected binary labels {{0,1}}; observed {sorted(out['y'].unique())}"
        )

    if ((out["p"] < 0) | (out["p"] > 1)).any():
        raise RuntimeError("Probability values fall outside [0,1].")

    return out.reset_index(drop=True)


def probability_metrics(y, p):
    return {
        "AUROC": float(roc_auc_score(y, p)),
        "AUPRC": float(average_precision_score(y, p)),
        "Brier": float(brier_score_loss(y, p)),
    }


def threshold_metrics(y, p, threshold):
    pred = (np.asarray(p, dtype=float) >= float(threshold)).astype(int)
    return {
        "Accuracy": float(accuracy_score(y, pred)),
        "F1": float(f1_score(y, pred, zero_division=0)),
    }


def max_f1_threshold(y, p):
    """
    Exact validation-only threshold selection using all PR-curve candidate thresholds.
    Returns threshold, validation F1.
    """
    precision, recall, thresholds = precision_recall_curve(y, p)

    if len(thresholds) == 0:
        raise RuntimeError("No candidate thresholds returned by precision_recall_curve.")

    precision = precision[:-1]
    recall = recall[:-1]

    denom = precision + recall
    f1 = np.divide(
        2.0 * precision * recall,
        denom,
        out=np.zeros_like(denom, dtype=float),
        where=denom > 0,
    )

    best_idx = int(np.nanargmax(f1))
    return float(thresholds[best_idx]), float(f1[best_idx])


def extract_binary_prediction(raw):
    col = find_col(raw.columns, BINARY_PRED_CANDIDATES)

    if col is None:
        return None, None

    vals = pd.to_numeric(raw[col], errors="coerce")

    if vals.isna().any():
        return None, None

    uniq = set(vals.astype(int).unique())

    if not uniq.issubset({0, 1}):
        return None, None

    return vals.astype(int).to_numpy(), col


def extract_scalar_threshold(raw):
    col = find_col(raw.columns, THRESHOLD_CANDIDATES)

    if col is None:
        return None, None

    vals = pd.to_numeric(raw[col], errors="coerce").dropna().unique()

    if len(vals) != 1:
        return None, None

    value = float(vals[0])

    if not (0 <= value <= 1):
        return None, None

    return value, col


def update_accuracy(dataset, model, training_strategy, accuracy):
    mask = (
        df["Dataset"].eq(dataset)
        & df["Model"].eq(model)
        & df["Training strategy"].eq(training_strategy)
    )

    if int(mask.sum()) != 1:
        raise RuntimeError(
            f"Expected exactly one Table 3 row for "
            f"{dataset} / {model} / {training_strategy}; found {int(mask.sum())}."
        )

    df.loc[mask, "Accuracy"] = float(accuracy)


def get_target_row(dataset, model, training_strategy):
    mask = (
        df["Dataset"].eq(dataset)
        & df["Model"].eq(model)
        & df["Training strategy"].eq(training_strategy)
    )

    if int(mask.sum()) != 1:
        raise RuntimeError(
            f"Expected exactly one target row for "
            f"{dataset} / {model} / {training_strategy}; found {int(mask.sum())}."
        )

    return df.loc[mask].iloc[0]


accuracy_audit = []


# ------------------------------------------------------------------------------------------
# A. EXACT FINAL WCMC/NYP 5-SEED VCG + MATCHED RESNET
# ------------------------------------------------------------------------------------------

require_file(WCMC_FINAL_PRED)
require_file(WCMC_TEST_SPLIT)

wcmc_test = pd.read_csv(WCMC_TEST_SPLIT, low_memory=False)

if "sex_binary" not in wcmc_test.columns:
    raise RuntimeError(
        f"Locked WCMC test split missing sex_binary:\n{WCMC_TEST_SPLIT}"
    )

reference_y = pd.to_numeric(
    wcmc_test["sex_binary"],
    errors="raise",
).astype(int).to_numpy()

if len(reference_y) != EXPECTED_N["WCMC/NYP"]:
    raise RuntimeError(
        f"WCMC locked test N={len(reference_y):,}; "
        f"expected {EXPECTED_N['WCMC/NYP']:,}."
    )

final_wcmc = pd.read_csv(WCMC_FINAL_PRED, low_memory=False)

required_final_cols = [
    "sex_true_binary",
    "prob_vcg",
    "prob_resnet",
]

missing = [c for c in required_final_cols if c not in final_wcmc.columns]

if missing:
    raise RuntimeError(
        f"Final WCMC prediction file missing columns {missing}:\n{WCMC_FINAL_PRED}"
    )

final_y = pd.to_numeric(
    final_wcmc["sex_true_binary"],
    errors="raise",
).astype(int).to_numpy()

if len(final_y) != EXPECTED_N["WCMC/NYP"]:
    raise RuntimeError(
        f"Final WCMC prediction N={len(final_y):,}; "
        f"expected {EXPECTED_N['WCMC/NYP']:,}."
    )

if not np.array_equal(final_y, reference_y):
    raise RuntimeError(
        "Final 5-seed WCMC labels do not match the locked-test reference ordering."
    )

for model, prob_col, threshold, manuscript_model in [
    (
        "ResNet1D",
        "prob_resnet",
        RESNET_THRESHOLD,
        "ResNet1D",
    ),
    (
        "VCG-Geometry ResNet",
        "prob_vcg",
        VCG_THRESHOLD,
        "VCG-Geometry ResNet",
    ),
]:
    p = pd.to_numeric(final_wcmc[prob_col], errors="raise").to_numpy(float)

    pm = probability_metrics(final_y, p)
    tm = threshold_metrics(final_y, p, threshold)

    target = get_target_row(
        "WCMC/NYP",
        manuscript_model,
        "MIMIC-IV → WCMC/NYP",
    )

    # Strict audit against final stored manuscript point estimates.
    if abs(pm["AUROC"] - float(target["AUROC"])) > 5e-5:
        raise RuntimeError(
            f"WCMC {model}: AUROC source mismatch "
            f"({pm['AUROC']:.6f} vs stored {float(target['AUROC']):.6f})."
        )

    if abs(pm["AUPRC"] - float(target["AUPRC"])) > 5e-5:
        raise RuntimeError(
            f"WCMC {model}: AUPRC source mismatch "
            f"({pm['AUPRC']:.6f} vs stored {float(target['AUPRC']):.6f})."
        )

    if abs(pm["Brier"] - float(target["Brier"])) > 5e-5:
        raise RuntimeError(
            f"WCMC {model}: Brier source mismatch "
            f"({pm['Brier']:.6f} vs stored {float(target['Brier']):.6f})."
        )

    if abs(tm["F1"] - float(target["F1"])) > 5e-4:
        raise RuntimeError(
            f"WCMC {model}: threshold/F1 mismatch "
            f"({tm['F1']:.6f} vs stored {float(target['F1']):.6f})."
        )

    update_accuracy(
        "WCMC/NYP",
        manuscript_model,
        "MIMIC-IV → WCMC/NYP",
        tm["Accuracy"],
    )

    accuracy_audit.append({
        "Dataset": "WCMC/NYP",
        "Model": model,
        "Training strategy": "MIMIC-IV → WCMC/NYP",
        "Accuracy": tm["Accuracy"],
        "Threshold": threshold,
        "Threshold source": "Frozen WCMC validation threshold",
        "Prediction source": str(WCMC_FINAL_PRED),
        "Source status": "EXACT FINAL 5-SEED",
    })


# ------------------------------------------------------------------------------------------
# B. WCMC-ONLY BASELINES WITH MISSING ACCURACY
# ------------------------------------------------------------------------------------------

LOCAL_MODEL_SPECS = {
    "ResNet1D": {
        "include_any": ["resnet", "resnet1d"],
        "exclude": [
            "5seed", "vcg", "external", "ptb", "chapman",
            "transfer", "mimic_to_wcmc",
        ],
    },
    "CNN-LSTM": {
        "include_any": ["cnnlstm", "cnn_lstm", "cnn-lstm"],
        "exclude": ["external", "ptb", "chapman", "vcg"],
    },
    "CNN": {
        "include_any": ["simplecnn", "cnn_only", "cnn1d"],
        "exclude": [
            "cnnlstm", "cnn_lstm", "lstm", "resnet", "transformer",
            "tcn", "trism", "external", "ptb", "chapman", "vcg",
        ],
    },
    "Transformer": {
        "include_any": ["puretransformer", "transformer_only", "transformer"],
        "exclude": [
            "trism", "cnn", "tcn", "resnet", "external",
            "ptb", "chapman", "vcg",
        ],
    },
}


def path_matches_spec(path, spec):
    s = str(path).lower()

    if not any(token in s for token in spec["include_any"]):
        return False

    if any(token in s for token in spec["exclude"]):
        return False

    return True


def prediction_file_score(path, split):
    s = str(path).lower()
    name = path.name.lower()
    score = 0

    if "predict" in name:
        score += 8

    if split == "test":
        if "locked" in name:
            score += 7
        if "test" in name:
            score += 6
        if "validation" in name or "_val" in name:
            score -= 10

    if split == "validation":
        if "validation" in name:
            score += 8
        elif "_val" in name or "val_" in name:
            score += 6
        if "locked" in name or "test" in name:
            score -= 10

    if "wcmc" in s:
        score += 3

    if "baseline" in s:
        score += 1

    return score


all_wcmc_csv = sorted(WCMC_ROOT.rglob("*.csv"))


def candidate_probability_column(columns):
    # Do not accidentally use prob_vcg/prob_resnet from unrelated combined files here.
    return find_col(
        columns,
        [
            "prob_male",
            "y_prob",
            "pred_prob",
            "probability",
            "prob",
            "score",
        ],
    )


def discover_local_test_prediction(model, spec, target):
    candidates = []

    for path in all_wcmc_csv:
        if not path_matches_spec(path, spec):
            continue

        score = prediction_file_score(path, "test")

        if score <= 0:
            continue

        try:
            head = pd.read_csv(path, nrows=5, low_memory=False)
        except Exception:
            continue

        ycol = find_col(head.columns, LABEL_CANDIDATES)
        pcol = candidate_probability_column(head.columns)

        if ycol is None or pcol is None:
            continue

        candidates.append((score, path, pcol))

    candidates.sort(key=lambda x: (-x[0], str(x[1])))

    accepted = []

    for score, path, pcol in candidates:
        try:
            raw = pd.read_csv(path, low_memory=False)

            if len(raw) != EXPECTED_N["WCMC/NYP"]:
                continue

            pred = standardize_prediction(raw, prob_col=pcol)

            if not np.array_equal(
                pred["y"].to_numpy(int),
                reference_y,
            ):
                continue

            pm = probability_metrics(
                pred["y"].to_numpy(int),
                pred["p"].to_numpy(float),
            )

            # Match the exact stored model result, not just the filename.
            if abs(pm["AUROC"] - float(target["AUROC"])) > 0.0015:
                continue
            if abs(pm["AUPRC"] - float(target["AUPRC"])) > 0.0015:
                continue
            if abs(pm["Brier"] - float(target["Brier"])) > 0.0015:
                continue

            accepted.append((score, path, pcol, raw, pred, pm))

        except Exception:
            continue

    if len(accepted) == 0:
        raise RuntimeError(
            f"Could not find a verified locked-test prediction file for "
            f"WCMC-only {model}. No accuracy will be invented."
        )

    # Highest path score; metric matching has already been enforced.
    accepted.sort(key=lambda x: (-x[0], str(x[1])))

    # Reject unresolved ties at the same score from different sources.
    top_score = accepted[0][0]
    top = [x for x in accepted if x[0] == top_score]

    if len(top) > 1:
        # If identical probability vectors, they are duplicate copies and safe.
        base_p = top[0][4]["p"].to_numpy(float)

        if not all(
            np.array_equal(base_p, x[4]["p"].to_numpy(float))
            for x in top[1:]
        ):
            raise RuntimeError(
                f"Multiple non-identical verified locked-test prediction sources "
                f"found for WCMC-only {model}:\n"
                + "\n".join(str(x[1]) for x in top)
            )

    return accepted[0]


def discover_validation_threshold(model, spec, test_path, target_f1):
    candidates = []

    for path in all_wcmc_csv:
        if path == test_path:
            continue

        if not path_matches_spec(path, spec):
            continue

        score = prediction_file_score(path, "validation")

        if score <= 0:
            continue

        try:
            head = pd.read_csv(path, nrows=5, low_memory=False)
        except Exception:
            continue

        ycol = find_col(head.columns, LABEL_CANDIDATES)
        pcol = candidate_probability_column(head.columns)

        if ycol is None or pcol is None:
            continue

        # Favor files near the verified locked-test source.
        try:
            common = Path(
                __import__("os").path.commonpath(
                    [str(path.parent), str(test_path.parent)]
                )
            )
            proximity = len(common.parts)
        except Exception:
            proximity = 0

        candidates.append((score + 0.1 * proximity, path, pcol))

    candidates.sort(key=lambda x: (-x[0], str(x[1])))

    viable = []

    for score, path, pcol in candidates:
        try:
            raw = pd.read_csv(path, low_memory=False)

            if len(raw) != 10687:
                continue

            val = standardize_prediction(raw, prob_col=pcol)
            threshold, val_f1 = max_f1_threshold(
                val["y"].to_numpy(int),
                val["p"].to_numpy(float),
            )

            viable.append((score, path, pcol, threshold, val_f1))

        except Exception:
            continue

    if not viable:
        raise RuntimeError(
            f"WCMC-only {model}: verified locked-test probabilities were found, "
            "but no matching N=10,687 validation prediction file could be identified "
            "to recover the prespecified maximum-F1 threshold."
        )

    return viable


for model, spec in LOCAL_MODEL_SPECS.items():
    target = get_target_row(
        "WCMC/NYP",
        model,
        "WCMC/NYP-only",
    )

    score, test_path, pcol, test_raw, test_pred, pm = discover_local_test_prediction(
        model,
        spec,
        target,
    )

    y_test = test_pred["y"].to_numpy(int)
    p_test = test_pred["p"].to_numpy(float)

    # First preference: binary predictions already saved by the original run.
    yhat_saved, yhat_col = extract_binary_prediction(test_raw)

    if yhat_saved is not None:
        saved_f1 = float(f1_score(y_test, yhat_saved, zero_division=0))

        if abs(saved_f1 - float(target["F1"])) <= 0.0015:
            acc = float(accuracy_score(y_test, yhat_saved))
            threshold_used = np.nan
            threshold_source = f"Saved locked-test binary prediction column: {yhat_col}"

            update_accuracy(
                "WCMC/NYP",
                model,
                "WCMC/NYP-only",
                acc,
            )

            accuracy_audit.append({
                "Dataset": "WCMC/NYP",
                "Model": model,
                "Training strategy": "WCMC/NYP-only",
                "Accuracy": acc,
                "Threshold": threshold_used,
                "Threshold source": threshold_source,
                "Prediction source": str(test_path),
                "Source status": "VERIFIED BY AUROC/AUPRC/BRIER/F1",
            })

            continue

    # Second preference: saved scalar threshold in the locked-test output.
    threshold_saved, threshold_col = extract_scalar_threshold(test_raw)

    if threshold_saved is not None:
        tm = threshold_metrics(y_test, p_test, threshold_saved)

        if abs(tm["F1"] - float(target["F1"])) <= 0.0015:
            update_accuracy(
                "WCMC/NYP",
                model,
                "WCMC/NYP-only",
                tm["Accuracy"],
            )

            accuracy_audit.append({
                "Dataset": "WCMC/NYP",
                "Model": model,
                "Training strategy": "WCMC/NYP-only",
                "Accuracy": tm["Accuracy"],
                "Threshold": threshold_saved,
                "Threshold source": f"Saved threshold column: {threshold_col}",
                "Prediction source": str(test_path),
                "Source status": "VERIFIED BY AUROC/AUPRC/BRIER/F1",
            })

            continue

    # Otherwise recover the threshold strictly from validation predictions.
    validation_candidates = discover_validation_threshold(
        model,
        spec,
        test_path,
        float(target["F1"]),
    )

    matches = []

    for vscore, val_path, val_pcol, threshold, val_f1 in validation_candidates:
        tm = threshold_metrics(y_test, p_test, threshold)
        f1_gap = abs(tm["F1"] - float(target["F1"]))

        if f1_gap <= 0.0015:
            matches.append(
                (
                    f1_gap,
                    -vscore,
                    str(val_path),
                    val_path,
                    threshold,
                    val_f1,
                    tm,
                )
            )

    if not matches:
        raise RuntimeError(
            f"WCMC-only {model}: no validation-derived maximum-F1 threshold "
            f"reproduced the stored locked-test F1={float(target['F1']):.6f}. "
            "Accuracy will not be guessed."
        )

    matches.sort()
    _, _, _, val_path, threshold, val_f1, tm = matches[0]

    update_accuracy(
        "WCMC/NYP",
        model,
        "WCMC/NYP-only",
        tm["Accuracy"],
    )

    accuracy_audit.append({
        "Dataset": "WCMC/NYP",
        "Model": model,
        "Training strategy": "WCMC/NYP-only",
        "Accuracy": tm["Accuracy"],
        "Threshold": threshold,
        "Threshold source": (
            f"Maximum validation F1 ({val_f1:.6f}) from {val_path}"
        ),
        "Prediction source": str(test_path),
        "Source status": "VERIFIED BY AUROC/AUPRC/BRIER/F1",
    })


# ------------------------------------------------------------------------------------------
# C. EXTERNAL PTB-XL / CHAPMAN-NINGBO FINAL MATCHED MODELS
# ------------------------------------------------------------------------------------------

EXTERNAL_SEARCH_ROOTS = {
    "PTB-XL": [
        EXT_ROOT,
        RUN / "ptbxl_ecgsex_from_scratch",
    ],
    "Chapman/Ningbo": [
        EXT_ROOT,
        RUN / "chapman_ningbo_ecgsex_from_scratch",
    ],
}

DATASET_TOKENS = {
    "PTB-XL": ["ptb", "ptbxl"],
    "Chapman/Ningbo": ["chapman", "ningbo"],
}


def external_perf_accuracy(dataset, model):
    """
    Use final external performance table only if it explicitly contains Accuracy.
    """
    if not EXT_PERF.exists():
        return None

    perf = pd.read_csv(EXT_PERF, low_memory=False)

    dcol = find_col(perf.columns, ["dataset", "cohort"])
    mcol = find_col(perf.columns, ["model"])
    acol = find_col(perf.columns, ["accuracy"])

    if dcol is None or mcol is None or acol is None:
        return None

    def model_match(x):
        s = str(x).lower()
        if model == "VCG-Geometry ResNet":
            return ("vcg" in s) and ("resnet" in s)
        return ("resnet" in s) and ("vcg" not in s)

    mask = (
        perf[dcol].astype(str).str.lower().apply(
            lambda s: (
                "ptb" in s
                if dataset == "PTB-XL"
                else ("chapman" in s or "ningbo" in s)
            )
        )
        & perf[mcol].apply(model_match)
    )

    sub = perf.loc[mask]

    if len(sub) != 1:
        return None

    value = pd.to_numeric(sub.iloc[0][acol], errors="coerce")

    if pd.isna(value):
        return None

    value = float(value)

    if not (0 <= value <= 1):
        return None

    return value


def collect_external_csv(dataset):
    seen = set()
    out = []

    for root in EXTERNAL_SEARCH_ROOTS[dataset]:
        if not root.exists():
            continue

        for path in root.rglob("*.csv"):
            if path in seen:
                continue
            seen.add(path)

            s = str(path).lower()

            if not any(tok in s for tok in DATASET_TOKENS[dataset]):
                continue

            if any(
                bad in s
                for bad in [
                    "bootstrap",
                    "paireddifference",
                    "paired_difference",
                    "performance",
                    "source",
                    "summary",
                    "audit",
                ]
            ):
                continue

            out.append(path)

    return sorted(out)


def external_candidate_score(path, dataset, model):
    s = str(path).lower()
    name = path.name.lower()
    score = 0

    if "predict" in name:
        score += 8
    if "final" in s:
        score += 5
    if "ensemble" in s or "5seed" in s:
        score += 4
    if "aligned" in name:
        score += 4
    if "zero" in s or "external" in s:
        score += 3

    if model == "VCG-Geometry ResNet":
        if "vcg" in s:
            score += 5
    else:
        if "resnet" in s:
            score += 5
        if "vcg" in name and "resnet" not in name:
            score -= 5

    return score


def select_external_probability_column(raw, model):
    if model == "VCG-Geometry ResNet" and "prob_vcg" in raw.columns:
        return "prob_vcg"

    if model == "ResNet1D" and "prob_resnet" in raw.columns:
        return "prob_resnet"

    return candidate_probability_column(raw.columns)


def discover_external_prediction(dataset, model, target):
    expected_n = EXPECTED_N[dataset]
    candidates = []

    for path in collect_external_csv(dataset):
        score = external_candidate_score(path, dataset, model)

        if score <= 0:
            continue

        try:
            head = pd.read_csv(path, nrows=5, low_memory=False)
        except Exception:
            continue

        if find_col(head.columns, LABEL_CANDIDATES) is None:
            continue

        pcol = select_external_probability_column(head, model)

        if pcol is None:
            continue

        candidates.append((score, path, pcol))

    candidates.sort(key=lambda x: (-x[0], str(x[1])))

    verified = []

    threshold = (
        VCG_THRESHOLD
        if model == "VCG-Geometry ResNet"
        else RESNET_THRESHOLD
    )

    for score, path, pcol in candidates:
        try:
            raw = pd.read_csv(path, low_memory=False)

            if len(raw) != expected_n:
                continue

            pred = standardize_prediction(raw, prob_col=pcol)
            y = pred["y"].to_numpy(int)
            p = pred["p"].to_numpy(float)

            pm = probability_metrics(y, p)

            if abs(pm["AUROC"] - float(target["AUROC"])) > 0.0015:
                continue
            if abs(pm["AUPRC"] - float(target["AUPRC"])) > 0.0015:
                continue
            if abs(pm["Brier"] - float(target["Brier"])) > 0.0015:
                continue

            tm = threshold_metrics(y, p, threshold)

            if abs(tm["F1"] - float(target["F1"])) > 0.0020:
                continue

            verified.append((score, path, pcol, tm, pm))

        except Exception:
            continue

    if not verified:
        return None

    verified.sort(key=lambda x: (-x[0], str(x[1])))

    top_score = verified[0][0]
    top = [x for x in verified if x[0] == top_score]

    if len(top) > 1:
        # Multiple verified copies are acceptable only if their resulting accuracy agrees.
        accs = np.array([x[3]["Accuracy"] for x in top], dtype=float)

        if np.max(accs) - np.min(accs) > 1e-12:
            raise RuntimeError(
                f"{dataset} / {model}: multiple equally ranked verified external "
                f"prediction sources yield different accuracies:\n"
                + "\n".join(str(x[1]) for x in top)
            )

    return verified[0]


for dataset in ["PTB-XL", "Chapman/Ningbo"]:
    for model in ["ResNet1D", "VCG-Geometry ResNet"]:
        strategy = "MIMIC-IV → WCMC/NYP; frozen external evaluation"

        target = get_target_row(
            dataset,
            model,
            strategy,
        )

        # Prefer patient-level prediction reconstruction.
        found = discover_external_prediction(
            dataset,
            model,
            target,
        )

        if found is not None:
            score, pred_path, pcol, tm, pm = found

            update_accuracy(
                dataset,
                model,
                strategy,
                tm["Accuracy"],
            )

            accuracy_audit.append({
                "Dataset": dataset,
                "Model": model,
                "Training strategy": strategy,
                "Accuracy": tm["Accuracy"],
                "Threshold": (
                    VCG_THRESHOLD
                    if model == "VCG-Geometry ResNet"
                    else RESNET_THRESHOLD
                ),
                "Threshold source": "Frozen WCMC validation threshold",
                "Prediction source": str(pred_path),
                "Source status": "VERIFIED BY AUROC/AUPRC/BRIER/F1",
            })

            continue

        # If final performance output explicitly stored accuracy, use that audited result.
        perf_acc = external_perf_accuracy(dataset, model)

        if perf_acc is not None:
            update_accuracy(
                dataset,
                model,
                strategy,
                perf_acc,
            )

            accuracy_audit.append({
                "Dataset": dataset,
                "Model": model,
                "Training strategy": strategy,
                "Accuracy": perf_acc,
                "Threshold": (
                    VCG_THRESHOLD
                    if model == "VCG-Geometry ResNet"
                    else RESNET_THRESHOLD
                ),
                "Threshold source": "Frozen WCMC threshold used by final external evaluation",
                "Prediction source": str(EXT_PERF),
                "Source status": "FINAL EXTERNAL PERFORMANCE TABLE",
            })

            continue

        raise RuntimeError(
            f"Could not recover a verified Accuracy for {dataset} / {model}. "
            "The script intentionally stops rather than leave a dash or invent a value."
        )


# ------------------------------------------------------------------------------------------
# D. AUDIT PRE-EXISTING ACCURACY VALUES + REQUIRE COMPLETE TABLE
# ------------------------------------------------------------------------------------------

for _, row in df.iterrows():
    if pd.isna(row["Accuracy"]):
        raise RuntimeError(
            "Accuracy remains missing after recovery for:\n"
            f"{row[['Dataset','Model','Training strategy']].to_dict()}"
        )

# Existing original benchmark values were already audited in prior analyses;
# record them in the source audit so every manuscript row is traceable.
for _, row in df.iterrows():
    exists = any(
        (
            a["Dataset"] == row["Dataset"]
            and a["Model"] == row["Model"]
            and a["Training strategy"] == row["Training strategy"]
        )
        for a in accuracy_audit
    )

    if exists:
        continue

    accuracy_audit.append({
        "Dataset": row["Dataset"],
        "Model": row["Model"],
        "Training strategy": row["Training strategy"],
        "Accuracy": float(row["Accuracy"]),
        "Threshold": np.nan,
        "Threshold source": "Previously audited final manuscript result",
        "Prediction source": "",
        "Source status": "PRE-EXISTING AUDITED VALUE",
    })

accuracy_audit_df = pd.DataFrame(accuracy_audit)

accuracy_audit_df = accuracy_audit_df.sort_values(
    ["Dataset", "Model", "Training strategy"]
).reset_index(drop=True)

accuracy_audit_df.to_csv(
    ACCURACY_AUDIT_OUT,
    index=False,
    encoding="utf-8-sig",
)

print("\n" + "=" * 140)
print("TABLE 3 ACCURACY AUDIT")
print("=" * 140)
print(
    accuracy_audit_df[
        [
            "Dataset",
            "Model",
            "Training strategy",
            "Accuracy",
            "Threshold",
            "Threshold source",
            "Prediction source",
            "Source status",
        ]
    ].to_string(index=False)
)



# ==========================================================================================
# FILL EXTERNAL 95% CIs FROM FINAL REAL BOOTSTRAP DISTRIBUTIONS
# ==========================================================================================

def bootstrap_ci(path, column):
    if not path.exists():
        raise FileNotFoundError(path)
    boot = pd.read_csv(path, low_memory=False)
    if column not in boot.columns:
        raise RuntimeError(
            f"Missing bootstrap column {column!r} in {path}\n"
            f"Available: {boot.columns.tolist()}"
        )
    vals = pd.to_numeric(boot[column], errors="coerce").dropna().to_numpy(float)
    if vals.size < 1000:
        raise RuntimeError(
            f"Insufficient bootstrap replicates for {path.name} / {column}: {vals.size:,}"
        )
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


external_boot = {
    "PTB-XL": PTB_BOOT,
    "Chapman/Ningbo": CHAPMAN_BOOT,
}

for dataset, boot_path in external_boot.items():
    for model, prefix in [
        ("ResNet1D", "ResNet"),
        ("VCG-Geometry ResNet", "VCG"),
    ]:
        mask = (df["Dataset"] == dataset) & (df["Model"] == model)
        if mask.sum() != 1:
            raise RuntimeError(
                f"Expected one row for {dataset} / {model}; found {int(mask.sum())}"
            )

        auc_l, auc_u = bootstrap_ci(boot_path, f"{prefix}__AUROC")
        ap_l, ap_u = bootstrap_ci(boot_path, f"{prefix}__AUPRC")

        df.loc[mask, "AUROC_L"] = auc_l
        df.loc[mask, "AUROC_U"] = auc_u
        df.loc[mask, "AUPRC_L"] = ap_l
        df.loc[mask, "AUPRC_U"] = ap_u

print("\nExternal bootstrap CIs loaded from final 2,000-replicate files.")


# ==========================================================================================
# FORMAT HELPERS
# ==========================================================================================

def fmt_ci(point, lo, hi):
    if pd.isna(point):
        return "—"
    if pd.isna(lo) or pd.isna(hi):
        return f"{point:.3f}"
    return f"{point:.3f} [{lo:.3f}–{hi:.3f}]"


def fmt_metric(x):
    return "—" if pd.isna(x) else f"{x:.3f}"


# ==========================================================================================
# SAVE LONG-FORM SOURCE TABLE
# ==========================================================================================

df.to_csv(
    LONG_OUT,
    index=False,
    encoding="utf-8-sig",
)


# ==========================================================================================
# MANUSCRIPT-FORMAT TABLE
# ==========================================================================================

manuscript = pd.DataFrame({
    "Dataset": df["Dataset"],
    "Evaluation": df["Evaluation"],
    "N": df["N"].map(lambda x: f"{int(x):,}"),
    "Model": df["Model"],
    "Training strategy": df["Training strategy"],
    "AUROC (95% CI)": [
        fmt_ci(p, l, u)
        for p, l, u in zip(df["AUROC"], df["AUROC_L"], df["AUROC_U"])
    ],
    "AUPRC (95% CI)": [
        fmt_ci(p, l, u)
        for p, l, u in zip(df["AUPRC"], df["AUPRC_L"], df["AUPRC_U"])
    ],
    "Accuracy": df["Accuracy"].map(fmt_metric),
    "F1 score": df["F1"].map(fmt_metric),
    "Brier score": df["Brier"].map(fmt_metric),
})

manuscript.to_csv(
    MANUSCRIPT_OUT,
    index=False,
    encoding="utf-8-sig",
)


# ==========================================================================================
# MANUSCRIPT FOOTNOTE
# ==========================================================================================

footnote = (
    "Values in brackets indicate 95% bootstrap confidence intervals. "
    "Point estimates are shown without confidence intervals where bootstrap distributions "
    "were not available. Accuracy and F1 were calculated using validation-selected operating "
    "thresholds; for PTB-XL and Chapman/Ningbo, the WCMC/NYP validation thresholds were "
    "kept frozen without external threshold adjustment. External evaluations were performed "
    "without retraining or recalibration."
)

FOOTNOTE_OUT.write_text(footnote, encoding="utf-8")


# ==========================================================================================
# AUDIT
# ==========================================================================================

expected_dataset_n = {
    "WCMC/NYP": 21374,
    "PTB-XL": 18752,
    "Chapman/Ningbo": 42997,
}

for dataset, expected_n in expected_dataset_n.items():
    observed = set(
        df.loc[df["Dataset"] == dataset, "N"].astype(int).tolist()
    )
    if observed != {expected_n}:
        raise RuntimeError(
            f"{dataset}: unexpected N values {sorted(observed)}; expected {expected_n:,}"
        )

# Final VCG should have the highest AUROC in each final evaluation dataset.
for dataset in expected_dataset_n:
    sub = df[df["Dataset"] == dataset]
    best_model = sub.loc[sub["AUROC"].idxmax(), "Model"]
    if best_model != "VCG-Geometry ResNet":
        raise RuntimeError(
            f"{dataset}: highest AUROC is unexpectedly {best_model}"
        )


# ==========================================================================================
# PRINT
# ==========================================================================================

print("=" * 140)
print("TABLE 3 — ECG-SEX COMPARATIVE PERFORMANCE TABLE CREATED")
print("=" * 140)

print("\nLong-format source CSV:")
print(LONG_OUT)

print("\nManuscript-format CSV:")
print(MANUSCRIPT_OUT)

print("\nManuscript table preview:\n")
print(manuscript.to_string(index=False))

print("\nFootnote:")
print(footnote)
print("\nFootnote file:")
print(FOOTNOTE_OUT)

print("\nAccuracy source audit:")
print(ACCURACY_AUDIT_OUT)

print("\nDONE")
