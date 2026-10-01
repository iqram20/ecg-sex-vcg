#!/usr/bin/env python3
# ==========================================================================================
# ECG-SEX REVIEWER ANALYSIS
# WCMC-ONLY VCG-GEOMETRY RESNET vs WCMC-ONLY RESNET1D — 5 SEEDS
# + 2 x 2 ARCHITECTURE × PRETRAINING COMPARISON
# ==========================================================================================
#
# PURPOSE
# -------
# Address the reviewer concern that the final VCG-Geometry ResNet was evaluated only after
# MIMIC-IV pretraining of the waveform pathway. This analysis asks whether:
#
#   (1) VCG geometry improves over waveform-only ResNet1D when BOTH are trained from scratch
#       using WCMC/NYP only; and
#   (2) MIMIC-IV pretraining provides benefit within BOTH architectures.
#
# The final 2 x 2 comparison is:
#
#                         WCMC-only scratch        MIMIC -> WCMC transfer
#   ResNet1D              newly trained here       existing final 5-seed ensemble
#   VCG-Geometry ResNet   newly trained here       existing final 5-seed ensemble
#
# IMPORTANT DESIGN
# ----------------
# - Exact same WCMC train / validation / locked-test participant partitions as the final study.
# - Exact same final VCG-Geometry architecture and preprocessing are IMPORTED from the
#   original final training script; they are not reimplemented here.
# - WCMC-only models are initialized from scratch.
# - No MIMIC checkpoint is loaded into either WCMC-only model.
# - WCMC-only optimization follows the manuscript's local-training protocol:
#       AdamW, LR=3e-4 for all trainable parameters, WD=1e-4,
#       unweighted BCE, no scheduler, max 30 epochs, patience 5,
#       checkpoint by validation AUPRC, threshold by validation F1.
# - The VCG inverse-Dower projection retains the same canonical initialization, bounded
#   correction, geometry branch, gated residual fusion, and Dower regularization as the
#   final architecture. "From scratch" refers to learned neural-network weights, not removal
#   of the physiologic inverse-Dower initialization.
# - Five seeds: 11, 22, 33, 42, 55.
# - Locked test is evaluated only after validation-selected checkpoints are fixed.
# - FINAL mode forms equal-probability five-seed ensembles and performs 2,000 paired
#   patient-level bootstrap replicates.
#
# MODES
# -----
# WCMCONLY_MODE=RUN
#     Train BOTH WCMC-only VCG-Geometry ResNet and WCMC-only ResNet1D for one seed.
#
# WCMCONLY_MODE=FINAL
#     Require all five RUN jobs, build local ensembles, merge with the already-completed
#     transferred five-seed ensembles, and generate the final 2 x 2 comparison.
#
# Female=0, Male=1
# ==========================================================================================

from __future__ import annotations

import os

# Keep import of the original final script inert and valid.
os.environ["VCGFINAL_MODE"] = "RUN"
os.environ["VCGFINAL_SEED"] = "42"

import gc
import json
import math
import random
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    brier_score_loss,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==========================================================================================
# CONFIG
# ==========================================================================================

MODE = os.environ.get("WCMCONLY_MODE", "RUN").strip().upper()
SEED = int(os.environ.get("WCMCONLY_SEED", "42"))

SEEDS = [11, 22, 33, 42, 55]

if MODE == "RUN" and SEED not in SEEDS:
    raise ValueError(f"Unsupported WCMCONLY_SEED={SEED}; expected one of {SEEDS}")

BATCH_SIZE = int(os.environ.get("WCMCONLY_BATCH_SIZE", "64"))
NUM_WORKERS = int(os.environ.get("WCMCONLY_NUM_WORKERS", "8"))
MAX_EPOCHS = int(os.environ.get("WCMCONLY_MAX_EPOCHS", "30"))
PATIENCE = int(os.environ.get("WCMCONLY_PATIENCE", "5"))

LR_SCRATCH = float(os.environ.get("WCMCONLY_LR", "3e-4"))
WEIGHT_DECAY = float(os.environ.get("WCMCONLY_WEIGHT_DECAY", "1e-4"))
GRAD_CLIP = float(os.environ.get("WCMCONLY_GRAD_CLIP", "5.0"))

MAX_DOWER_DELTA = float(os.environ.get("WCMCONLY_MAX_DOWER_DELTA", "0.03"))
DOWER_REG_LAMBDA = float(os.environ.get("WCMCONLY_DOWER_REG", "1e-3"))

N_BOOT = int(os.environ.get("WCMCONLY_N_BOOT", "2000"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================================================================
# PATHS
# ==========================================================================================

CODE_DIR = Path("/athena/madelab/scratch/iqh4001/ECG/ECG-gender/Code")

ORIGINAL_SOURCE_CANDIDATES = [
    CODE_DIR / "WCMC_ECGSex_VCGGeometry_ResNet_5Seed_LockedTest.py",
    Path(
        "/athena/madelab/scratch/iqh4001/ECG/GitHub/ecg-sex-vcg/"
        "scripts/training/WCMC_ECGSex_VCGGeometry_ResNet_5Seed_LockedTest.py"
    ),
]

ORIGINAL_SOURCE = next(
    (p for p in ORIGINAL_SOURCE_CANDIDATES if p.exists()),
    None,
)

if ORIGINAL_SOURCE is None:
    raise FileNotFoundError(
        "Could not locate original final VCG training script.\nTried:\n"
        + "\n".join(str(p) for p in ORIGINAL_SOURCE_CANDIDATES)
    )

RUN_DIR = Path(
    "/athena/madelab/scratch/iqh4001/ECG/ECG_Echo/Runs/"
    "Master_ECG_Sex_MIMIC_EchoNext/"
    "20260609_020154_MASTER_ECG_SEX_MIMIC_ECHONEXT"
)

WCMC_ROOT = RUN_DIR / "wcmc_ecgsex_from_scratch"

SPLIT_DIR = WCMC_ROOT / "04_wcmc_finetune_from_mimic" / "splits"
TRAIN_CSV = SPLIT_DIR / "WCMC_ECGSex_train.csv"
VAL_CSV = SPLIT_DIR / "WCMC_ECGSex_validation.csv"
TEST_CSV = SPLIT_DIR / "WCMC_ECGSex_test.csv"

TRANSFER_FINAL_DIR = (
    WCMC_ROOT
    / "19_vcg_geometry_resnet_5seed_locked_wcmc"
    / "final_5seed_ensemble_bootstrap"
)

TRANSFER_VAL = (
    TRANSFER_FINAL_DIR
    / "WCMC_Validation_Aligned_5SeedEnsemble_Predictions.csv"
)

TRANSFER_TEST = (
    TRANSFER_FINAL_DIR
    / "WCMC_LockedTest_Aligned_5SeedEnsemble_Predictions.csv"
)

OUT_ROOT = (
    WCMC_ROOT
    / "21_wcmc_only_vcg_resnet_5seed_2x2"
)

FINAL_DIR = OUT_ROOT / "final_5seed_2x2_bootstrap"


# ==========================================================================================
# IMPORT EXACT FINAL ARCHITECTURE / PREPROCESSING
# ==========================================================================================

spec = importlib.util.spec_from_file_location(
    "ecgsex_final_vcg_source",
    ORIGINAL_SOURCE,
)

if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not import original source: {ORIGINAL_SOURCE}")

final = importlib.util.module_from_spec(spec)
spec.loader.exec_module(final)

# Force the imported helpers to use this analysis configuration.
final.BATCH_SIZE = BATCH_SIZE
final.NUM_WORKERS = NUM_WORKERS
final.DOWER_REG_LAMBDA = DOWER_REG_LAMBDA
final.MAX_DOWER_DELTA = MAX_DOWER_DELTA
final.GRAD_CLIP = GRAD_CLIP
final.DEVICE = DEVICE


# ==========================================================================================
# UTILITIES
# ==========================================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Match the existing final pipeline behavior.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def seed_dir(seed: int) -> Path:
    return OUT_ROOT / "seeds" / f"seed_{seed}"


def check_splits():
    for p in [TRAIN_CSV, VAL_CSV, TEST_CSV, TRANSFER_VAL, TRANSFER_TEST]:
        if not p.exists():
            raise FileNotFoundError(p)

    train_df = pd.read_csv(TRAIN_CSV, low_memory=False)
    val_df = pd.read_csv(VAL_CSV, low_memory=False)
    test_df = pd.read_csv(TEST_CSV, low_memory=False)

    for df, name in [(train_df, "train"), (val_df, "validation"), (test_df, "test")]:
        required = {"patient_key", "ecg_key", "sex_binary", "raw_local_path"}
        missing = required - set(df.columns)
        if missing:
            raise RuntimeError(f"{name} split missing columns: {sorted(missing)}")

    tr = set(train_df["patient_key"].astype(str))
    va = set(val_df["patient_key"].astype(str))
    te = set(test_df["patient_key"].astype(str))

    if tr & va or tr & te or va & te:
        raise RuntimeError("Patient overlap detected across WCMC train/validation/test splits.")

    return train_df, val_df, test_df


def make_loader(df: pd.DataFrame, shuffle: bool):
    return final.make_loader(df, shuffle=shuffle)


def select_f1_threshold(y, p):
    return final.select_f1_threshold(
        np.asarray(y, dtype=int),
        np.asarray(p, dtype=float),
    )


def metrics_at_threshold(y, p, threshold):
    return final.metrics_at_threshold(
        np.asarray(y, dtype=int),
        np.asarray(p, dtype=float),
        float(threshold),
    )


def train_resnet_one_epoch(model, loader, optimizer, criterion, scaler):
    model.train()

    total = 0.0
    total_n = 0
    failures = 0

    for batch in loader:
        failures += len(batch["invalid"])

        if batch["x_base"] is None:
            continue

        x = batch["x_base"].to(
            DEVICE,
            non_blocking=True,
        )

        y = torch.tensor(
            [b["y"] for b in batch["valid"]],
            dtype=torch.float32,
            device=DEVICE,
        )

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=DEVICE.type == "cuda"):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP,
        )

        scaler.step(optimizer)
        scaler.update()

        n = int(y.numel())
        total += float(loss.detach().cpu()) * n
        total_n += n

    if total_n == 0:
        raise RuntimeError("No valid ResNet training samples.")

    return total / total_n, failures


def save_dower_matrix(model, path: Path):
    effective = (
        model.vcg_projection
        .effective_matrix()
        .detach()
        .cpu()
        .numpy()
    )

    base = (
        model.vcg_projection
        .base_matrix
        .detach()
        .cpu()
        .numpy()
    )

    axes = ["X", "Y", "Z"]
    leads = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]

    rows = []
    for i, axis in enumerate(axes):
        for j, lead in enumerate(leads):
            rows.append(
                {
                    "axis": axis,
                    "lead": lead,
                    "base": float(base[i, j]),
                    "effective": float(effective[i, j]),
                    "delta": float(effective[i, j] - base[i, j]),
                }
            )

    pd.DataFrame(rows).to_csv(path, index=False)


# ==========================================================================================
# TRAIN ONE LOCAL MODEL
# ==========================================================================================

def train_vcg_from_scratch(seed, train_df, val_df, outdir):
    print("\n" + "=" * 120)
    print(f"WCMC-ONLY VCG-GEOMETRY RESNET | seed={seed}")
    print("=" * 120)

    set_seed(seed)

    train_loader = make_loader(train_df, shuffle=True)
    val_loader = make_loader(val_df, shuffle=False)

    # CRITICAL: random neural-network initialization. No MIMIC transfer call.
    model = final.VCGGeometryResNet().to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR_SCRATCH,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler(enabled=DEVICE.type == "cuda")

    ckpt = outdir / "best_VCGGeometryResNet_WCMCOnly.pt"
    hist_path = outdir / "VCGGeometryResNet_WCMCOnly_history.csv"

    history = []
    best_auprc = -np.inf
    best_epoch = -1
    no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_failures = final.train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
        )

        val_pred, val_fail = final.predict_vcg(
            model,
            val_loader,
        )

        y = val_pred["sex_true_binary"].to_numpy(dtype=int)
        p = val_pred["prob_male"].to_numpy(dtype=float)

        threshold = select_f1_threshold(y, p)
        met = metrics_at_threshold(y, p, threshold)

        mean_gate = float(val_pred["vcg_gate"].mean())

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_failures": train_failures,
                "validation_failures": len(val_fail),
                "val_AUROC": met["AUROC"],
                "val_AUPRC": met["AUPRC"],
                "val_F1": met["F1"],
                "val_Brier": met["Brier"],
                "val_threshold": threshold,
                "mean_vcg_gate": mean_gate,
            }
        )

        print(
            f"VCG scratch | seed={seed} | epoch={epoch:02d}/{MAX_EPOCHS} | "
            f"loss={train_loss:.5f} | AUROC={met['AUROC']:.6f} | "
            f"AUPRC={met['AUPRC']:.6f} | F1={met['F1']:.6f} | "
            f"Brier={met['Brier']:.6f} | gate={mean_gate:.4f}"
        )

        if met["AUPRC"] > best_auprc + 1e-7:
            best_auprc = met["AUPRC"]
            best_epoch = epoch
            no_improve = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "architecture": "VCG-Geometry ResNet",
                    "training_strategy": "WCMC/NYP-only random initialization",
                    "seed": seed,
                    "epoch": epoch,
                    "best_val_AUPRC": best_auprc,
                    "learning_rate": LR_SCRATCH,
                    "weight_decay": WEIGHT_DECAY,
                    "MAX_DOWER_DELTA": MAX_DOWER_DELTA,
                    "DOWER_REG_LAMBDA": DOWER_REG_LAMBDA,
                },
                ckpt,
            )
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            print(f"VCG early stop at epoch {epoch}; best epoch={best_epoch}.")
            break

    pd.DataFrame(history).to_csv(hist_path, index=False)

    best = torch.load(
        ckpt,
        map_location=DEVICE,
        weights_only=False,
    )
    model.load_state_dict(best["model_state_dict"], strict=True)

    val_pred, val_fail = final.predict_vcg(model, val_loader)
    threshold = select_f1_threshold(
        val_pred["sex_true_binary"],
        val_pred["prob_male"],
    )

    val_pred.to_csv(
        outdir / "VCGGeometryResNet_WCMCOnly_validation_predictions.csv",
        index=False,
    )

    save_dower_matrix(
        model,
        outdir / "VCGGeometryResNet_WCMCOnly_learned_inverse_dower_matrix.csv",
    )

    del train_loader, val_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model, val_pred, threshold, best, val_fail


def train_resnet_from_scratch(seed, train_df, val_df, outdir):
    print("\n" + "=" * 120)
    print(f"WCMC-ONLY RESNET1D | seed={seed}")
    print("=" * 120)

    set_seed(seed)

    train_loader = make_loader(train_df, shuffle=True)
    val_loader = make_loader(val_df, shuffle=False)

    # CRITICAL: exact waveform architecture, random initialization, no MIMIC checkpoint.
    model = final.ResNet1D().to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR_SCRATCH,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler(enabled=DEVICE.type == "cuda")

    ckpt = outdir / "best_ResNet1D_WCMCOnly.pt"
    hist_path = outdir / "ResNet1D_WCMCOnly_history.csv"

    history = []
    best_auprc = -np.inf
    best_epoch = -1
    no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_failures = train_resnet_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
        )

        val_pred, val_fail = final.predict_resnet(
            model,
            val_loader,
        )

        y = val_pred["sex_true_binary"].to_numpy(dtype=int)
        p = val_pred["prob_male"].to_numpy(dtype=float)

        threshold = select_f1_threshold(y, p)
        met = metrics_at_threshold(y, p, threshold)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_failures": train_failures,
                "validation_failures": len(val_fail),
                "val_AUROC": met["AUROC"],
                "val_AUPRC": met["AUPRC"],
                "val_F1": met["F1"],
                "val_Brier": met["Brier"],
                "val_threshold": threshold,
            }
        )

        print(
            f"ResNet scratch | seed={seed} | epoch={epoch:02d}/{MAX_EPOCHS} | "
            f"loss={train_loss:.5f} | AUROC={met['AUROC']:.6f} | "
            f"AUPRC={met['AUPRC']:.6f} | F1={met['F1']:.6f} | "
            f"Brier={met['Brier']:.6f}"
        )

        if met["AUPRC"] > best_auprc + 1e-7:
            best_auprc = met["AUPRC"]
            best_epoch = epoch
            no_improve = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "architecture": "ResNet1D",
                    "training_strategy": "WCMC/NYP-only random initialization",
                    "seed": seed,
                    "epoch": epoch,
                    "best_val_AUPRC": best_auprc,
                    "learning_rate": LR_SCRATCH,
                    "weight_decay": WEIGHT_DECAY,
                },
                ckpt,
            )
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            print(f"ResNet early stop at epoch {epoch}; best epoch={best_epoch}.")
            break

    pd.DataFrame(history).to_csv(hist_path, index=False)

    best = torch.load(
        ckpt,
        map_location=DEVICE,
        weights_only=False,
    )
    model.load_state_dict(best["model_state_dict"], strict=True)

    val_pred, val_fail = final.predict_resnet(model, val_loader)
    threshold = select_f1_threshold(
        val_pred["sex_true_binary"],
        val_pred["prob_male"],
    )

    val_pred.to_csv(
        outdir / "ResNet1D_WCMCOnly_validation_predictions.csv",
        index=False,
    )

    del train_loader, val_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model, val_pred, threshold, best, val_fail


# ==========================================================================================
# RUN MODE — TRAIN BOTH MODELS FOR ONE SEED
# ==========================================================================================

def run_one_seed():
    if DEVICE.type != "cuda":
        raise RuntimeError("RUN mode requires CUDA/GPU.")

    outdir = seed_dir(SEED)
    outdir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = check_splits()

    print("=" * 120)
    print("WCMC-ONLY ARCHITECTURE COMPARISON — ONE SEED")
    print("=" * 120)
    print("Seed:", SEED)
    print("GPU :", torch.cuda.get_device_name(0))
    print("Original architecture source:", ORIGINAL_SOURCE)
    print(f"Train N={len(train_df):,}")
    print(f"Validation N={len(val_df):,}")
    print(f"Locked test N={len(test_df):,}")
    print("Scratch learning rate:", LR_SCRATCH)
    print("No MIMIC checkpoint will be loaded.")

    vcg_model, vcg_val, vcg_thr, vcg_best, vcg_val_fail = train_vcg_from_scratch(
        SEED,
        train_df,
        val_df,
        outdir,
    )

    res_model, res_val, res_thr, res_best, res_val_fail = train_resnet_from_scratch(
        SEED,
        train_df,
        val_df,
        outdir,
    )

    # Test is touched only after BOTH validation-selected checkpoints are fixed.
    test_loader = make_loader(test_df, shuffle=False)

    vcg_test, vcg_test_fail = final.predict_vcg(
        vcg_model,
        test_loader,
    )
    vcg_test.to_csv(
        outdir / "VCGGeometryResNet_WCMCOnly_locked_test_predictions.csv",
        index=False,
    )

    res_test, res_test_fail = final.predict_resnet(
        res_model,
        test_loader,
    )
    res_test.to_csv(
        outdir / "ResNet1D_WCMCOnly_locked_test_predictions.csv",
        index=False,
    )

    vcg_metrics = metrics_at_threshold(
        vcg_test["sex_true_binary"],
        vcg_test["prob_male"],
        vcg_thr,
    )

    res_metrics = metrics_at_threshold(
        res_test["sex_true_binary"],
        res_test["prob_male"],
        res_thr,
    )

    summary = {
        "seed": SEED,
        "protocol": "WCMC/NYP-only random initialization",
        "learning_rate_all_parameters": LR_SCRATCH,
        "weight_decay": WEIGHT_DECAY,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "original_architecture_source": str(ORIGINAL_SOURCE),
        "VCG_best_epoch": int(vcg_best["epoch"]),
        "ResNet_best_epoch": int(res_best["epoch"]),
        "VCG_validation_threshold": float(vcg_thr),
        "ResNet_validation_threshold": float(res_thr),
        "VCG_locked_test": vcg_metrics,
        "ResNet_locked_test": res_metrics,
        "delta_VCG_minus_ResNet": {
            metric: float(vcg_metrics[metric] - res_metrics[metric])
            for metric in ["AUROC", "AUPRC", "F1", "Brier"]
        },
        "VCG_validation_failures": int(len(vcg_val_fail)),
        "ResNet_validation_failures": int(len(res_val_fail)),
        "VCG_test_failures": int(len(vcg_test_fail)),
        "ResNet_test_failures": int(len(res_test_fail)),
    }

    with open(outdir / "seed_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 120)
    print("LOCKED TEST — SEED RESULT")
    print("=" * 120)
    print(json.dumps(summary, indent=2))

    del vcg_model, res_model, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ==========================================================================================
# FINAL MODE — LOCAL ENSEMBLES + 2 x 2 COMPARISON
# ==========================================================================================

def load_local_seed_prediction(seed, model, split):
    root = seed_dir(seed)

    if model == "VCG":
        name = (
            "VCGGeometryResNet_WCMCOnly_validation_predictions.csv"
            if split == "validation"
            else "VCGGeometryResNet_WCMCOnly_locked_test_predictions.csv"
        )
    elif model == "ResNet":
        name = (
            "ResNet1D_WCMCOnly_validation_predictions.csv"
            if split == "validation"
            else "ResNet1D_WCMCOnly_locked_test_predictions.csv"
        )
    else:
        raise ValueError(model)

    path = root / name
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path, low_memory=False)

    for c in ["patient_key", "ecg_key"]:
        df[c] = df[c].astype(str)

    return df


def build_local_ensemble(model, split):
    merged = None

    for seed in SEEDS:
        df = load_local_seed_prediction(seed, model, split)

        keep = df[
            ["patient_key", "ecg_key", "sex_true_binary", "prob_male"]
        ].copy()

        keep = keep.rename(columns={"prob_male": f"prob_seed{seed}"})

        if merged is None:
            merged = keep
        else:
            keep = keep.rename(
                columns={"sex_true_binary": f"sex_seed{seed}"}
            )

            merged = merged.merge(
                keep,
                on=["patient_key", "ecg_key"],
                how="inner",
                validate="one_to_one",
            )

            if not np.array_equal(
                merged["sex_true_binary"].to_numpy(dtype=int),
                merged[f"sex_seed{seed}"].to_numpy(dtype=int),
            ):
                raise RuntimeError(f"{model}: label mismatch across local seeds.")

            merged = merged.drop(columns=[f"sex_seed{seed}"])

    prob_cols = [f"prob_seed{s}" for s in SEEDS]
    merged["prob_ensemble"] = merged[prob_cols].mean(axis=1)

    return merged


def load_transfer(split):
    path = TRANSFER_VAL if split == "validation" else TRANSFER_TEST
    df = pd.read_csv(path, low_memory=False)

    required = {
        "patient_key",
        "ecg_key",
        "sex_true_binary",
        "prob_vcg",
        "prob_resnet",
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"Transferred ensemble file missing columns {sorted(missing)}:\n{path}"
        )

    for c in ["patient_key", "ecg_key"]:
        df[c] = df[c].astype(str)

    return df[
        [
            "patient_key",
            "ecg_key",
            "sex_true_binary",
            "prob_vcg",
            "prob_resnet",
        ]
    ].copy()


def prepare_four_model_table(split):
    local_vcg = build_local_ensemble("VCG", split)[
        ["patient_key", "ecg_key", "sex_true_binary", "prob_ensemble"]
    ].rename(columns={"prob_ensemble": "p_local_vcg"})

    local_res = build_local_ensemble("ResNet", split)[
        ["patient_key", "ecg_key", "sex_true_binary", "prob_ensemble"]
    ].rename(
        columns={
            "sex_true_binary": "sex_local_res",
            "prob_ensemble": "p_local_resnet",
        }
    )

    transfer = load_transfer(split).rename(
        columns={
            "sex_true_binary": "sex_transfer",
            "prob_vcg": "p_transfer_vcg",
            "prob_resnet": "p_transfer_resnet",
        }
    )

    out = local_vcg.merge(
        local_res,
        on=["patient_key", "ecg_key"],
        how="inner",
        validate="one_to_one",
    ).merge(
        transfer,
        on=["patient_key", "ecg_key"],
        how="inner",
        validate="one_to_one",
    )

    y = out["sex_true_binary"].to_numpy(dtype=int)

    if not np.array_equal(y, out["sex_local_res"].to_numpy(dtype=int)):
        raise RuntimeError("Local VCG/ResNet label mismatch.")
    if not np.array_equal(y, out["sex_transfer"].to_numpy(dtype=int)):
        raise RuntimeError("Local/transfer label mismatch.")

    out = out.drop(columns=["sex_local_res", "sex_transfer"])

    return out


MODEL_SPECS = {
    "Local_ResNet": {
        "label": "ResNet1D — WCMC-only",
        "prob": "p_local_resnet",
        "architecture": "ResNet1D",
        "pretraining": "No",
    },
    "Local_VCG": {
        "label": "VCG-Geometry ResNet — WCMC-only",
        "prob": "p_local_vcg",
        "architecture": "VCG-Geometry ResNet",
        "pretraining": "No",
    },
    "Transfer_ResNet": {
        "label": "ResNet1D — MIMIC→WCMC",
        "prob": "p_transfer_resnet",
        "architecture": "ResNet1D",
        "pretraining": "MIMIC-IV waveform pretraining",
    },
    "Transfer_VCG": {
        "label": "VCG-Geometry ResNet — MIMIC→WCMC",
        "prob": "p_transfer_vcg",
        "architecture": "VCG-Geometry ResNet",
        "pretraining": "MIMIC-IV waveform pretraining",
    },
}


def simple_metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)
    return {
        "AUROC": float(roc_auc_score(y, p)),
        "AUPRC": float(average_precision_score(y, p)),
        "Accuracy": float(accuracy_score(y, pred)),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "Brier": float(brier_score_loss(y, p)),
    }


def p_two_sided(delta):
    x = np.asarray(delta, dtype=float)
    if len(x) == 0:
        return np.nan
    p = 2.0 * min(np.mean(x <= 0), np.mean(x >= 0))
    return max(float(p), 1.0 / (len(x) + 1.0))


def final_2x2():
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    missing = []
    for seed in SEEDS:
        p = seed_dir(seed) / "seed_summary.json"
        if not p.exists():
            missing.append(str(p))

    if missing:
        raise RuntimeError(
            "Missing local seed runs; refusing partial finalization:\n"
            + "\n".join(missing)
        )

    print("=" * 120)
    print("FINAL WCMC 2 x 2: ARCHITECTURE × MIMIC PRETRAINING")
    print("=" * 120)

    val = prepare_four_model_table("validation")
    test = prepare_four_model_table("test")

    thresholds = {}
    for key, spec_model in MODEL_SPECS.items():
        thresholds[key] = select_f1_threshold(
            val["sex_true_binary"].to_numpy(dtype=int),
            val[spec_model["prob"]].to_numpy(dtype=float),
        )

    with open(FINAL_DIR / "validation_thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)

    val.to_csv(
        FINAL_DIR / "WCMC_2x2_Aligned_Validation_Predictions.csv",
        index=False,
    )
    test.to_csv(
        FINAL_DIR / "WCMC_2x2_Aligned_LockedTest_Predictions.csv",
        index=False,
    )

    y = test["sex_true_binary"].to_numpy(dtype=int)

    point = {}
    for key, spec_model in MODEL_SPECS.items():
        point[key] = simple_metrics(
            y,
            test[spec_model["prob"]].to_numpy(dtype=float),
            thresholds[key],
        )

    rng = np.random.default_rng(20260930)
    n = len(test)
    boot_rows = []

    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        yy = y[idx]

        if np.unique(yy).size < 2:
            continue

        row = {"bootstrap_id": b + 1}

        for key, spec_model in MODEL_SPECS.items():
            pp = test[spec_model["prob"]].to_numpy(dtype=float)[idx]
            met = simple_metrics(yy, pp, thresholds[key])

            for metric, value in met.items():
                row[f"{key}__{metric}"] = value

        # Architecture effect: VCG minus ResNet within training strategy.
        for metric in ["AUROC", "AUPRC", "Accuracy", "F1", "Brier"]:
            row[f"ArchEffect_Local__{metric}"] = (
                row[f"Local_VCG__{metric}"]
                - row[f"Local_ResNet__{metric}"]
            )
            row[f"ArchEffect_Transfer__{metric}"] = (
                row[f"Transfer_VCG__{metric}"]
                - row[f"Transfer_ResNet__{metric}"]
            )

            # Transfer effect within architecture.
            row[f"TransferEffect_ResNet__{metric}"] = (
                row[f"Transfer_ResNet__{metric}"]
                - row[f"Local_ResNet__{metric}"]
            )
            row[f"TransferEffect_VCG__{metric}"] = (
                row[f"Transfer_VCG__{metric}"]
                - row[f"Local_VCG__{metric}"]
            )

            # Difference-in-differences:
            # Does the architecture increment change after pretraining?
            row[f"Interaction_DiD__{metric}"] = (
                row[f"ArchEffect_Transfer__{metric}"]
                - row[f"ArchEffect_Local__{metric}"]
            )

        boot_rows.append(row)

        if (b + 1) % 100 == 0:
            print(f"bootstrap {b+1:,}/{N_BOOT:,}")

    boot = pd.DataFrame(boot_rows)
    boot.to_csv(
        FINAL_DIR / "WCMC_2x2_PairedBootstrap_Distributions.csv",
        index=False,
    )

    # Performance table.
    perf_rows = []
    for key, spec_model in MODEL_SPECS.items():
        row = {
            "model_key": key,
            "model": spec_model["label"],
            "architecture": spec_model["architecture"],
            "pretraining": spec_model["pretraining"],
            "N_common_locked_test": n,
            "validation_threshold": thresholds[key],
        }

        for metric in ["AUROC", "AUPRC", "Accuracy", "F1", "Brier"]:
            vals = boot[f"{key}__{metric}"].to_numpy(dtype=float)
            row[metric] = point[key][metric]
            row[f"{metric}_CI_low"] = float(np.percentile(vals, 2.5))
            row[f"{metric}_CI_high"] = float(np.percentile(vals, 97.5))

        perf_rows.append(row)

    perf = pd.DataFrame(perf_rows)
    perf.to_csv(
        FINAL_DIR / "WCMC_2x2_ArchitectureTransfer_Performance.csv",
        index=False,
    )

    comparisons = [
        (
            "Architecture effect without pretraining",
            "Local VCG minus Local ResNet",
            "ArchEffect_Local",
            "Local_VCG",
            "Local_ResNet",
        ),
        (
            "Architecture effect after MIMIC pretraining",
            "Transfer VCG minus Transfer ResNet",
            "ArchEffect_Transfer",
            "Transfer_VCG",
            "Transfer_ResNet",
        ),
        (
            "Transfer effect within ResNet1D",
            "Transfer ResNet minus Local ResNet",
            "TransferEffect_ResNet",
            "Transfer_ResNet",
            "Local_ResNet",
        ),
        (
            "Transfer effect within VCG-Geometry ResNet",
            "Transfer VCG minus Local VCG",
            "TransferEffect_VCG",
            "Transfer_VCG",
            "Local_VCG",
        ),
    ]

    diff_rows = []

    for family, comparison, prefix, a_key, b_key in comparisons:
        for metric in ["AUROC", "AUPRC", "Accuracy", "F1", "Brier"]:
            vals = boot[f"{prefix}__{metric}"].to_numpy(dtype=float)
            point_delta = point[a_key][metric] - point[b_key][metric]

            diff_rows.append(
                {
                    "family": family,
                    "comparison": comparison,
                    "metric": metric,
                    "point_delta": point_delta,
                    "CI_low": float(np.percentile(vals, 2.5)),
                    "CI_high": float(np.percentile(vals, 97.5)),
                    "P_bootstrap_two_sided": p_two_sided(vals),
                }
            )

    diffs = pd.DataFrame(diff_rows)
    diffs.to_csv(
        FINAL_DIR / "WCMC_2x2_PairedDifferences.csv",
        index=False,
    )

    # Interaction / difference-in-differences.
    did_rows = []
    for metric in ["AUROC", "AUPRC", "Accuracy", "F1", "Brier"]:
        vals = boot[f"Interaction_DiD__{metric}"].to_numpy(dtype=float)

        point_local_arch = (
            point["Local_VCG"][metric]
            - point["Local_ResNet"][metric]
        )
        point_transfer_arch = (
            point["Transfer_VCG"][metric]
            - point["Transfer_ResNet"][metric]
        )
        point_did = point_transfer_arch - point_local_arch

        did_rows.append(
            {
                "metric": metric,
                "architecture_effect_WCMC_only": point_local_arch,
                "architecture_effect_after_transfer": point_transfer_arch,
                "difference_in_differences": point_did,
                "CI_low": float(np.percentile(vals, 2.5)),
                "CI_high": float(np.percentile(vals, 97.5)),
                "P_bootstrap_two_sided": p_two_sided(vals),
            }
        )

    did = pd.DataFrame(did_rows)
    did.to_csv(
        FINAL_DIR / "WCMC_2x2_Interaction_DifferenceInDifferences.csv",
        index=False,
    )

    # Figure: AUROC + AUPRC with bootstrap CIs.
    labels = [
        "ResNet1D\nWCMC-only",
        "VCG-Geometry\nWCMC-only",
        "ResNet1D\nMIMIC→WCMC",
        "VCG-Geometry\nMIMIC→WCMC",
    ]
    keys = ["Local_ResNet", "Local_VCG", "Transfer_ResNet", "Transfer_VCG"]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))

    for ax, metric in zip(axes, ["AUROC", "AUPRC"]):
        x = np.arange(len(keys))
        yv = np.array([point[k][metric] for k in keys], dtype=float)
        lo = np.array(
            [
                np.percentile(boot[f"{k}__{metric}"], 2.5)
                for k in keys
            ],
            dtype=float,
        )
        hi = np.array(
            [
                np.percentile(boot[f"{k}__{metric}"], 97.5)
                for k in keys
            ],
            dtype=float,
        )

        ax.errorbar(
            x,
            yv,
            yerr=np.vstack([yv - lo, hi - yv]),
            fmt="o",
            capsize=4,
            linewidth=1.2,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric}: architecture × pretraining")
        ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(
        FINAL_DIR / "Figure_Reviewer_WCMC_ArchitectureTransfer_2x2.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        FINAL_DIR / "Figure_Reviewer_WCMC_ArchitectureTransfer_2x2.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    # Manuscript-ready summary.
    lines = []
    lines.append("WCMC/NYP ARCHITECTURE × PRETRAINING — FINAL 5-SEED COMPARISON")
    lines.append("=" * 100)
    lines.append(f"Common locked-test N = {n:,}")
    lines.append("")
    lines.append("PERFORMANCE")
    lines.append("-" * 100)

    for key in keys:
        r = perf.loc[perf["model_key"].eq(key)].iloc[0]
        lines.append(
            f"{r['model']}: "
            f"AUROC {r['AUROC']:.3f} ({r['AUROC_CI_low']:.3f}–{r['AUROC_CI_high']:.3f}); "
            f"AUPRC {r['AUPRC']:.3f} ({r['AUPRC_CI_low']:.3f}–{r['AUPRC_CI_high']:.3f}); "
            f"F1 {r['F1']:.3f}; Brier {r['Brier']:.3f}."
        )

    lines.append("")
    lines.append("PRIMARY REVIEWER COMPARISONS")
    lines.append("-" * 100)

    for family in [
        "Architecture effect without pretraining",
        "Architecture effect after MIMIC pretraining",
        "Transfer effect within ResNet1D",
        "Transfer effect within VCG-Geometry ResNet",
    ]:
        z = diffs[
            diffs["family"].eq(family)
            & diffs["metric"].eq("AUROC")
        ].iloc[0]

        lines.append(
            f"{family}: ΔAUROC {z['point_delta']:+.4f} "
            f"({z['CI_low']:+.4f} to {z['CI_high']:+.4f}); "
            f"bootstrap P={z['P_bootstrap_two_sided']:.4g}."
        )

    z = did[did["metric"].eq("AUROC")].iloc[0]
    lines.append(
        "Architecture × pretraining difference-in-differences (AUROC): "
        f"{z['difference_in_differences']:+.4f} "
        f"({z['CI_low']:+.4f} to {z['CI_high']:+.4f}); "
        f"bootstrap P={z['P_bootstrap_two_sided']:.4g}."
    )

    lines.append("")
    lines.append("INTERPRETATION GUIDE")
    lines.append("-" * 100)
    lines.append(
        "The WCMC-only VCG-minus-ResNet comparison isolates the incremental value of the "
        "VCG-Geometry architecture without MIMIC pretraining. The transfer-minus-local "
        "comparisons quantify the benefit of MIMIC pretraining within each architecture. "
        "The difference-in-differences tests whether the VCG architecture increment changes "
        "after pretraining. Interpret effect sizes and confidence intervals; do not choose a "
        "model based only on nominal P values."
    )

    text = "\n".join(lines)
    (FINAL_DIR / "WCMC_2x2_MANUSCRIPT_NUMBERS.txt").write_text(text)

    summary = {
        "N_common_locked_test": n,
        "seeds": SEEDS,
        "N_bootstrap": N_BOOT,
        "scratch_protocol": {
            "optimizer": "AdamW",
            "learning_rate_all_parameters": LR_SCRATCH,
            "weight_decay": WEIGHT_DECAY,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "selection_metric": "validation AUPRC",
            "threshold_selection": "validation maximum F1",
            "MIMIC_pretraining_loaded": False,
        },
        "performance": perf.to_dict(orient="records"),
        "paired_differences": diffs.to_dict(orient="records"),
        "difference_in_differences": did.to_dict(orient="records"),
    }

    with open(FINAL_DIR / "WCMC_2x2_FinalSummary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + text)
    print("\nSaved to:", FINAL_DIR)


# ==========================================================================================
# MAIN
# ==========================================================================================

if __name__ == "__main__":
    if MODE == "RUN":
        run_one_seed()
    elif MODE == "FINAL":
        final_2x2()
    else:
        raise ValueError(f"Unknown WCMCONLY_MODE={MODE}")
