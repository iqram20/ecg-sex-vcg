# ============================================================
# WCMC/NYP ECG-SEX — ResNet1D TRANSFER
# MIMIC PRETRAINING -> WCMC FINE-TUNING
#
# PRIMARY QUESTION
# ----------------
# Does the residual multi-scale morphology stem that improved
# WCMC-only TRiSM also improve the FINAL MIMIC->WCMC model?
#
# ARCHITECTURE
# ------------
# ResNet1D:
#   LeadAttention
#     -> ResidualMultiScaleConv 24->64
#     -> ResidualMultiScaleConv 64->128
#     -> ResidualMultiScaleConv 128->192
#     -> residual TCN dilations 1/2/4/8/16
#     -> 6 Transformer blocks, nhead=4
#     -> 0.5 attention pooling + 0.5 mean pooling
#     -> binary sex head
#
# IMPORTANT FAIRNESS
# ------------------
# Compared with the current TRiSM transfer pathway, this script keeps:
#
# MIMIC pretraining:
#   - exact original MIMIC ECG-sex master cohort
#   - Female=0, Male=1
#   - 85/15 stratified split, random_state=42
#   - exact v4 preprocessing
#   - exact v4 ECG augmentation
#   - AdamW, LR=3e-4
#   - weight decay=1e-4
#   - gradient clip=5
#   - cosine annealing
#   - class-weighted BCE
#   - checkpoint selected by validation AUPRC
#
# WCMC fine-tuning:
#   - exact existing locked WCMC train/validation/test CSVs
#   - full-model fine-tuning
#   - AdamW, LR=1e-5
#   - weight decay=1e-4
#   - gradient clip=5
#   - NO test-set tuning
#   - best checkpoint selected by WCMC validation AUPRC
#   - threshold selected ONLY on WCMC validation by max F1
#
# FINAL COMPARISON
# ----------------
# Paired bootstrap on identical locked WCMC test ECGs:
#   1. ResNet1D MIMIC pretrained (frozen)
#   2. ResNet1D MIMIC->WCMC fine-tuned
#   3. Current TRiSM MIMIC->WCMC fine-tuned
#   4. ResNet1D WCMC-only (if available)
#
# Resume-safe:
#   - Existing MIMIC pretraining checkpoint is reused unless forced.
#   - Existing WCMC fine-tuned checkpoint is reused unless forced.
#
# Environment variables:
#   RESNET_TRANSFER_FORCE_MIMIC=0/1
#   RESNET_TRANSFER_FORCE_WCMC=0/1
#   RESNET_TRANSFER_MIMIC_EPOCHS=30
#   RESNET_TRANSFER_WCMC_EPOCHS=25
#   RESNET_TRANSFER_N_BOOT=2000
#
# Female = 0
# Male   = 1
#
# Standalone 1-cell script.
# ============================================================

import os
import gc
import json
import random
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    brier_score_loss,
    matthews_corrcoef,
    precision_recall_curve,
)

warnings.filterwarnings("ignore")

try:
    import wfdb
    WFDB_AVAILABLE = True
except Exception:
    WFDB_AVAILABLE = False


# PRIMARY QUESTION
# ----------------
# Does ResNet1D still outperform TRiSM when ResNet1D receives the
# SAME large-scale MIMIC pretraining and SAME WCMC full-model
# fine-tuning protocol used for the final TRiSM transfer model?
#
# This is a direct transfer-learning architecture comparison.
#
# ============================================================
# CONFIG
# ============================================================

SEED = 42

TARGET_LEN = 2500
USE_DERIVATIVE = True

IN_CH = 24
D_MODEL = 192
DEPTH = 6
DROPOUT = 0.08

# ---------------- MIMIC pretraining ----------------
MIMIC_VAL_SIZE = 0.15

MIMIC_EPOCHS = int(
    os.environ.get(
        "RESNET_TRANSFER_MIMIC_EPOCHS",
        "30",
    )
)

MIMIC_PATIENCE = int(
    os.environ.get(
        "RESNET_TRANSFER_MIMIC_PATIENCE",
        "7",
    )
)

MIMIC_BATCH_SIZE = int(
    os.environ.get(
        "RESNET_TRANSFER_MIMIC_BATCH_SIZE",
        "64",
    )
)

LR_MIMIC = float(
    os.environ.get(
        "RESNET_TRANSFER_LR_MIMIC",
        "3e-4",
    )
)

# ---------------- WCMC fine-tuning ----------------
WCMC_EPOCHS = int(
    os.environ.get(
        "RESNET_TRANSFER_WCMC_EPOCHS",
        "25",
    )
)

WCMC_PATIENCE = int(
    os.environ.get(
        "RESNET_TRANSFER_WCMC_PATIENCE",
        "5",
    )
)

WCMC_BATCH_SIZE = int(
    os.environ.get(
        "RESNET_TRANSFER_WCMC_BATCH_SIZE",
        "128",
    )
)

LR_WCMC = float(
    os.environ.get(
        "RESNET_TRANSFER_LR_WCMC",
        "1e-5",
    )
)

NUM_WORKERS = int(
    os.environ.get(
        "RESNET_TRANSFER_NUM_WORKERS",
        "8",
    )
)

WEIGHT_DECAY = 1e-4
GRAD_CLIP = 5.0

N_BOOT = int(
    os.environ.get(
        "RESNET_TRANSFER_N_BOOT",
        "2000",
    )
)

FORCE_MIMIC = (
    os.environ.get(
        "RESNET_TRANSFER_FORCE_MIMIC",
        "0",
    )
    ==
    "1"
)

FORCE_WCMC = (
    os.environ.get(
        "RESNET_TRANSFER_FORCE_WCMC",
        "0",
    )
    ==
    "1"
)


# ============================================================
# PATHS — ORIGINAL MIMIC + CURRENT LOCKED WCMC
# ============================================================

MIMIC_CSV = Path(
    "/athena/madelab/scratch/iqh4001/ECG/ECG_mimic_IV/"
    "Data/Output/master_ecg_full_dataset_20260228_155252.csv"
)

MIMIC_ECG_ROOT = Path(
    "/athena/madelab/scratch/iqh4001/ECG/ECG_mimic_IV/Data/"
    "mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/files"
)

MIMIC_PATH_COL = "path"
MIMIC_PATIENT_COL = "subject_id"
MIMIC_SEX_COL = "gender"

RUN_DIR = Path(
    "/athena/madelab/scratch/iqh4001/ECG/ECG_Echo/Runs/"
    "Master_ECG_Sex_MIMIC_EchoNext/"
    "20260609_020154_MASTER_ECG_SEX_MIMIC_ECHONEXT"
)

WCMC_ROOT = (
    RUN_DIR
    / "wcmc_ecgsex_from_scratch"
)

WCMC_SPLIT_DIR = (
    WCMC_ROOT
    / "04_wcmc_finetune_from_mimic"
    / "splits"
)

WCMC_TRAIN_CSV = (
    WCMC_SPLIT_DIR
    / "WCMC_ECGSex_train.csv"
)

WCMC_VAL_CSV = (
    WCMC_SPLIT_DIR
    / "WCMC_ECGSex_validation.csv"
)

WCMC_TEST_CSV = (
    WCMC_SPLIT_DIR
    / "WCMC_ECGSex_test.csv"
)

# Existing validated primary model for paired comparison.
CURRENT_TRISM_TRANSFER_PRED = (
    WCMC_ROOT
    / "04_wcmc_finetune_from_mimic"
    / "predictions"
    / "WCMC_ECGSex_finetuned_locked_test_predictions.csv"
)

CURRENT_TRISM_TRANSFER_THRESHOLD = 0.385254

# Current strongest WCMC-only ResNet1D result.
RESNET_LOCAL_PRED = (
    WCMC_ROOT
    / "07_architecture_baselines"
    / "predictions"
    / "ResNet1D_WCMCOnly_test_predictions.csv"
)

RESNET_LOCAL_THRESHOLD_JSON = (
    WCMC_ROOT
    / "07_architecture_baselines"
    / "predictions"
    / "ResNet1D_WCMCOnly_threshold.json"
)

# ---------------- Output ----------------
OUT_DIR = (
    WCMC_ROOT
    / "14_resnet1d_mimic_to_wcmc"
)

SPLIT_OUT = OUT_DIR / "splits"
CKPT_DIR = OUT_DIR / "checkpoints"
PRED_DIR = OUT_DIR / "predictions"
HISTORY_DIR = OUT_DIR / "histories"
BOOT_DIR = OUT_DIR / "bootstrap"

for d in [
    OUT_DIR,
    SPLIT_OUT,
    CKPT_DIR,
    PRED_DIR,
    HISTORY_DIR,
    BOOT_DIR,
]:
    d.mkdir(
        parents=True,
        exist_ok=True,
    )

MIMIC_TRAIN_SPLIT_CSV = (
    SPLIT_OUT
    / "MIMIC_ResNet1D_train.csv"
)

MIMIC_VAL_SPLIT_CSV = (
    SPLIT_OUT
    / "MIMIC_ResNet1D_validation.csv"
)

MIMIC_CKPT = (
    CKPT_DIR
    / "best_ResNet1D_MIMICPretrain_ECGSex.pt"
)

WCMC_CKPT = (
    CKPT_DIR
    / "best_ResNet1D_MIMICPretrain_WCMCFinetune_ECGSex.pt"
)

MIMIC_HISTORY_CSV = (
    HISTORY_DIR
    / "ResNet1D_MIMICPretrain_history.csv"
)

WCMC_HISTORY_CSV = (
    HISTORY_DIR
    / "ResNet1D_MIMICtoWCMC_finetune_history.csv"
)

MIMIC_VAL_PRED_CSV = (
    PRED_DIR
    / "ResNet1D_MIMIC_validation_predictions.csv"
)

WCMC_FROZEN_TEST_PRED_CSV = (
    PRED_DIR
    / "ResNet1D_MIMICPretrain_WCMC_locked_test_predictions.csv"
)

WCMC_VAL_PRED_CSV = (
    PRED_DIR
    / "ResNet1D_MIMICtoWCMC_validation_predictions.csv"
)

WCMC_TEST_PRED_CSV = (
    PRED_DIR
    / "ResNet1D_MIMICtoWCMC_locked_test_predictions.csv"
)

PERF_CSV = (
    OUT_DIR
    / "ResNet1D_MIMICtoWCMC_Performance.csv"
)

ALIGNED_CSV = (
    PRED_DIR
    / "ResNet1D_MIMICtoWCMC_AlignedComparisonPredictions.csv"
)

BOOT_CSV = (
    BOOT_DIR
    / "ResNet1D_MIMICtoWCMC_PairedBootstrap2000.csv"
)

DIFF_CSV = (
    BOOT_DIR
    / "ResNet1D_MIMICtoWCMC_PairedDifferences.csv"
)

SUMMARY_JSON = (
    OUT_DIR
    / "ResNet1D_MIMICtoWCMC_Summary.json"
)


# ============================================================
# REPRODUCIBILITY / DEVICE
# ============================================================

def set_seed(
    seed=42,
):
    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


set_seed(
    SEED
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else
    "cpu"
)

if DEVICE.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    try:
        torch.set_float32_matmul_precision(
            "high"
        )
    except Exception:
        pass


# ============================================================
# BASIC FILE VALIDATION
# ============================================================

for p in [
    MIMIC_CSV,
    MIMIC_ECG_ROOT,
    WCMC_TRAIN_CSV,
    WCMC_VAL_CSV,
    WCMC_TEST_CSV,
]:
    if not p.exists():
        raise FileNotFoundError(
            f"Missing required input:\n{p}"
        )

if not WFDB_AVAILABLE:
    raise ImportError(
        "wfdb is required for MIMIC ECG loading."
    )


# ============================================================
# HEADER
# ============================================================

print(
    "=" * 126
)

print(
    "WCMC/NYP ECG-SEX — ResNet1D TRANSFER"
)

print(
    "MIMIC PRETRAINING -> WCMC FINE-TUNING"
)

print(
    "=" * 126
)

print(
    "Device:",
    DEVICE,
)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(
            0
        ),
    )

print(
    "\nArchitecture:"
)

print(
    "  ResNet1D = exact current ECG-sex residual CNN baseline"
)

print(
    "\nMIMIC:"
)

print(
    "  Master:",
    MIMIC_CSV,
)

print(
    "  ECG root:",
    MIMIC_ECG_ROOT,
)

print(
    f"  Max epochs : {MIMIC_EPOCHS}"
)

print(
    f"  Patience   : {MIMIC_PATIENCE}"
)

print(
    f"  LR         : {LR_MIMIC}"
)

print(
    f"  Batch      : {MIMIC_BATCH_SIZE}"
)

print(
    "\nWCMC:"
)

print(
    f"  Max epochs : {WCMC_EPOCHS}"
)

print(
    f"  Patience   : {WCMC_PATIENCE}"
)

print(
    f"  LR         : {LR_WCMC}"
)

print(
    f"  Batch      : {WCMC_BATCH_SIZE}"
)

print(
    f"\nWorkers      : {NUM_WORKERS}"
)

print(
    f"Bootstrap    : {N_BOOT}"
)

print(
    f"Force MIMIC  : {FORCE_MIMIC}"
)

print(
    f"Force WCMC   : {FORCE_WCMC}"
)

print(
    "Output:",
    OUT_DIR,
)


# ============================================================
# COMMON HELPERS
# ============================================================

def normalize_sex_value(
    x,
):
    if pd.isna(
        x
    ):
        return np.nan

    s = str(
        x
    ).strip().lower()

    if s in {
        "m",
        "male",
        "man",
        "1",
        "true",
    }:
        return 1.0

    if s in {
        "f",
        "female",
        "woman",
        "0",
        "false",
    }:
        return 0.0

    return np.nan


def clean_ecg_array(
    arr,
    target_len=TARGET_LEN,
    use_derivative=USE_DERIVATIVE,
):
    arr = np.asarray(
        arr,
        dtype=np.float32,
    )

    arr = np.squeeze(
        arr
    )

    if arr.ndim != 2:
        raise ValueError(
            f"ECG must be 2D, got {arr.shape}"
        )

    if arr.shape[
        0
    ] == 12:
        x = arr

    elif arr.shape[
        1
    ] == 12:
        x = arr.T

    else:
        raise ValueError(
            "Cannot infer ECG lead dimension "
            f"from shape {arr.shape}"
        )

    if x.shape[
        1
    ] != target_len:
        old_len = x.shape[
            1
        ]

        new_x = np.zeros(
            (
                12,
                target_len,
            ),
            dtype=np.float32,
        )

        old_grid = np.linspace(
            0,
            1,
            old_len,
        )

        new_grid = np.linspace(
            0,
            1,
            target_len,
        )

        for i in range(
            12
        ):
            new_x[
                i
            ] = np.interp(
                new_grid,
                old_grid,
                x[
                    i
                ],
            )

        x = new_x

    x = np.nan_to_num(
        x,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    mu = x.mean(
        axis=1,
        keepdims=True,
    )

    sd = x.std(
        axis=1,
        keepdims=True,
    )

    sd[
        sd
        <
        1e-5
    ] = 1.0

    x = (
        x
        -
        mu
    ) / sd

    if use_derivative:
        dx = np.diff(
            x,
            axis=1,
            prepend=x[
                :,
                :1,
            ],
        )

        x = np.concatenate(
            [
                x,
                dx,
            ],
            axis=0,
        )

    return torch.tensor(
        x,
        dtype=torch.float32,
    )


def augment_ecg_tensor(
    x,
):
    # Exact original v4 augmentation.

    if random.random() < 0.5:
        x = (
            x
            +
            torch.randn_like(
                x
            )
            *
            0.01
        )

    if random.random() < 0.3:
        x = (
            x
            *
            float(
                0.9
                +
                0.2
                *
                torch.rand(
                    1
                )
            )
        )

    if random.random() < 0.3:
        shift = random.randint(
            -30,
            30,
        )

        x = torch.roll(
            x,
            shifts=shift,
            dims=-1,
        )

    if random.random() < 0.2:
        lead = random.randint(
            0,
            11,
        )

        x[
            lead,
            :
        ] = 0.0

        if USE_DERIVATIVE:
            x[
                lead
                +
                12,
                :
            ] = 0.0

    return x


def normalize_record_path(
    p,
):
    p = str(
        p
    ).strip()

    if (
        p.endswith(
            ".hea"
        )
        or
        p.endswith(
            ".dat"
        )
    ):
        p = p[
            :-4
        ]

    p = p.lstrip(
        "./"
    )

    if p.startswith(
        "files/"
    ):
        p = p[
            len(
                "files/"
            ):
        ]

    return p


def read_mimic_wfdb(
    path_value,
):
    raw = str(
        path_value
    ).strip()

    if (
        raw.endswith(".hea")
        or
        raw.endswith(".dat")
    ):
        raw = raw[:-4]

    if Path(raw).is_absolute():
        record_path = raw

    else:
        p = normalize_record_path(
            raw
        )

        record_path = str(
            MIMIC_ECG_ROOT
            /
            p
        )

    record = wfdb.rdrecord(
        record_path
    )

    return record.p_signal


# ============================================================
# RESNET1D ARCHITECTURE — EXACT CURRENT ECG-SEX BASELINE
# ============================================================

class ResBlock1D(
    nn.Module
):

    def __init__(
        self,
        in_ch,
        out_ch,
        stride=1,
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(
                in_ch,
                out_ch,
                7,
                stride=stride,
                padding=3,
            ),
            nn.BatchNorm1d(
                out_ch
            ),
            nn.GELU(),

            nn.Conv1d(
                out_ch,
                out_ch,
                7,
                padding=3,
            ),
            nn.BatchNorm1d(
                out_ch
            ),
        )

        self.skip = nn.Identity()

        if (
            stride != 1
            or
            in_ch != out_ch
        ):
            self.skip = nn.Sequential(
                nn.Conv1d(
                    in_ch,
                    out_ch,
                    1,
                    stride=stride,
                ),
                nn.BatchNorm1d(
                    out_ch
                ),
            )

        self.act = nn.GELU()

    def forward(
        self,
        x,
    ):
        return self.act(
            self.conv(
                x
            )
            +
            self.skip(
                x
            )
        )


class ResNet1D(
    nn.Module
):

    def __init__(
        self,
        in_ch=24,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(
                in_ch,
                64,
                7,
                stride=2,
                padding=3,
            ),
            nn.BatchNorm1d(
                64
            ),
            nn.GELU(),

            ResBlock1D(
                64,
                64,
            ),

            ResBlock1D(
                64,
                128,
                stride=2,
            ),

            ResBlock1D(
                128,
                128,
            ),

            ResBlock1D(
                128,
                256,
                stride=2,
            ),

            ResBlock1D(
                256,
                256,
            ),

            nn.AdaptiveAvgPool1d(
                1
            ),
        )

        self.head = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                256,
                128,
            ),

            nn.GELU(),

            nn.Dropout(
                DROPOUT
            ),

            nn.Linear(
                128,
                1,
            ),
        )

    def forward(
        self,
        x,
        return_features=False,
    ):
        h = self.net(
            x
        )

        z = torch.flatten(
            h,
            1,
        )

        y = (
            self.head(
                h
            )
            .squeeze(
                1
            )
        )

        if return_features:
            return (
                y,
                z,
            )

        return y


def make_model():
    return ResNet1D(
        in_ch=IN_CH,
    ).to(
        DEVICE
    )


# ============================================================
# DATASETS
# ============================================================

class MIMICSexDataset(
    Dataset
):

    def __init__(
        self,
        df,
        augment=False,
    ):
        self.df = (
            df
            .reset_index(
                drop=True
            )
            .copy()
        )

        self.augment = augment

    def __len__(
        self,
    ):
        return len(
            self.df
        )

    def __getitem__(
        self,
        idx,
    ):
        row = self.df.iloc[
            idx
        ]

        try:
            arr = read_mimic_wfdb(
                row[
                    MIMIC_PATH_COL
                ]
            )

            x = clean_ecg_array(
                arr
            )

            if self.augment:
                x = augment_ecg_tensor(
                    x
                )

            if not torch.isfinite(
                x
            ).all():
                raise ValueError(
                    "Non-finite ECG."
                )

            ecg_key = (
                str(
                    row[
                        "study_id"
                    ]
                )
                if (
                    "study_id"
                    in row.index
                    and
                    pd.notna(
                        row[
                            "study_id"
                        ]
                    )
                )
                else
                str(
                    idx
                )
            )

            return {
                "x":
                    x,

                "y":
                    float(
                        row[
                            "sex_binary"
                        ]
                    ),

                "patient_key":
                    str(
                        row[
                            MIMIC_PATIENT_COL
                        ]
                    ),

                "ecg_key":
                    ecg_key,

                "path":
                    str(
                        row[
                            MIMIC_PATH_COL
                        ]
                    ),

                "error":
                    None,
            }

        except Exception as exc:
            return {
                "x":
                    None,

                "y":
                    float(
                        row[
                            "sex_binary"
                        ]
                    ),

                "patient_key":
                    str(
                        row[
                            MIMIC_PATIENT_COL
                        ]
                    ),

                "ecg_key":
                    str(
                        idx
                    ),

                "path":
                    str(
                        row[
                            MIMIC_PATH_COL
                        ]
                    ),

                "error":
                    repr(
                        exc
                    ),
            }


class WCMCSexDataset(
    Dataset
):

    def __init__(
        self,
        df,
        augment=False,
    ):
        self.df = (
            df
            .reset_index(
                drop=True
            )
            .copy()
        )

        self.augment = augment

    def __len__(
        self,
    ):
        return len(
            self.df
        )

    def __getitem__(
        self,
        idx,
    ):
        row = self.df.iloc[
            idx
        ]

        path = str(
            row[
                "raw_local_path"
            ]
        )

        try:
            arr = np.load(
                path,
                allow_pickle=False,
            )

            x = clean_ecg_array(
                arr
            )

            # Keep WCMC transfer aligned with current pipeline:
            # no additional augmentation here.

            if not torch.isfinite(
                x
            ).all():
                raise ValueError(
                    "Non-finite ECG."
                )

            return {
                "x":
                    x,

                "y":
                    float(
                        row[
                            "sex_binary"
                        ]
                    ),

                "patient_key":
                    str(
                        row[
                            "patient_key"
                        ]
                    ),

                "ecg_key":
                    str(
                        row[
                            "ecg_key"
                        ]
                    ),

                "path":
                    path,

                "error":
                    None,
            }

        except Exception as exc:
            return {
                "x":
                    None,

                "y":
                    float(
                        row[
                            "sex_binary"
                        ]
                    ),

                "patient_key":
                    str(
                        row[
                            "patient_key"
                        ]
                    ),

                "ecg_key":
                    str(
                        row[
                            "ecg_key"
                        ]
                    ),

                "path":
                    path,

                "error":
                    repr(
                        exc
                    ),
            }


def collate_fn(
    batch,
):
    valid = [
        b
        for b in batch
        if b[
            "x"
        ] is not None
    ]

    invalid = [
        b
        for b in batch
        if b[
            "x"
        ] is None
    ]

    if not valid:
        return {
            "x":
                None,

            "valid":
                [],

            "invalid":
                invalid,
        }

    return {
        "x":
            torch.stack(
                [
                    b[
                        "x"
                    ]
                    for b in valid
                ]
            ),

        "valid":
            valid,

        "invalid":
            invalid,
    }


def make_loader(
    dataset,
    batch_size,
    shuffle=False,
):
    kwargs = {
        "dataset":
            dataset,

        "batch_size":
            batch_size,

        "shuffle":
            shuffle,

        "num_workers":
            NUM_WORKERS,

        "pin_memory":
            (
                DEVICE.type
                ==
                "cuda"
            ),

        "drop_last":
            False,

        "collate_fn":
            collate_fn,
    }

    if NUM_WORKERS > 0:
        kwargs[
            "persistent_workers"
        ] = True

        kwargs[
            "prefetch_factor"
        ] = 2

    return DataLoader(
        **kwargs
    )


# ============================================================
# METRICS
# ============================================================

def safe_auc(
    y,
    p,
):
    if len(
        np.unique(
            y
        )
    ) < 2:
        return np.nan

    return float(
        roc_auc_score(
            y,
            p,
        )
    )


def safe_auprc(
    y,
    p,
):
    if len(
        np.unique(
            y
        )
    ) < 2:
        return np.nan

    return float(
        average_precision_score(
            y,
            p,
        )
    )


def select_f1_threshold(
    y_true,
    y_prob,
):
    precision, recall, thresholds = (
        precision_recall_curve(
            y_true,
            y_prob,
        )
    )

    if len(
        thresholds
    ) == 0:
        return 0.5

    precision = precision[
        :-1
    ]

    recall = recall[
        :-1
    ]

    f1 = (
        2
        *
        precision
        *
        recall
        /
        np.maximum(
            precision
            +
            recall,
            1e-12,
        )
    )

    best = int(
        np.nanargmax(
            f1
        )
    )

    return float(
        thresholds[
            best
        ]
    )


def binary_metrics(
    y_true,
    y_prob,
    threshold,
):
    y_true = np.asarray(
        y_true
    ).astype(
        int
    )

    y_prob = np.asarray(
        y_prob
    ).astype(
        float
    )

    y_pred = (
        y_prob
        >=
        threshold
    ).astype(
        int
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[
            0,
            1,
        ],
    ).ravel()

    return {
        "N":
            int(
                len(
                    y_true
                )
            ),

        "Female_N":
            int(
                (
                    y_true
                    ==
                    0
                ).sum()
            ),

        "Male_N":
            int(
                (
                    y_true
                    ==
                    1
                ).sum()
            ),

        "threshold":
            float(
                threshold
            ),

        "AUROC":
            safe_auc(
                y_true,
                y_prob,
            ),

        "AUPRC":
            safe_auprc(
                y_true,
                y_prob,
            ),

        "Accuracy":
            float(
                accuracy_score(
                    y_true,
                    y_pred,
                )
            ),

        "Balanced_Accuracy":
            float(
                balanced_accuracy_score(
                    y_true,
                    y_pred,
                )
            ),

        "Precision":
            float(
                precision_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),

        "Sensitivity":
            float(
                recall_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),

        "Specificity":
            float(
                tn
                /
                max(
                    tn
                    +
                    fp,
                    1,
                )
            ),

        "F1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),

        "MCC":
            float(
                matthews_corrcoef(
                    y_true,
                    y_pred,
                )
            ),

        "Brier":
            float(
                brier_score_loss(
                    y_true,
                    y_prob,
                )
            ),

        "TN":
            int(
                tn
            ),

        "FP":
            int(
                fp
            ),

        "FN":
            int(
                fn
            ),

        "TP":
            int(
                tp
            ),
    }


# ============================================================
# TRAIN / PREDICT HELPERS
# ============================================================

def batch_targets(
    valid,
):
    return torch.tensor(
        [
            float(
                v[
                    "y"
                ]
            )
            for v in valid
        ],
        dtype=torch.float32,
        device=DEVICE,
    )


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler,
):
    model.train()

    total_loss = 0.0
    total_n = 0
    failures = 0

    train_true = []
    train_prob = []

    for batch in loader:
        failures += len(
            batch[
                "invalid"
            ]
        )

        if batch[
            "x"
        ] is None:
            continue

        x = batch[
            "x"
        ].to(
            DEVICE,
            non_blocking=True,
        )

        y = batch_targets(
            batch[
                "valid"
            ]
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with autocast(
            enabled=(
                DEVICE.type
                ==
                "cuda"
            )
        ):
            logits = model(
                x
            )

            loss = criterion(
                logits,
                y,
            )

        scaler.scale(
            loss
        ).backward()

        scaler.unscale_(
            optimizer
        )

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP,
        )

        scaler.step(
            optimizer
        )

        scaler.update()

        bs = int(
            y.numel()
        )

        total_loss += (
            float(
                loss.detach().cpu()
            )
            *
            bs
        )

        total_n += bs

        train_true.append(
            y.detach().cpu().numpy()
        )

        train_prob.append(
            torch.sigmoid(
                logits
            )
            .detach()
            .cpu()
            .numpy()
        )

    if train_true:
        train_true = np.concatenate(
            train_true
        ).astype(
            int
        )

        train_prob = np.concatenate(
            train_prob
        ).astype(
            float
        )

        train_auc = safe_auc(
            train_true,
            train_prob,
        )

        train_ap = safe_auprc(
            train_true,
            train_prob,
        )

    else:
        train_auc = np.nan
        train_ap = np.nan

    return {
        "loss":
            total_loss
            /
            max(
                total_n,
                1,
            ),

        "N":
            int(
                total_n
            ),

        "failures":
            int(
                failures
            ),

        "AUROC":
            train_auc,

        "AUPRC":
            train_ap,
    }


@torch.no_grad()
def predict_loader(
    model,
    loader,
):
    model.eval()

    rows = []
    failures = []

    for batch in loader:
        for bad in batch[
            "invalid"
        ]:
            failures.append(
                {
                    "patient_key":
                        bad[
                            "patient_key"
                        ],

                    "ecg_key":
                        bad[
                            "ecg_key"
                        ],

                    "path":
                        bad[
                            "path"
                        ],

                    "error":
                        bad[
                            "error"
                        ],
                }
            )

        if batch[
            "x"
        ] is None:
            continue

        x = batch[
            "x"
        ].to(
            DEVICE,
            non_blocking=True,
        )

        with autocast(
            enabled=(
                DEVICE.type
                ==
                "cuda"
            )
        ):
            logits = model(
                x
            )

        probs = (
            torch.sigmoid(
                logits
            )
            .detach()
            .cpu()
            .numpy()
        )

        logits_np = (
            logits
            .detach()
            .cpu()
            .numpy()
        )

        for item, logit, prob in zip(
            batch[
                "valid"
            ],
            logits_np,
            probs,
        ):
            rows.append(
                {
                    "patient_key":
                        str(
                            item[
                                "patient_key"
                            ]
                        ),

                    "ecg_key":
                        str(
                            item[
                                "ecg_key"
                            ]
                        ),

                    "sex_true_binary":
                        int(
                            item[
                                "y"
                            ]
                        ),

                    "logit_male":
                        float(
                            logit
                        ),

                    "prob_male":
                        float(
                            prob
                        ),

                    "path":
                        str(
                            item[
                                "path"
                            ]
                        ),
                }
            )

    return (
        pd.DataFrame(
            rows
        ),
        pd.DataFrame(
            failures
        ),
    )


def load_model_checkpoint(
    path,
):
    ckpt = torch.load(
        path,
        map_location=DEVICE,
        weights_only=False,
    )

    state = (
        ckpt[
            "model_state_dict"
        ]
        if (
            isinstance(
                ckpt,
                dict
            )
            and
            "model_state_dict"
            in ckpt
        )
        else
        ckpt
    )

    if any(
        str(
            k
        ).startswith(
            "module."
        )
        for k in state
    ):
        state = {
            str(
                k
            ).replace(
                "module.",
                "",
                1,
            ):
                v
            for k, v in state.items()
        }

    model = make_model()

    incompatible = model.load_state_dict(
        state,
        strict=True,
    )

    if (
        incompatible.missing_keys
        or
        incompatible.unexpected_keys
    ):
        raise RuntimeError(
            "Checkpoint mismatch:\n"
            f"missing={incompatible.missing_keys}\n"
            f"unexpected={incompatible.unexpected_keys}"
        )

    return (
        model,
        ckpt,
    )


# ============================================================
# PREPARE MIMIC METADATA + EXACT SPLIT
# ============================================================

print(
    "\n"
    +
    "=" * 126
)

print(
    "PREPARING MIMIC ECG-SEX COHORT"
)

print(
    "=" * 126
)

mimic = pd.read_csv(
    MIMIC_CSV,
    low_memory=False,
)

required_mimic = [
    MIMIC_PATH_COL,
    MIMIC_PATIENT_COL,
    MIMIC_SEX_COL,
]

missing = [
    c
    for c in required_mimic
    if c not in mimic.columns
]

if missing:
    raise RuntimeError(
        f"MIMIC missing required columns: {missing}"
    )

mimic[
    "sex_binary"
] = mimic[
    MIMIC_SEX_COL
].apply(
    normalize_sex_value
)

mimic = mimic.dropna(
    subset=[
        MIMIC_PATH_COL,
        MIMIC_PATIENT_COL,
        "sex_binary",
    ]
).copy()

mimic[
    "sex_binary"
] = mimic[
    "sex_binary"
].astype(
    int
)

mimic[
    MIMIC_PATIENT_COL
] = mimic[
    MIMIC_PATIENT_COL
].astype(
    str
)

mimic = mimic.reset_index(
    drop=True
)

print(
    "MIMIC usable ECGs:",
    f"{len(mimic):,}"
)

print(
    "MIMIC unique patients:",
    f"{mimic[MIMIC_PATIENT_COL].nunique():,}"
)

print(
    "MIMIC sex counts:"
)

print(
    mimic[
        "sex_binary"
    ].value_counts().sort_index()
)

# The original ECG-sex run had one ECG per patient.
if (
    mimic[
        MIMIC_PATIENT_COL
    ].nunique()
    !=
    len(
        mimic
    )
):
    raise RuntimeError(
        "Expected one ECG per patient in the original MIMIC ECG-sex "
        "cohort, but duplicates were found. Stop rather than silently "
        "changing the source cohort."
    )


if (
    MIMIC_TRAIN_SPLIT_CSV.exists()
    and
    MIMIC_VAL_SPLIT_CSV.exists()
    and
    not FORCE_MIMIC
):
    mimic_train = pd.read_csv(
        MIMIC_TRAIN_SPLIT_CSV,
        low_memory=False,
    )

    mimic_val = pd.read_csv(
        MIMIC_VAL_SPLIT_CSV,
        low_memory=False,
    )

    print(
        "\nReusing saved MIMIC split."
    )

else:
    mimic_train, mimic_val = train_test_split(
        mimic,
        test_size=MIMIC_VAL_SIZE,
        random_state=SEED,
        stratify=mimic[
            "sex_binary"
        ],
    )

    mimic_train = mimic_train.reset_index(
        drop=True
    )

    mimic_val = mimic_val.reset_index(
        drop=True
    )

    mimic_train.to_csv(
        MIMIC_TRAIN_SPLIT_CSV,
        index=False,
    )

    mimic_val.to_csv(
        MIMIC_VAL_SPLIT_CSV,
        index=False,
    )


print(
    "\nMIMIC split:"
)

print(
    f"Train      N={len(mimic_train):,} | "
    f"Female={(mimic_train['sex_binary'] == 0).sum():,} | "
    f"Male={(mimic_train['sex_binary'] == 1).sum():,}"
)

print(
    f"Validation N={len(mimic_val):,} | "
    f"Female={(mimic_val['sex_binary'] == 0).sum():,} | "
    f"Male={(mimic_val['sex_binary'] == 1).sum():,}"
)

if set(
    mimic_train[
        MIMIC_PATIENT_COL
    ]
) & set(
    mimic_val[
        MIMIC_PATIENT_COL
    ]
):
    raise RuntimeError(
        "MIMIC patient leakage detected."
    )


# ============================================================
# MIMIC LOADERS
# ============================================================

mimic_train_loader = make_loader(
    MIMICSexDataset(
        mimic_train,
        augment=True,
    ),
    batch_size=MIMIC_BATCH_SIZE,
    shuffle=True,
)

mimic_val_loader = make_loader(
    MIMICSexDataset(
        mimic_val,
        augment=False,
    ),
    batch_size=MIMIC_BATCH_SIZE,
    shuffle=False,
)


# ============================================================
# STAGE 1 — MIMIC PRETRAINING
# ============================================================

if (
    MIMIC_CKPT.exists()
    and
    not FORCE_MIMIC
):
    print(
        "\n"
        +
        "=" * 126
    )

    print(
        "STAGE 1 — REUSING EXISTING MIMIC ResNet1D CHECKPOINT"
    )

    print(
        "=" * 126
    )

    print(
        MIMIC_CKPT
    )

else:
    print(
        "\n"
        +
        "=" * 126
    )

    print(
        "STAGE 1 — MIMIC PRETRAINING ResNet1D"
    )

    print(
        "=" * 126
    )

    set_seed(
        SEED
    )

    model = make_model()

    n_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "Parameters:",
        f"{n_params:,}"
    )

    n_female_mimic = int(
        (
            mimic_train[
                "sex_binary"
            ]
            ==
            0
        ).sum()
    )

    n_male_mimic = int(
        (
            mimic_train[
                "sex_binary"
            ]
            ==
            1
        ).sum()
    )

    mimic_pos_weight = (
        n_female_mimic
        /
        max(
            n_male_mimic,
            1,
        )
    )

    print(
        "MIMIC pos_weight male:",
        f"{mimic_pos_weight:.6f}"
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [
                mimic_pos_weight
            ],
            dtype=torch.float32,
            device=DEVICE,
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR_MIMIC,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(
            1,
            MIMIC_EPOCHS,
        ),
        eta_min=LR_MIMIC
        *
        0.05,
    )

    scaler = GradScaler(
        enabled=(
            DEVICE.type
            ==
            "cuda"
        )
    )

    best_val_auprc = -np.inf
    best_epoch = -1
    no_improve = 0

    history = []

    for epoch in range(
        1,
        MIMIC_EPOCHS
        +
        1,
    ):
        t0 = datetime.now()

        tr = train_one_epoch(
            model,
            mimic_train_loader,
            optimizer,
            criterion,
            scaler,
        )

        val_pred, val_fail = predict_loader(
            model,
            mimic_val_loader,
        )

        yv = val_pred[
            "sex_true_binary"
        ].to_numpy(
            dtype=int
        )

        pv = val_pred[
            "prob_male"
        ].to_numpy(
            dtype=float
        )

        val_auc = safe_auc(
            yv,
            pv,
        )

        val_ap = safe_auprc(
            yv,
            pv,
        )

        val_thr = select_f1_threshold(
            yv,
            pv,
        )

        val_f1 = f1_score(
            yv,
            (
                pv
                >=
                val_thr
            ).astype(
                int
            ),
            zero_division=0,
        )

        sec = (
            datetime.now()
            -
            t0
        ).total_seconds()

        row = {
            "epoch":
                epoch,

            "lr":
                float(
                    optimizer.param_groups[
                        0
                    ][
                        "lr"
                    ]
                ),

            "train_loss":
                tr[
                    "loss"
                ],

            "train_AUROC":
                tr[
                    "AUROC"
                ],

            "train_AUPRC":
                tr[
                    "AUPRC"
                ],

            "train_N":
                tr[
                    "N"
                ],

            "train_failures":
                tr[
                    "failures"
                ],

            "val_N":
                len(
                    val_pred
                ),

            "val_failures":
                len(
                    val_fail
                ),

            "val_AUROC":
                val_auc,

            "val_AUPRC":
                val_ap,

            "val_F1":
                float(
                    val_f1
                ),

            "val_threshold_F1":
                float(
                    val_thr
                ),

            "epoch_time_sec":
                sec,
        }

        history.append(
            row
        )

        pd.DataFrame(
            history
        ).to_csv(
            MIMIC_HISTORY_CSV,
            index=False,
        )

        print(
            f"MIMIC | Epoch {epoch:02d}/{MIMIC_EPOCHS:02d} | "
            f"LR {row['lr']:.6g} | "
            f"Loss {tr['loss']:.5f} | "
            f"Train AUC {tr['AUROC']:.5f} | "
            f"Val AUC {val_auc:.5f} | "
            f"Val AUPRC {val_ap:.5f} | "
            f"Val F1 {val_f1:.5f} | "
            f"{sec:.1f}s"
        )

        if (
            np.isfinite(
                val_ap
            )
            and
            val_ap
            >
            best_val_auprc
            +
            1e-6
        ):
            best_val_auprc = float(
                val_ap
            )

            best_epoch = int(
                epoch
            )

            no_improve = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "epoch":
                        best_epoch,

                    "best_val_AUPRC":
                        best_val_auprc,

                    "val_AUROC":
                        float(
                            val_auc
                        ),

                    "val_threshold_F1":
                        float(
                            val_thr
                        ),

                    "model_name":
                        "ResNet1D_MIMICPretrain_ECGSex",

                    "architecture":
                        "Residual multi-scale CNN-TCN-Transformer",

                    "config": {
                        "seed":
                            SEED,

                        "target_len":
                            TARGET_LEN,

                        "in_ch":
                            IN_CH,

                        "d_model":
                            D_MODEL,

                        "depth":
                            DEPTH,

                        "dropout":
                            DROPOUT,

                        "lr":
                            LR_MIMIC,

                        "weight_decay":
                            WEIGHT_DECAY,

                        "mimic_val_size":
                            MIMIC_VAL_SIZE,

                        "pos_weight":
                            mimic_pos_weight,
                    },
                },
                MIMIC_CKPT,
            )

            print(
                f"  ✓ BEST MIMIC: epoch={epoch}, "
                f"val AUPRC={val_ap:.6f}"
            )

        else:
            no_improve += 1

            print(
                f"  no improvement "
                f"({no_improve}/{MIMIC_PATIENCE})"
            )

        scheduler.step()

        if no_improve >= MIMIC_PATIENCE:
            print(
                f"\nMIMIC early stopping at epoch {epoch}; "
                f"best epoch={best_epoch}, "
                f"best val AUPRC={best_val_auprc:.6f}"
            )

            break

    del model

    gc.collect()

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


# ============================================================
# RECOVER MIMIC VALIDATION THRESHOLD
# ============================================================

mimic_model, mimic_ckpt = load_model_checkpoint(
    MIMIC_CKPT
)

mimic_val_pred, mimic_val_fail = predict_loader(
    mimic_model,
    mimic_val_loader,
)

MIMIC_VAL_THRESHOLD = select_f1_threshold(
    mimic_val_pred[
        "sex_true_binary"
    ],
    mimic_val_pred[
        "prob_male"
    ],
)

mimic_val_pred[
    "threshold"
] = MIMIC_VAL_THRESHOLD

mimic_val_pred.to_csv(
    MIMIC_VAL_PRED_CSV,
    index=False,
)

print(
    "\nMIMIC best epoch:",
    mimic_ckpt.get(
        "epoch",
        "unknown",
    )
)

print(
    "MIMIC best validation AUPRC:",
    mimic_ckpt.get(
        "best_val_AUPRC",
        "unknown",
    )
)

print(
    "MIMIC validation-selected F1 threshold:",
    f"{MIMIC_VAL_THRESHOLD:.6f}"
)


# ============================================================
# LOAD CURRENT LOCKED WCMC SPLITS
# ============================================================

def load_wcmc_split(
    path,
    name,
):
    df = pd.read_csv(
        path,
        low_memory=False,
    )

    required = [
        "patient_key",
        "ecg_key",
        "sex_binary",
        "raw_local_path",
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"{name}: missing {missing}"
        )

    df[
        "patient_key"
    ] = df[
        "patient_key"
    ].astype(
        str
    )

    df[
        "ecg_key"
    ] = df[
        "ecg_key"
    ].astype(
        str
    )

    df[
        "sex_binary"
    ] = pd.to_numeric(
        df[
            "sex_binary"
        ],
        errors="raise",
    ).astype(
        int
    )

    return (
        df
        .reset_index(
            drop=True
        )
        .copy()
    )


wcmc_train = load_wcmc_split(
    WCMC_TRAIN_CSV,
    "train",
)

wcmc_val = load_wcmc_split(
    WCMC_VAL_CSV,
    "validation",
)

wcmc_test = load_wcmc_split(
    WCMC_TEST_CSV,
    "test",
)


print(
    "\n"
    +
    "=" * 126
)

print(
    "LOCKED WCMC SPLITS"
)

print(
    "=" * 126
)

for name, df in [
    (
        "Train",
        wcmc_train,
    ),
    (
        "Validation",
        wcmc_val,
    ),
    (
        "Test",
        wcmc_test,
    ),
]:
    print(
        f"{name:10s} "
        f"N={len(df):,} | "
        f"Female={(df['sex_binary'] == 0).sum():,} | "
        f"Male={(df['sex_binary'] == 1).sum():,}"
    )


if (
    set(
        wcmc_train[
            "patient_key"
        ]
    )
    &
    set(
        wcmc_val[
            "patient_key"
        ]
    )
    or
    set(
        wcmc_train[
            "patient_key"
        ]
    )
    &
    set(
        wcmc_test[
            "patient_key"
        ]
    )
    or
    set(
        wcmc_val[
            "patient_key"
        ]
    )
    &
    set(
        wcmc_test[
            "patient_key"
        ]
    )
):
    raise RuntimeError(
        "Patient leakage in WCMC splits."
    )


wcmc_train_loader = make_loader(
    WCMCSexDataset(
        wcmc_train,
        augment=False,
    ),
    batch_size=WCMC_BATCH_SIZE,
    shuffle=True,
)

wcmc_val_loader = make_loader(
    WCMCSexDataset(
        wcmc_val,
        augment=False,
    ),
    batch_size=WCMC_BATCH_SIZE,
    shuffle=False,
)

wcmc_test_loader = make_loader(
    WCMCSexDataset(
        wcmc_test,
        augment=False,
    ),
    batch_size=WCMC_BATCH_SIZE,
    shuffle=False,
)


# ============================================================
# FROZEN MIMIC ResNet1D ON LOCKED WCMC TEST
# ============================================================

print(
    "\n"
    +
    "=" * 126
)

print(
    "MIMIC-PRETRAINED ResNet1D — FROZEN WCMC TEST"
)

print(
    "=" * 126
)

frozen_test_pred, frozen_test_fail = predict_loader(
    mimic_model,
    wcmc_test_loader,
)

if len(
    frozen_test_pred
) != len(
    wcmc_test
):
    raise RuntimeError(
        f"Frozen WCMC inference N mismatch: "
        f"{len(frozen_test_pred):,} vs {len(wcmc_test):,}"
    )

frozen_test_pred.to_csv(
    WCMC_FROZEN_TEST_PRED_CSV,
    index=False,
)

frozen_metrics = binary_metrics(
    frozen_test_pred[
        "sex_true_binary"
    ],
    frozen_test_pred[
        "prob_male"
    ],
    MIMIC_VAL_THRESHOLD,
)

print(
    pd.DataFrame(
        [
            {
                "Model":
                    "ResNet1D MIMIC pretrained (frozen)",

                **frozen_metrics,
            }
        ]
    ).to_string(
        index=False
    )
)

del mimic_model

gc.collect()

if DEVICE.type == "cuda":
    torch.cuda.empty_cache()


# ============================================================
# STAGE 2 — WCMC FINE-TUNING
# ============================================================

if (
    WCMC_CKPT.exists()
    and
    not FORCE_WCMC
):
    print(
        "\n"
        +
        "=" * 126
    )

    print(
        "STAGE 2 — REUSING EXISTING MIMIC->WCMC ResNet1D CHECKPOINT"
    )

    print(
        "=" * 126
    )

    print(
        WCMC_CKPT
    )

else:
    print(
        "\n"
        +
        "=" * 126
    )

    print(
        "STAGE 2 — WCMC FINE-TUNING"
    )

    print(
        "=" * 126
    )

    model, _ = load_model_checkpoint(
        MIMIC_CKPT
    )

    # Exact current WCMC transfer recipe:
    # unweighted BCE, full-model AdamW LR=1e-5.
    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR_WCMC,
        weight_decay=WEIGHT_DECAY,
    )

    scaler = GradScaler(
        enabled=(
            DEVICE.type
            ==
            "cuda"
        )
    )

    best_val_auprc = -np.inf
    best_epoch = -1
    no_improve = 0

    history = []

    for epoch in range(
        1,
        WCMC_EPOCHS
        +
        1,
    ):
        t0 = datetime.now()

        tr = train_one_epoch(
            model,
            wcmc_train_loader,
            optimizer,
            criterion,
            scaler,
        )

        val_pred, val_fail = predict_loader(
            model,
            wcmc_val_loader,
        )

        yv = val_pred[
            "sex_true_binary"
        ].to_numpy(
            dtype=int
        )

        pv = val_pred[
            "prob_male"
        ].to_numpy(
            dtype=float
        )

        val_auc = safe_auc(
            yv,
            pv,
        )

        val_ap = safe_auprc(
            yv,
            pv,
        )

        val_thr = select_f1_threshold(
            yv,
            pv,
        )

        val_m = binary_metrics(
            yv,
            pv,
            val_thr,
        )

        sec = (
            datetime.now()
            -
            t0
        ).total_seconds()

        row = {
            "epoch":
                epoch,

            "lr":
                float(
                    optimizer.param_groups[
                        0
                    ][
                        "lr"
                    ]
                ),

            "train_loss":
                tr[
                    "loss"
                ],

            "train_AUROC":
                tr[
                    "AUROC"
                ],

            "train_AUPRC":
                tr[
                    "AUPRC"
                ],

            "train_N":
                tr[
                    "N"
                ],

            "train_failures":
                tr[
                    "failures"
                ],

            "val_N":
                len(
                    val_pred
                ),

            "val_failures":
                len(
                    val_fail
                ),

            "val_AUROC":
                val_auc,

            "val_AUPRC":
                val_ap,

            "val_F1":
                val_m[
                    "F1"
                ],

            "val_threshold_F1":
                val_thr,

            "epoch_time_sec":
                sec,
        }

        history.append(
            row
        )

        pd.DataFrame(
            history
        ).to_csv(
            WCMC_HISTORY_CSV,
            index=False,
        )

        print(
            f"WCMC | Epoch {epoch:02d}/{WCMC_EPOCHS:02d} | "
            f"Loss {tr['loss']:.5f} | "
            f"Train AUC {tr['AUROC']:.5f} | "
            f"Val AUC {val_auc:.5f} | "
            f"Val AUPRC {val_ap:.5f} | "
            f"Val F1 {val_m['F1']:.5f} | "
            f"thr {val_thr:.6f} | "
            f"{sec:.1f}s"
        )

        if (
            np.isfinite(
                val_ap
            )
            and
            val_ap
            >
            best_val_auprc
            +
            1e-6
        ):
            best_val_auprc = float(
                val_ap
            )

            best_epoch = int(
                epoch
            )

            no_improve = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "epoch":
                        best_epoch,

                    "best_val_AUPRC":
                        best_val_auprc,

                    "val_AUROC":
                        float(
                            val_auc
                        ),

                    "val_threshold_F1":
                        float(
                            val_thr
                        ),

                    "initialization_checkpoint":
                        str(
                            MIMIC_CKPT
                        ),

                    "model_name":
                        "ResNet1D_MIMICPretrain_WCMCFinetune_ECGSex",

                    "architecture":
                        "Residual multi-scale CNN-TCN-Transformer",

                    "config": {
                        "seed":
                            SEED,

                        "target_len":
                            TARGET_LEN,

                        "in_ch":
                            IN_CH,

                        "d_model":
                            D_MODEL,

                        "depth":
                            DEPTH,

                        "dropout":
                            DROPOUT,

                        "lr":
                            LR_WCMC,

                        "weight_decay":
                            WEIGHT_DECAY,

                        "batch_size":
                            WCMC_BATCH_SIZE,

                        "selection":
                            "validation AUPRC",

                        "threshold":
                            "validation max F1",
                    },
                },
                WCMC_CKPT,
            )

            print(
                f"  ✓ BEST WCMC: epoch={epoch}, "
                f"val AUPRC={val_ap:.6f}"
            )

        else:
            no_improve += 1

            print(
                f"  no improvement "
                f"({no_improve}/{WCMC_PATIENCE})"
            )

        if no_improve >= WCMC_PATIENCE:
            print(
                f"\nWCMC early stopping at epoch {epoch}; "
                f"best epoch={best_epoch}, "
                f"best val AUPRC={best_val_auprc:.6f}"
            )

            break

    del model

    gc.collect()

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


# ============================================================
# FINAL VALIDATION THRESHOLD + LOCKED TEST
# ============================================================

best_model, best_wcmc_ckpt = load_model_checkpoint(
    WCMC_CKPT
)

final_val_pred, final_val_fail = predict_loader(
    best_model,
    wcmc_val_loader,
)

FINAL_THRESHOLD = select_f1_threshold(
    final_val_pred[
        "sex_true_binary"
    ],
    final_val_pred[
        "prob_male"
    ],
)

final_val_pred[
    "threshold"
] = FINAL_THRESHOLD

final_val_pred.to_csv(
    WCMC_VAL_PRED_CSV,
    index=False,
)

print(
    "\nFinal WCMC validation threshold:",
    f"{FINAL_THRESHOLD:.6f}"
)

print(
    "Best WCMC epoch:",
    best_wcmc_ckpt.get(
        "epoch",
        "unknown",
    )
)

print(
    "Best WCMC validation AUPRC:",
    best_wcmc_ckpt.get(
        "best_val_AUPRC",
        "unknown",
    )
)


print(
    "\n"
    +
    "=" * 126
)

print(
    "FINAL UNTOUCHED LOCKED WCMC TEST"
)

print(
    "=" * 126
)

final_test_pred, final_test_fail = predict_loader(
    best_model,
    wcmc_test_loader,
)

if len(
    final_test_pred
) != len(
    wcmc_test
):
    raise RuntimeError(
        f"Final WCMC inference N mismatch: "
        f"{len(final_test_pred):,} vs {len(wcmc_test):,}"
    )

final_test_pred[
    "threshold"
] = FINAL_THRESHOLD

final_test_pred[
    "sex_pred_binary"
] = (
    final_test_pred[
        "prob_male"
    ]
    >=
    FINAL_THRESHOLD
).astype(
    int
)

final_test_pred[
    "correct"
] = (
    final_test_pred[
        "sex_pred_binary"
    ]
    ==
    final_test_pred[
        "sex_true_binary"
    ]
).astype(
    int
)

final_test_pred.to_csv(
    WCMC_TEST_PRED_CSV,
    index=False,
)

final_metrics = binary_metrics(
    final_test_pred[
        "sex_true_binary"
    ],
    final_test_pred[
        "prob_male"
    ],
    FINAL_THRESHOLD,
)

print(
    pd.DataFrame(
        [
            {
                "Model":
                    "ResNet1D MIMIC->WCMC",

                **final_metrics,
            }
        ]
    ).to_string(
        index=False
    )
)


# ============================================================
# ALIGN WITH EXISTING CURRENT-RUN PREDICTIONS
# ============================================================

def normalize_prediction_file(
    path,
    prob_name,
):
    if not path.exists():
        return None

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    required = [
        "patient_key",
        "ecg_key",
        "sex_true_binary",
        "prob_male",
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"{path} missing {missing}"
        )

    df = df[
        required
    ].copy()

    df[
        "patient_key"
    ] = df[
        "patient_key"
    ].astype(
        str
    )

    df[
        "ecg_key"
    ] = df[
        "ecg_key"
    ].astype(
        str
    )

    df[
        "sex_true_binary"
    ] = pd.to_numeric(
        df[
            "sex_true_binary"
        ],
        errors="raise",
    ).astype(
        int
    )

    df[
        "prob_male"
    ] = pd.to_numeric(
        df[
            "prob_male"
        ],
        errors="raise",
    )

    return df.rename(
        columns={
            "prob_male":
                prob_name
        }
    )


aligned = final_test_pred[
    [
        "patient_key",
        "ecg_key",
        "sex_true_binary",
        "prob_male",
    ]
].copy()

aligned[
    "patient_key"
] = aligned[
    "patient_key"
].astype(
    str
)

aligned[
    "ecg_key"
] = aligned[
    "ecg_key"
].astype(
    str
)

aligned = aligned.rename(
    columns={
        "prob_male":
            "ResNet_MIMICtoWCMC"
    }
)

frozen_ref = normalize_prediction_file(
    WCMC_FROZEN_TEST_PRED_CSV,
    "ResNet_MIMIC_frozen",
)

current_transfer_ref = normalize_prediction_file(
    CURRENT_TRISM_TRANSFER_PRED,
    "TRiSM_MIMICtoWCMC",
)

local_ref = normalize_prediction_file(
    RESNET_LOCAL_PRED,
    "ResNet_WCMCOnly",
)

refs = [
    frozen_ref,
    current_transfer_ref,
    local_ref,
]

for ref in refs:
    if ref is None:
        continue

    prob_cols = [
        c
        for c in ref.columns
        if c
        not in {
            "patient_key",
            "ecg_key",
            "sex_true_binary",
        }
    ]

    if len(
        prob_cols
    ) != 1:
        raise RuntimeError(
            "Could not uniquely identify reference probability column."
        )

    prob_col = prob_cols[
        0
    ]

    merged = aligned.merge(
        ref,
        on=[
            "patient_key",
            "ecg_key",
        ],
        how="inner",
        suffixes=(
            "",
            "_ref",
        ),
        validate="one_to_one",
    )

    if len(
        merged
    ) != len(
        aligned
    ):
        raise RuntimeError(
            f"{prob_col}: alignment changed N "
            f"{len(aligned):,} -> {len(merged):,}"
        )

    if not np.array_equal(
        merged[
            "sex_true_binary"
        ].to_numpy(
            dtype=int
        ),
        merged[
            "sex_true_binary_ref"
        ].to_numpy(
            dtype=int
        ),
    ):
        raise RuntimeError(
            f"{prob_col}: sex-label mismatch."
        )

    aligned = merged.drop(
        columns=[
            "sex_true_binary_ref"
        ]
    )


aligned.to_csv(
    ALIGNED_CSV,
    index=False,
)


# ============================================================
# THRESHOLDS
# ============================================================

THRESHOLDS = {
    "ResNet_MIMICtoWCMC":
        FINAL_THRESHOLD,

    "ResNet_MIMIC_frozen":
        MIMIC_VAL_THRESHOLD,

    "TRiSM_MIMICtoWCMC":
        CURRENT_TRISM_TRANSFER_THRESHOLD,
}

# Recover current WCMC-only ResNet1D validation threshold.
if (
    "ResNet_WCMCOnly"
    in aligned.columns
):
    if RESNET_LOCAL_THRESHOLD_JSON.exists():
        with open(
            RESNET_LOCAL_THRESHOLD_JSON,
            "r",
        ) as f:
            local_info = json.load(
                f
            )

        THRESHOLDS[
            "ResNet_WCMCOnly"
        ] = float(
            local_info[
                "threshold"
            ]
        )

    else:
        raise FileNotFoundError(
            "Current WCMC-only ResNet1D predictions were found, "
            "but ResNet1D_WCMCOnly_threshold.json is missing."
        )


# ============================================================
# POINT PERFORMANCE TABLE
# ============================================================

y = aligned[
    "sex_true_binary"
].to_numpy(
    dtype=int
)

model_columns = [
    c
    for c in [
        "ResNet_MIMICtoWCMC",
        "TRiSM_MIMICtoWCMC",
        "ResNet_WCMCOnly",
        "ResNet_MIMIC_frozen",
    ]
    if c in aligned.columns
]

point_rows = []

for model_key in model_columns:
    p = aligned[
        model_key
    ].to_numpy(
        dtype=float
    )

    m = binary_metrics(
        y,
        p,
        THRESHOLDS[
            model_key
        ],
    )

    point_rows.append(
        {
            "model":
                model_key,

            **m,
        }
    )

point_df = pd.DataFrame(
    point_rows
)


# ============================================================
# PAIRED PATIENT-LEVEL BOOTSTRAP
# ============================================================

print(
    "\n"
    +
    "=" * 126
)

print(
    f"PAIRED WCMC BOOTSTRAP — {N_BOOT:,} REPLICATES"
)

print(
    "=" * 126
)

rng = np.random.default_rng(
    SEED
)

n = len(
    aligned
)

boot_rows = []

for b in range(
    N_BOOT
):
    idx = rng.integers(
        0,
        n,
        n,
    )

    yy = y[
        idx
    ]

    if len(
        np.unique(
            yy
        )
    ) < 2:
        continue

    row = {
        "bootstrap_id":
            b,
    }

    for model_key in model_columns:
        pp = aligned[
            model_key
        ].to_numpy(
            dtype=float
        )[
            idx
        ]

        pred = (
            pp
            >=
            THRESHOLDS[
                model_key
            ]
        ).astype(
            int
        )

        row[
            model_key
            +
            "__AUROC"
        ] = roc_auc_score(
            yy,
            pp,
        )

        row[
            model_key
            +
            "__AUPRC"
        ] = average_precision_score(
            yy,
            pp,
        )

        row[
            model_key
            +
            "__F1"
        ] = f1_score(
            yy,
            pred,
            zero_division=0,
        )

        row[
            model_key
            +
            "__Brier"
        ] = brier_score_loss(
            yy,
            pp,
        )

    boot_rows.append(
        row
    )

    if (
        b
        +
        1
    ) % 100 == 0:
        print(
            f"  bootstrap {b+1:,}/{N_BOOT:,}"
        )


boot_df = pd.DataFrame(
    boot_rows
)

boot_df.to_csv(
    BOOT_CSV,
    index=False,
)


# ============================================================
# PERFORMANCE WITH 95% CI
# ============================================================

perf_rows = []

for model_key in model_columns:
    point = point_df.loc[
        point_df[
            "model"
        ]
        ==
        model_key
    ].iloc[
        0
    ]

    row = {
        "model":
            model_key,
    }

    for metric in [
        "AUROC",
        "AUPRC",
        "F1",
        "Brier",
    ]:
        vals = boot_df[
            model_key
            +
            "__"
            +
            metric
        ].to_numpy(
            dtype=float
        )

        row[
            metric
        ] = float(
            point[
                metric
            ]
        )

        row[
            metric
            +
            "_CI_low"
        ] = float(
            np.percentile(
                vals,
                2.5,
            )
        )

        row[
            metric
            +
            "_CI_high"
        ] = float(
            np.percentile(
                vals,
                97.5,
            )
        )

    for metric in [
        "Accuracy",
        "Balanced_Accuracy",
        "Precision",
        "Sensitivity",
        "Specificity",
        "MCC",
    ]:
        row[
            metric
        ] = float(
            point[
                metric
            ]
        )

    row[
        "threshold"
    ] = float(
        point[
            "threshold"
        ]
    )

    perf_rows.append(
        row
    )


perf_df = pd.DataFrame(
    perf_rows
)

perf_df.to_csv(
    PERF_CSV,
    index=False,
)


# ============================================================
# PAIRED DIFFERENCES — NEW FINAL MODEL VS COMPARATORS
# ============================================================

diff_rows = []

reference = "ResNet_MIMICtoWCMC"

comparators = [
    c
    for c in [
        "TRiSM_MIMICtoWCMC",
        "ResNet_WCMCOnly",
        "ResNet_MIMIC_frozen",
    ]
    if c in model_columns
]

for comparator in comparators:
    for metric in [
        "AUROC",
        "AUPRC",
        "F1",
        "Brier",
    ]:
        a = boot_df[
            reference
            +
            "__"
            +
            metric
        ].to_numpy(
            dtype=float
        )

        bb = boot_df[
            comparator
            +
            "__"
            +
            metric
        ].to_numpy(
            dtype=float
        )

        d = (
            a
            -
            bb
        )

        point_a = float(
            perf_df.loc[
                perf_df[
                    "model"
                ]
                ==
                reference,
                metric,
            ].iloc[
                0
            ]
        )

        point_b = float(
            perf_df.loc[
                perf_df[
                    "model"
                ]
                ==
                comparator,
                metric,
            ].iloc[
                0
            ]
        )

        p_two = (
            2
            *
            min(
                np.mean(
                    d
                    <=
                    0
                ),
                np.mean(
                    d
                    >=
                    0
                ),
            )
        )

        # Resolution floor for N_BOOT bootstrap replicates.
        p_two = max(
            p_two,
            1
            /
            (
                len(
                    d
                )
                +
                1
            ),
        )

        diff_rows.append(
            {
                "reference":
                    reference,

                "comparator":
                    comparator,

                "metric":
                    metric,

                "difference_reference_minus_comparator":
                    point_a
                    -
                    point_b,

                "CI_low":
                    float(
                        np.percentile(
                            d,
                            2.5,
                        )
                    ),

                "CI_high":
                    float(
                        np.percentile(
                            d,
                            97.5,
                        )
                    ),

                "P_two_sided":
                    float(
                        p_two
                    ),
            }
        )


diff_df = pd.DataFrame(
    diff_rows
)

diff_df.to_csv(
    DIFF_CSV,
    index=False,
)


# ============================================================
# SUMMARY JSON
# ============================================================

summary = {
    "model":
        "ResNet1D MIMIC->WCMC",

    "architecture":
        "Residual multi-scale CNN-TCN-Transformer",

    "mimic_pretraining": {
        "cohort_csv":
            str(
                MIMIC_CSV
            ),

        "train_N":
            int(
                len(
                    mimic_train
                )
            ),

        "validation_N":
            int(
                len(
                    mimic_val
                )
            ),

        "best_checkpoint":
            str(
                MIMIC_CKPT
            ),

        "best_epoch":
            int(
                mimic_ckpt.get(
                    "epoch",
                    -1,
                )
            ),

        "best_val_AUPRC":
            float(
                mimic_ckpt.get(
                    "best_val_AUPRC",
                    np.nan,
                )
            ),

        "validation_threshold":
            float(
                MIMIC_VAL_THRESHOLD
            ),
    },

    "wcmc_finetuning": {
        "train_N":
            int(
                len(
                    wcmc_train
                )
            ),

        "validation_N":
            int(
                len(
                    wcmc_val
                )
            ),

        "test_N":
            int(
                len(
                    wcmc_test
                )
            ),

        "best_checkpoint":
            str(
                WCMC_CKPT
            ),

        "best_epoch":
            int(
                best_wcmc_ckpt.get(
                    "epoch",
                    -1,
                )
            ),

        "best_val_AUPRC":
            float(
                best_wcmc_ckpt.get(
                    "best_val_AUPRC",
                    np.nan,
                )
            ),

        "validation_threshold":
            float(
                FINAL_THRESHOLD
            ),
    },

    "paired_bootstrap_replicates":
        int(
            len(
                boot_df
            )
        ),

    "performance_file":
        str(
            PERF_CSV
        ),

    "paired_difference_file":
        str(
            DIFF_CSV
        ),
}

with open(
    SUMMARY_JSON,
    "w",
) as f:
    json.dump(
        summary,
        f,
        indent=2,
    )


# ============================================================
# CONSOLE RESULTS
# ============================================================

print(
    "\n"
    +
    "=" * 126
)

print(
    "FINAL LOCKED WCMC PERFORMANCE — 95% CI"
)

print(
    "=" * 126
)

print(
    perf_df.to_string(
        index=False
    )
)

print(
    "\n"
    +
    "=" * 126
)

print(
    "PAIRED DIFFERENCES — ResNet1D MIMIC->WCMC MINUS COMPARATOR"
)

print(
    "=" * 126
)

print(
    diff_df.to_string(
        index=False
    )
)


new_auc = float(
    perf_df.loc[
        perf_df[
            "model"
        ]
        ==
        "ResNet_MIMICtoWCMC",
        "AUROC",
    ].iloc[
        0
    ]
)

new_ap = float(
    perf_df.loc[
        perf_df[
            "model"
        ]
        ==
        "ResNet_MIMICtoWCMC",
        "AUPRC",
    ].iloc[
        0
    ]
)

print(
    "\n"
    +
    "=" * 126
)

print(
    "MODEL-IMPROVEMENT DECISION"
)

print(
    "=" * 126
)

print(
    f"New ResTRiSM MIMIC->WCMC AUROC : {new_auc:.6f}"
)

print(
    f"New ResTRiSM MIMIC->WCMC AUPRC : {new_ap:.6f}"
)

print(
    "Current TRiSM MIMIC->WCMC AUROC: 0.944491"
)

print(
    "Current TRiSM MIMIC->WCMC AUPRC: 0.939860"
)

if new_auc > 0.944491:
    print(
        "\n✓ ResNet1D improves AUROC over the current FINAL transfer model."
    )

else:
    print(
        "\nResNet1D does not improve AUROC over the current FINAL transfer model."
    )


print(
    "\nSaved:"
)

print(
    "MIMIC checkpoint:",
    MIMIC_CKPT,
)

print(
    "WCMC checkpoint:",
    WCMC_CKPT,
)

print(
    "MIMIC history:",
    MIMIC_HISTORY_CSV,
)

print(
    "WCMC history:",
    WCMC_HISTORY_CSV,
)

print(
    "Frozen WCMC predictions:",
    WCMC_FROZEN_TEST_PRED_CSV,
)

print(
    "Fine-tuned WCMC predictions:",
    WCMC_TEST_PRED_CSV,
)

print(
    "Aligned predictions:",
    ALIGNED_CSV,
)

print(
    "Performance:",
    PERF_CSV,
)

print(
    "Bootstrap:",
    BOOT_CSV,
)

print(
    "Paired differences:",
    DIFF_CSV,
)

print(
    "Summary:",
    SUMMARY_JSON,
)

print(
    "\nDONE"
)
