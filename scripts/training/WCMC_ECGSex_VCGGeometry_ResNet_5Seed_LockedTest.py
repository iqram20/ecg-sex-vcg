#!/usr/bin/env python
# ==========================================================================================
# WCMC ECG-SEX — VCG-GEOMETRY RESNET
# 5-SEED CONFIRMATION + LOCKED WCMC TEST + PAIRED BOOTSTRAP
# ==========================================================================================
#
# This is the CONFIRMATORY stage for the validation-selected VCG-Geometry ResNet.
#
# ARCHITECTURE / HYPERPARAMETERS
# ------------------------------
# Frozen from the successful 3-seed validation screen:
# - exact pretrained ResNet1D pathway
# - inverse-Dower initialized VCG branch
# - bounded learned Dower correction |ΔD| <= 0.03
# - VCG geometry features: XYZ, dXYZ, magnitude, speed, unit direction,
#   radial velocity, trajectory turning
# - 3 learned VCG trajectory slots
# - zero-initialized gated residual correction
# - WCMC LR base=1e-5, novel branch=1e-4
# - AdamW, WD=1e-4, unweighted BCE, no scheduler
# - max 25 epochs, patience 5
# - checkpoint by WCMC validation AUPRC
# - threshold from WCMC validation only by maximum F1
#
# SEEDS
# -----
# 11, 22, 33, 42, 55
#
# MODES
# -----
# VCGFINAL_MODE=RUN
#   - trains one VCG seed on WCMC train/validation
#   - selects checkpoint by validation AUPRC
#   - after training is finished, evaluates locked WCMC test once
#   - also runs the already-completed matched ResNet 5-seed checkpoint
#     on the same validation/test rows for exact alignment
#
# VCGFINAL_MODE=FINAL
#   - requires all five RUN jobs
#   - equal-probability 5-seed ensemble for VCG and ResNet
#   - threshold for each ensemble determined from mean WCMC validation predictions only
#   - locked WCMC test metrics
#   - 2,000 paired patient-level bootstrap replicates
#   - primary comparison: VCG-Geometry ResNet minus ResNet
#
# Female=0, Male=1
# ==========================================================================================

import os
import gc
import json
import math
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    brier_score_loss,
    precision_recall_curve,
)

warnings.filterwarnings("ignore")


# ==========================================================================================
# CONFIG
# ==========================================================================================

MODE = os.environ.get(
    "VCGFINAL_MODE",
    "RUN",
).strip().upper()

SEED = int(
    os.environ.get(
        "VCGFINAL_SEED",
        "42",
    )
)

SEEDS = [
    11,
    22,
    33,
    42,
    55,
]

if MODE == "RUN" and SEED not in SEEDS:
    raise ValueError(
        f"Unsupported seed={SEED}; expected one of {SEEDS}"
    )

TARGET_LEN = 2500
N_LEADS = 12
IN_CH = 24
DROPOUT = 0.08

DOWER_LEAD_INDICES = [
    0,  # I
    1,  # II
    6,  # V1
    7,  # V2
    8,  # V3
    9,  # V4
    10, # V5
    11, # V6
]

VCG_DIM = 192
N_TRAJECTORY_SLOTS = 3

MAX_DOWER_DELTA = float(
    os.environ.get(
        "VCGFINAL_MAX_DOWER_DELTA",
        "0.03",
    )
)

DOWER_REG_LAMBDA = float(
    os.environ.get(
        "VCGFINAL_DOWER_REG",
        "1e-3",
    )
)

BATCH_SIZE = int(
    os.environ.get(
        "VCGFINAL_BATCH_SIZE",
        "64",
    )
)

NUM_WORKERS = int(
    os.environ.get(
        "VCGFINAL_NUM_WORKERS",
        "8",
    )
)

MAX_EPOCHS = int(
    os.environ.get(
        "VCGFINAL_MAX_EPOCHS",
        "25",
    )
)

PATIENCE = int(
    os.environ.get(
        "VCGFINAL_PATIENCE",
        "5",
    )
)

LR_BASE = float(
    os.environ.get(
        "VCGFINAL_LR_BASE",
        "1e-5",
    )
)

LR_NEW = float(
    os.environ.get(
        "VCGFINAL_LR_NEW",
        "1e-4",
    )
)

WEIGHT_DECAY = 1e-4
GRAD_CLIP = 5.0

N_BOOT = int(
    os.environ.get(
        "VCGFINAL_N_BOOT",
        "2000",
    )
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else
    "cpu"
)


# ==========================================================================================
# PATHS
# ==========================================================================================

RUN_DIR = Path(
    "/athena/madelab/scratch/iqh4001/ECG/ECG_Echo/Runs/"
    "Master_ECG_Sex_MIMIC_EchoNext/"
    "20260609_020154_MASTER_ECG_SEX_MIMIC_ECHONEXT"
)

WCMC_ROOT = (
    RUN_DIR
    /
    "wcmc_ecgsex_from_scratch"
)

SPLIT_DIR = (
    WCMC_ROOT
    /
    "04_wcmc_finetune_from_mimic"
    /
    "splits"
)

TRAIN_CSV = (
    SPLIT_DIR
    /
    "WCMC_ECGSex_train.csv"
)

VAL_CSV = (
    SPLIT_DIR
    /
    "WCMC_ECGSex_validation.csv"
)

TEST_CSV = (
    SPLIT_DIR
    /
    "WCMC_ECGSex_test.csv"
)

MIMIC_RESNET_CKPT = (
    WCMC_ROOT
    /
    "14_resnet1d_mimic_to_wcmc"
    /
    "checkpoints"
    /
    "best_ResNet1D_MIMICPretrain_ECGSex.pt"
)

# Completed 5-seed ResNet confirmation from the comprehensive pipeline.
RESNET_5SEED_ROOT = (
    WCMC_ROOT
    /
    "16_comprehensive_resnet_hybrid_pipeline"
    /
    "03_selected_5seed"
    /
    "resnet_only"
)

OUT_ROOT = (
    WCMC_ROOT
    /
    "19_vcg_geometry_resnet_5seed_locked_wcmc"
)

if MODE == "RUN":
    OUT_DIR = (
        OUT_ROOT
        /
        "seeds"
        /
        f"seed_{SEED}"
    )

    CKPT_PATH = (
        OUT_DIR
        /
        "best_VCGGeometryResNet_WCMC.pt"
    )

    HISTORY_PATH = (
        OUT_DIR
        /
        "wcmc_history.csv"
    )

    VCG_VAL_PRED_PATH = (
        OUT_DIR
        /
        "VCGGeometryResNet_validation_predictions.csv"
    )

    VCG_TEST_PRED_PATH = (
        OUT_DIR
        /
        "VCGGeometryResNet_locked_test_predictions.csv"
    )

    RESNET_VAL_PRED_PATH = (
        OUT_DIR
        /
        "ResNet_validation_predictions.csv"
    )

    RESNET_TEST_PRED_PATH = (
        OUT_DIR
        /
        "ResNet_locked_test_predictions.csv"
    )

    MATRIX_PATH = (
        OUT_DIR
        /
        "learned_inverse_dower_matrix.csv"
    )

    SUMMARY_PATH = (
        OUT_DIR
        /
        "seed_summary.json"
    )

FINAL_DIR = (
    OUT_ROOT
    /
    "final_5seed_ensemble_bootstrap"
)


# ==========================================================================================
# REPRODUCIBILITY
# ==========================================================================================

def set_seed(
    seed,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


# ==========================================================================================
# INVERSE DOWER
# ==========================================================================================

def inverse_dower_matrix():
    # Rows X,Y,Z; columns I,II,V1,V2,V3,V4,V5,V6.
    return torch.tensor(
        [
            [
                 0.156,
                -0.010,
                -0.172,
                -0.074,
                 0.122,
                 0.231,
                 0.239,
                 0.194,
            ],
            [
                -0.227,
                 0.887,
                 0.057,
                -0.019,
                -0.106,
                -0.022,
                 0.041,
                 0.048,
            ],
            [
                 0.022,
                 0.102,
                -0.229,
                -0.310,
                -0.246,
                -0.063,
                 0.055,
                 0.108,
            ],
        ],
        dtype=torch.float32,
    )


# ==========================================================================================
# PREPROCESSING
# ==========================================================================================

def orient_and_resample(
    arr,
):
    arr = np.asarray(
        arr,
        dtype=np.float32,
    )

    arr = np.squeeze(arr)

    if arr.ndim != 2:
        raise ValueError(
            f"ECG must be 2D; got {arr.shape}"
        )

    if arr.shape[0] == N_LEADS:
        x = arr

    elif arr.shape[1] == N_LEADS:
        x = arr.T

    else:
        raise ValueError(
            f"Cannot infer 12-lead axis from {arr.shape}"
        )

    if x.shape[1] != TARGET_LEN:
        old_grid = np.linspace(
            0.0,
            1.0,
            x.shape[1],
        )

        new_grid = np.linspace(
            0.0,
            1.0,
            TARGET_LEN,
        )

        out = np.zeros(
            (
                N_LEADS,
                TARGET_LEN,
            ),
            dtype=np.float32,
        )

        for lead in range(N_LEADS):
            out[lead] = np.interp(
                new_grid,
                old_grid,
                x[lead],
            )

        x = out

    x = np.nan_to_num(
        x,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if not np.isfinite(x).all():
        raise ValueError(
            "Non-finite ECG after cleanup."
        )

    return x.astype(
        np.float32,
        copy=False,
    )


def make_base_input(
    x12,
):
    mu = x12.mean(
        axis=1,
        keepdims=True,
    )

    sd = x12.std(
        axis=1,
        keepdims=True,
    )

    sd[
        sd < 1e-5
    ] = 1.0

    x = (
        x12
        -
        mu
    ) / sd

    dx = np.diff(
        x,
        axis=1,
        prepend=x[:, :1],
    )

    x24 = np.concatenate(
        [
            x,
            dx,
        ],
        axis=0,
    )

    return torch.tensor(
        x24,
        dtype=torch.float32,
    )


def make_vcg_input(
    x12,
):
    # Preserve inter-lead relative amplitudes.
    x = x12.copy()

    center = np.median(
        x,
        axis=1,
        keepdims=True,
    )

    x = x - center

    rms = float(
        np.sqrt(
            np.mean(
                x ** 2
            )
            +
            1e-8
        )
    )

    if (
        not np.isfinite(rms)
        or
        rms < 1e-8
    ):
        rms = 1.0

    x = x / rms

    x8 = x[
        DOWER_LEAD_INDICES,
        :
    ]

    return torch.tensor(
        x8,
        dtype=torch.float32,
    )


# ==========================================================================================
# DATA
# ==========================================================================================

class WCMCDataset(
    Dataset
):

    def __init__(
        self,
        df,
    ):
        self.df = df.reset_index(
            drop=True
        ).copy()

    def __len__(
        self,
    ):
        return len(self.df)

    def __getitem__(
        self,
        idx,
    ):
        row = self.df.iloc[idx]

        path = str(
            row["raw_local_path"]
        )

        try:
            arr = np.load(
                path,
                allow_pickle=False,
            )

            x12 = orient_and_resample(
                arr
            )

            return {
                "x_base":
                    make_base_input(
                        x12
                    ),

                "x_vcg8":
                    make_vcg_input(
                        x12
                    ),

                "y":
                    float(
                        row["sex_binary"]
                    ),

                "patient_key":
                    str(
                        row["patient_key"]
                    ),

                "ecg_key":
                    str(
                        row["ecg_key"]
                    ),

                "path":
                    path,

                "error":
                    None,
            }

        except Exception as exc:
            return {
                "x_base":
                    None,

                "x_vcg8":
                    None,

                "y":
                    float(
                        row["sex_binary"]
                    ),

                "patient_key":
                    str(
                        row["patient_key"]
                    ),

                "ecg_key":
                    str(
                        row["ecg_key"]
                    ),

                "path":
                    path,

                "error":
                    repr(exc),
            }


def collate_fn(
    batch,
):
    valid = [
        b
        for b in batch
        if b["x_base"] is not None
    ]

    invalid = [
        b
        for b in batch
        if b["x_base"] is None
    ]

    if not valid:
        return {
            "x_base":
                None,

            "x_vcg8":
                None,

            "valid":
                [],

            "invalid":
                invalid,
        }

    return {
        "x_base":
            torch.stack(
                [
                    b["x_base"]
                    for b in valid
                ]
            ),

        "x_vcg8":
            torch.stack(
                [
                    b["x_vcg8"]
                    for b in valid
                ]
            ),

        "valid":
            valid,

        "invalid":
            invalid,
    }


def make_loader(
    df,
    shuffle,
):
    kwargs = {
        "dataset":
            WCMCDataset(df),

        "batch_size":
            BATCH_SIZE,

        "shuffle":
            shuffle,

        "num_workers":
            NUM_WORKERS,

        "pin_memory":
            DEVICE.type == "cuda",

        "drop_last":
            False,

        "collate_fn":
            collate_fn,
    }

    if NUM_WORKERS > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2

    return DataLoader(
        **kwargs
    )


# ==========================================================================================
# RESNET
# ==========================================================================================

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

        else:
            self.skip = nn.Identity()

        self.act = nn.GELU()

    def forward(
        self,
        x,
    ):
        return self.act(
            self.conv(x)
            +
            self.skip(x)
        )


class ResNet1D(
    nn.Module
):

    def __init__(
        self,
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(
                IN_CH,
                64,
                7,
                stride=2,
                padding=3,
            ),
            nn.BatchNorm1d(
                64
            ),
            nn.GELU(),
        )

        self.block1 = ResBlock1D(
            64,
            64,
        )

        self.block2 = ResBlock1D(
            64,
            128,
            stride=2,
        )

        self.block3 = ResBlock1D(
            128,
            128,
        )

        self.block4 = ResBlock1D(
            128,
            256,
            stride=2,
        )

        self.block5 = ResBlock1D(
            256,
            256,
        )

        self.pool = nn.AdaptiveAvgPool1d(
            1
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
    ):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.pool(x)

        return self.head(
            x
        ).squeeze(1)


# ==========================================================================================
# VCG PROJECTION
# ==========================================================================================

class BoundedInverseDower(
    nn.Module
):

    def __init__(
        self,
    ):
        super().__init__()

        self.register_buffer(
            "base_matrix",
            inverse_dower_matrix(),
            persistent=True,
        )

        self.delta_raw = nn.Parameter(
            torch.zeros(
                3,
                8,
            )
        )

    def effective_matrix(
        self,
    ):
        return (
            self.base_matrix
            +
            MAX_DOWER_DELTA
            *
            torch.tanh(
                self.delta_raw
            )
        )

    def forward(
        self,
        x8,
    ):
        matrix = self.effective_matrix()

        xyz = torch.einsum(
            "ce,bet->bct",
            matrix,
            x8,
        )

        rms = torch.sqrt(
            torch.mean(
                xyz ** 2,
                dim=(
                    1,
                    2,
                ),
                keepdim=True,
            )
            +
            1e-6
        )

        return xyz / rms

    def regularization(
        self,
    ):
        correction = (
            MAX_DOWER_DELTA
            *
            torch.tanh(
                self.delta_raw
            )
        )

        return torch.mean(
            correction ** 2
        )


def first_difference(
    x,
):
    return torch.diff(
        x,
        dim=-1,
        prepend=x[..., :1],
    )


def vcg_geometry_features(
    xyz,
):
    eps = 1e-5

    vel = first_difference(
        xyz
    )

    acc = first_difference(
        vel
    )

    magnitude = torch.sqrt(
        torch.sum(
            xyz ** 2,
            dim=1,
            keepdim=True,
        )
        +
        eps
    )

    speed = torch.sqrt(
        torch.sum(
            vel ** 2,
            dim=1,
            keepdim=True,
        )
        +
        eps
    )

    unit = xyz / magnitude

    radial_velocity = (
        torch.sum(
            xyz
            *
            vel,
            dim=1,
            keepdim=True,
        )
        /
        magnitude
    )

    v = vel.transpose(
        1,
        2,
    )

    a = acc.transpose(
        1,
        2,
    )

    cross = torch.cross(
        v,
        a,
        dim=-1,
    )

    cross_mag = torch.sqrt(
        torch.sum(
            cross ** 2,
            dim=-1,
            keepdim=True,
        )
        +
        eps
    ).transpose(
        1,
        2,
    )

    turning = torch.tanh(
        cross_mag
        /
        (
            speed ** 2
            +
            1e-3
        )
    )

    features = torch.cat(
        [
            xyz,
            vel,
            magnitude,
            speed,
            unit,
            radial_velocity,
            turning,
        ],
        dim=1,
    )

    if features.shape[1] != 13:
        raise RuntimeError(
            f"Expected 13 geometry channels; got {features.shape}"
        )

    return features


class VCGGeometryEncoder(
    nn.Module
):

    def __init__(
        self,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(
                13,
                64,
                9,
                stride=2,
                padding=4,
            ),
            nn.BatchNorm1d(
                64
            ),
            nn.GELU(),

            nn.Conv1d(
                64,
                96,
                7,
                stride=2,
                padding=3,
            ),
            nn.BatchNorm1d(
                96
            ),
            nn.GELU(),

            nn.Conv1d(
                96,
                128,
                5,
                stride=2,
                padding=2,
            ),
            nn.BatchNorm1d(
                128
            ),
            nn.GELU(),

            nn.Conv1d(
                128,
                VCG_DIM,
                3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm1d(
                VCG_DIM
            ),
            nn.GELU(),

            nn.Conv1d(
                VCG_DIM,
                VCG_DIM,
                3,
                padding=1,
            ),
            nn.BatchNorm1d(
                VCG_DIM
            ),
            nn.GELU(),
        )

        self.slot_queries = nn.Parameter(
            torch.empty(
                N_TRAJECTORY_SLOTS,
                VCG_DIM,
            )
        )

        nn.init.normal_(
            self.slot_queries,
            mean=0.0,
            std=0.02,
        )

        self.slot_norm = nn.LayerNorm(
            VCG_DIM
        )

        self.fuse = nn.Sequential(
            nn.Linear(
                VCG_DIM
                *
                (
                    N_TRAJECTORY_SLOTS
                    +
                    1
                ),
                384,
            ),
            nn.GELU(),
            nn.Dropout(
                DROPOUT
            ),
            nn.Linear(
                384,
                256,
            ),
            nn.GELU(),
            nn.Dropout(
                DROPOUT
            ),
        )

    def forward(
        self,
        features,
    ):
        h = self.encoder(
            features
        )

        ht = h.transpose(
            1,
            2,
        )

        scores = torch.einsum(
            "btd,sd->bts",
            ht,
            self.slot_queries,
        ) / math.sqrt(
            VCG_DIM
        )

        weights = torch.softmax(
            scores,
            dim=1,
        )

        slots = torch.einsum(
            "bts,btd->bsd",
            weights,
            ht,
        )

        slots = self.slot_norm(
            slots
        )

        mean_pool = torch.mean(
            h,
            dim=-1,
        )

        fused = torch.cat(
            [
                slots.reshape(
                    slots.shape[0],
                    -1,
                ),
                mean_pool,
            ],
            dim=1,
        )

        return self.fuse(
            fused
        )


class VCGGeometryResNet(
    nn.Module
):

    def __init__(
        self,
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(
                IN_CH,
                64,
                7,
                stride=2,
                padding=3,
            ),
            nn.BatchNorm1d(
                64
            ),
            nn.GELU(),
        )

        self.block1 = ResBlock1D(
            64,
            64,
        )

        self.block2 = ResBlock1D(
            64,
            128,
            stride=2,
        )

        self.block3 = ResBlock1D(
            128,
            128,
        )

        self.block4 = ResBlock1D(
            128,
            256,
            stride=2,
        )

        self.block5 = ResBlock1D(
            256,
            256,
        )

        self.base_pool = nn.AdaptiveAvgPool1d(
            1
        )

        self.base_head = nn.Sequential(
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

        self.vcg_projection = BoundedInverseDower()
        self.vcg_encoder = VCGGeometryEncoder()

        self.vcg_delta_head = nn.Sequential(
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

        nn.init.zeros_(
            self.vcg_delta_head[-1].weight
        )

        nn.init.zeros_(
            self.vcg_delta_head[-1].bias
        )

        self.gate = nn.Linear(
            512,
            1,
        )

        nn.init.zeros_(
            self.gate.weight
        )

        nn.init.constant_(
            self.gate.bias,
            -2.0,
        )

    def base_forward(
        self,
        x,
    ):
        h = self.stem(x)
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)

        pooled = self.base_pool(h)
        z_base = torch.flatten(
            pooled,
            1,
        )

        base_logit = self.base_head(
            pooled
        ).squeeze(1)

        return (
            base_logit,
            z_base,
        )

    def forward(
        self,
        x_base,
        x_vcg8,
        return_components=False,
    ):
        base_logit, z_base = self.base_forward(
            x_base
        )

        xyz = self.vcg_projection(
            x_vcg8
        )

        geometry = vcg_geometry_features(
            xyz
        )

        z_vcg = self.vcg_encoder(
            geometry
        )

        delta = self.vcg_delta_head(
            z_vcg
        ).squeeze(1)

        gate = torch.sigmoid(
            self.gate(
                torch.cat(
                    [
                        z_base,
                        z_vcg,
                    ],
                    dim=1,
                )
            )
        ).squeeze(1)

        final_logit = (
            base_logit
            +
            gate
            *
            delta
        )

        if return_components:
            return {
                "logit":
                    final_logit,

                "base_logit":
                    base_logit,

                "delta":
                    delta,

                "gate":
                    gate,
            }

        return final_logit

    def base_parameters(
        self,
    ):
        modules = [
            self.stem,
            self.block1,
            self.block2,
            self.block3,
            self.block4,
            self.block5,
            self.base_head,
        ]

        for module in modules:
            yield from module.parameters()

    def novel_parameters(
        self,
    ):
        modules = [
            self.vcg_projection,
            self.vcg_encoder,
            self.vcg_delta_head,
            self.gate,
        ]

        for module in modules:
            yield from module.parameters()


# ==========================================================================================
# CHECKPOINT UTILITIES
# ==========================================================================================

def extract_state_dict(
    checkpoint,
):
    if (
        isinstance(
            checkpoint,
            dict,
        )
        and
        "model_state_dict"
        in checkpoint
    ):
        state = checkpoint[
            "model_state_dict"
        ]

    elif (
        isinstance(
            checkpoint,
            dict,
        )
        and
        "state_dict"
        in checkpoint
    ):
        state = checkpoint[
            "state_dict"
        ]

    elif (
        isinstance(
            checkpoint,
            dict,
        )
        and
        all(
            torch.is_tensor(v)
            for v in checkpoint.values()
        )
    ):
        state = checkpoint

    else:
        raise RuntimeError(
            "Could not identify checkpoint state_dict."
        )

    out = {}

    for key, value in state.items():
        key = str(key)

        if key.startswith(
            "module."
        ):
            key = key[
                len("module.") :
            ]

        out[key] = value

    return out


def initialize_vcg_from_mimic_resnet(
    model,
):
    checkpoint = torch.load(
        MIMIC_RESNET_CKPT,
        map_location=DEVICE,
        weights_only=False,
    )

    source = extract_state_dict(
        checkpoint
    )

    target = model.state_dict()

    mapping = {
        "net.0.":
            "stem.0.",

        "net.1.":
            "stem.1.",

        "net.3.":
            "block1.",

        "net.4.":
            "block2.",

        "net.5.":
            "block3.",

        "net.6.":
            "block4.",

        "net.7.":
            "block5.",

        "head.":
            "base_head.",
    }

    transferred = 0

    for key, value in source.items():
        target_key = None

        for src_prefix, dst_prefix in mapping.items():
            if key.startswith(src_prefix):
                target_key = (
                    dst_prefix
                    +
                    key[len(src_prefix):]
                )
                break

        if (
            target_key is not None
            and
            target_key in target
            and
            target[target_key].shape == value.shape
        ):
            target[target_key] = value
            transferred += 1

    model.load_state_dict(
        target,
        strict=True,
    )

    if transferred < 80:
        raise RuntimeError(
            f"Only {transferred} MIMIC ResNet tensors transferred."
        )

    return transferred


def resnet_seed_checkpoint(
    seed,
):
    path = (
        RESNET_5SEED_ROOT
        /
        f"seed_{seed}"
        /
        "best_wcmc.pt"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing completed ResNet seed checkpoint:\n{path}"
        )

    return path


def load_resnet_seed_model(
    seed,
):
    model = ResNet1D().to(
        DEVICE
    )

    checkpoint = torch.load(
        resnet_seed_checkpoint(
            seed
        ),
        map_location=DEVICE,
        weights_only=False,
    )

    source = extract_state_dict(
        checkpoint
    )

    target = model.state_dict()

    # Comprehensive ResNet-only checkpoint.
    mapping = {}

    for key in source:
        if key.startswith("stem."):
            mapping[key] = key

        elif key.startswith("block1."):
            mapping[key] = key

        elif key.startswith("block2."):
            mapping[key] = key

        elif key.startswith("block3."):
            mapping[key] = key

        elif key.startswith("block4."):
            mapping[key] = key

        elif key.startswith("block5."):
            mapping[key] = key

        elif key.startswith("base_head."):
            mapping[key] = (
                "head."
                +
                key[len("base_head."):]
            )

        elif key.startswith("head."):
            mapping[key] = key

    transferred = 0

    for src_key, dst_key in mapping.items():
        if (
            dst_key in target
            and
            target[dst_key].shape == source[src_key].shape
        ):
            target[dst_key] = source[src_key]
            transferred += 1

    model.load_state_dict(
        target,
        strict=True,
    )

    if transferred < 80:
        raise RuntimeError(
            f"Only {transferred} tensors loaded from ResNet seed {seed}."
        )

    return model


# ==========================================================================================
# METRICS
# ==========================================================================================

def select_f1_threshold(
    y,
    p,
):
    precision, recall, thresholds = precision_recall_curve(
        y,
        p,
    )

    if len(thresholds) == 0:
        return 0.5

    f1 = (
        2
        *
        precision[:-1]
        *
        recall[:-1]
        /
        np.clip(
            precision[:-1]
            +
            recall[:-1],
            1e-12,
            None,
        )
    )

    return float(
        thresholds[
            int(
                np.nanargmax(f1)
            )
        ]
    )


def metrics_at_threshold(
    y,
    p,
    threshold,
):
    pred = (
        p >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        pred,
        labels=[0, 1],
    ).ravel()

    return {
        "AUROC":
            float(
                roc_auc_score(
                    y,
                    p,
                )
            ),

        "AUPRC":
            float(
                average_precision_score(
                    y,
                    p,
                )
            ),

        "Accuracy":
            float(
                accuracy_score(
                    y,
                    pred,
                )
            ),

        "Balanced_Accuracy":
            float(
                balanced_accuracy_score(
                    y,
                    pred,
                )
            ),

        "Precision":
            float(
                precision_score(
                    y,
                    pred,
                    zero_division=0,
                )
            ),

        "Sensitivity":
            float(
                recall_score(
                    y,
                    pred,
                    zero_division=0,
                )
            ),

        "Specificity":
            float(
                tn
                /
                max(
                    tn + fp,
                    1,
                )
            ),

        "F1":
            float(
                f1_score(
                    y,
                    pred,
                    zero_division=0,
                )
            ),

        "Brier":
            float(
                brier_score_loss(
                    y,
                    p,
                )
            ),
    }


# ==========================================================================================
# TRAIN / INFERENCE
# ==========================================================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler,
):
    model.train()

    total = 0.0
    total_n = 0
    failures = 0

    for batch in loader:
        failures += len(
            batch["invalid"]
        )

        if batch["x_base"] is None:
            continue

        x_base = batch["x_base"].to(
            DEVICE,
            non_blocking=True,
        )

        x_vcg8 = batch["x_vcg8"].to(
            DEVICE,
            non_blocking=True,
        )

        y = torch.tensor(
            [
                b["y"]
                for b in batch["valid"]
            ],
            dtype=torch.float32,
            device=DEVICE,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with autocast(
            enabled=DEVICE.type == "cuda"
        ):
            logits = model(
                x_base,
                x_vcg8,
            )

            bce = criterion(
                logits,
                y,
            )

            reg = model.vcg_projection.regularization()

            loss = (
                bce
                +
                DOWER_REG_LAMBDA
                *
                reg
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

        n = int(y.numel())

        total += (
            float(
                loss.detach().cpu()
            )
            *
            n
        )

        total_n += n

    if total_n == 0:
        raise RuntimeError(
            "No valid training samples."
        )

    return (
        total
        /
        total_n,
        failures,
    )


@torch.no_grad()
def predict_vcg(
    model,
    loader,
):
    model.eval()

    rows = []
    failures = []

    for batch in loader:
        failures.extend(
            batch["invalid"]
        )

        if batch["x_base"] is None:
            continue

        x_base = batch["x_base"].to(
            DEVICE,
            non_blocking=True,
        )

        x_vcg8 = batch["x_vcg8"].to(
            DEVICE,
            non_blocking=True,
        )

        with autocast(
            enabled=DEVICE.type == "cuda"
        ):
            comp = model(
                x_base,
                x_vcg8,
                return_components=True,
            )

        probs = torch.sigmoid(
            comp["logit"]
        ).float().cpu().numpy()

        gates = comp["gate"].float().cpu().numpy()
        deltas = comp["delta"].float().cpu().numpy()

        for i, item in enumerate(
            batch["valid"]
        ):
            rows.append(
                {
                    "patient_key":
                        item["patient_key"],

                    "ecg_key":
                        item["ecg_key"],

                    "sex_true_binary":
                        int(
                            item["y"]
                        ),

                    "prob_male":
                        float(
                            probs[i]
                        ),

                    "vcg_gate":
                        float(
                            gates[i]
                        ),

                    "vcg_delta_logit":
                        float(
                            deltas[i]
                        ),
                }
            )

    return (
        pd.DataFrame(rows),
        pd.DataFrame(
            [
                {
                    "patient_key":
                        x["patient_key"],

                    "ecg_key":
                        x["ecg_key"],

                    "path":
                        x["path"],

                    "error":
                        x["error"],
                }
                for x in failures
            ]
        ),
    )


@torch.no_grad()
def predict_resnet(
    model,
    loader,
):
    model.eval()

    rows = []
    failures = []

    for batch in loader:
        failures.extend(
            batch["invalid"]
        )

        if batch["x_base"] is None:
            continue

        x = batch["x_base"].to(
            DEVICE,
            non_blocking=True,
        )

        with autocast(
            enabled=DEVICE.type == "cuda"
        ):
            logits = model(x)

        probs = torch.sigmoid(
            logits
        ).float().cpu().numpy()

        for i, item in enumerate(
            batch["valid"]
        ):
            rows.append(
                {
                    "patient_key":
                        item["patient_key"],

                    "ecg_key":
                        item["ecg_key"],

                    "sex_true_binary":
                        int(
                            item["y"]
                        ),

                    "prob_male":
                        float(
                            probs[i]
                        ),
                }
            )

    return (
        pd.DataFrame(rows),
        pd.DataFrame(
            [
                {
                    "patient_key":
                        x["patient_key"],

                    "ecg_key":
                        x["ecg_key"],

                    "path":
                        x["path"],

                    "error":
                        x["error"],
                }
                for x in failures
            ]
        ),
    )


def save_dower_matrix(
    model,
):
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

    axes = [
        "X",
        "Y",
        "Z",
    ]

    leads = [
        "I",
        "II",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
    ]

    rows = []

    for i, axis in enumerate(axes):
        for j, lead in enumerate(leads):
            rows.append(
                {
                    "axis":
                        axis,

                    "lead":
                        lead,

                    "base":
                        float(
                            base[i, j]
                        ),

                    "effective":
                        float(
                            effective[i, j]
                        ),

                    "delta":
                        float(
                            effective[i, j]
                            -
                            base[i, j]
                        ),
                }
            )

    pd.DataFrame(
        rows
    ).to_csv(
        MATRIX_PATH,
        index=False,
    )


# ==========================================================================================
# RUN ONE SEED
# ==========================================================================================

def run_one_seed():
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "RUN mode requires CUDA."
        )

    set_seed(SEED)

    print(
        "=" * 120
    )

    print(
        "VCG-GEOMETRY RESNET — 5-SEED CONFIRMATION"
    )

    print(
        "=" * 120
    )

    print(
        "Seed:",
        SEED,
    )

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    print(
        "Architecture/hyperparameters:",
        "FROZEN FROM VALIDATION SCREEN",
    )

    for path in [
        TRAIN_CSV,
        VAL_CSV,
        TEST_CSV,
        MIMIC_RESNET_CKPT,
        resnet_seed_checkpoint(SEED),
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    train_df = pd.read_csv(
        TRAIN_CSV,
        low_memory=False,
    )

    val_df = pd.read_csv(
        VAL_CSV,
        low_memory=False,
    )

    test_df = pd.read_csv(
        TEST_CSV,
        low_memory=False,
    )

    print(
        f"Train N={len(train_df):,}"
    )

    print(
        f"Validation N={len(val_df):,}"
    )

    print(
        f"Locked test N={len(test_df):,}"
    )

    # Strict patient-level separation.
    train_ids = set(
        train_df["patient_key"].astype(str)
    )

    val_ids = set(
        val_df["patient_key"].astype(str)
    )

    test_ids = set(
        test_df["patient_key"].astype(str)
    )

    if (
        train_ids & val_ids
        or
        train_ids & test_ids
        or
        val_ids & test_ids
    ):
        raise RuntimeError(
            "Patient overlap detected across WCMC splits."
        )

    train_loader = make_loader(
        train_df,
        shuffle=True,
    )

    val_loader = make_loader(
        val_df,
        shuffle=False,
    )

    test_loader = make_loader(
        test_df,
        shuffle=False,
    )

    model = VCGGeometryResNet().to(
        DEVICE
    )

    transferred = initialize_vcg_from_mimic_resnet(
        model
    )

    print(
        f"Transferred MIMIC ResNet tensors: {transferred}"
    )

    base_params = list(
        model.base_parameters()
    )

    novel_params = list(
        model.novel_parameters()
    )

    base_ids = {
        id(p)
        for p in base_params
    }

    novel_ids = {
        id(p)
        for p in novel_params
    }

    all_ids = {
        id(p)
        for p in model.parameters()
        if p.requires_grad
    }

    if base_ids & novel_ids:
        raise RuntimeError(
            "Optimizer groups overlap."
        )

    if (
        base_ids
        |
        novel_ids
    ) != all_ids:
        raise RuntimeError(
            "Optimizer groups do not cover all trainable parameters."
        )

    optimizer = torch.optim.AdamW(
        [
            {
                "params":
                    base_params,

                "lr":
                    LR_BASE,
            },
            {
                "params":
                    novel_params,

                "lr":
                    LR_NEW,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.BCEWithLogitsLoss()

    scaler = GradScaler(
        enabled=DEVICE.type == "cuda"
    )

    history = []
    best_auprc = -np.inf
    best_epoch = -1
    no_improve = 0

    # Model selection uses validation only.
    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):
        train_loss, train_failures = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
        )

        val_pred, val_fail = predict_vcg(
            model,
            val_loader,
        )

        y = val_pred[
            "sex_true_binary"
        ].to_numpy(
            dtype=int
        )

        p = val_pred[
            "prob_male"
        ].to_numpy(
            dtype=float
        )

        threshold = select_f1_threshold(
            y,
            p,
        )

        metrics = metrics_at_threshold(
            y,
            p,
            threshold,
        )

        mean_gate = float(
            val_pred[
                "vcg_gate"
            ].mean()
        )

        history.append(
            {
                "epoch":
                    epoch,

                "train_loss":
                    train_loss,

                "train_failures":
                    train_failures,

                "validation_failures":
                    len(val_fail),

                "val_AUROC":
                    metrics["AUROC"],

                "val_AUPRC":
                    metrics["AUPRC"],

                "val_F1":
                    metrics["F1"],

                "val_Brier":
                    metrics["Brier"],

                "val_threshold":
                    threshold,

                "mean_vcg_gate":
                    mean_gate,
            }
        )

        print(
            f"WCMC | VCG-Geometry ResNet | seed={SEED} | "
            f"epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"loss={train_loss:.5f} | "
            f"AUROC={metrics['AUROC']:.6f} | "
            f"AUPRC={metrics['AUPRC']:.6f} | "
            f"F1={metrics['F1']:.6f} | "
            f"Brier={metrics['Brier']:.6f} | "
            f"gate={mean_gate:.4f}"
        )

        if (
            metrics["AUPRC"]
            >
            best_auprc
            +
            1e-7
        ):
            best_auprc = metrics["AUPRC"]
            best_epoch = epoch
            no_improve = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "architecture":
                        "VCG-Geometry ResNet",

                    "seed":
                        SEED,

                    "epoch":
                        epoch,

                    "best_val_AUPRC":
                        best_auprc,

                    "LR_BASE":
                        LR_BASE,

                    "LR_NEW":
                        LR_NEW,

                    "MAX_DOWER_DELTA":
                        MAX_DOWER_DELTA,

                    "DOWER_REG_LAMBDA":
                        DOWER_REG_LAMBDA,
                },
                CKPT_PATH,
            )

        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            print(
                f"Early stop at epoch {epoch}; best epoch={best_epoch}."
            )

            break

    pd.DataFrame(
        history
    ).to_csv(
        HISTORY_PATH,
        index=False,
    )

    # Restore best validation checkpoint.
    checkpoint = torch.load(
        CKPT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    # Freeze architecture/checkpoint/threshold before reading test predictions.
    vcg_val, val_fail = predict_vcg(
        model,
        val_loader,
    )

    y_val = vcg_val[
        "sex_true_binary"
    ].to_numpy(
        dtype=int
    )

    p_val = vcg_val[
        "prob_male"
    ].to_numpy(
        dtype=float
    )

    vcg_threshold = select_f1_threshold(
        y_val,
        p_val,
    )

    vcg_val.to_csv(
        VCG_VAL_PRED_PATH,
        index=False,
    )

    # Locked test used only after validation-defined checkpoint/threshold.
    vcg_test, test_fail = predict_vcg(
        model,
        test_loader,
    )

    vcg_test.to_csv(
        VCG_TEST_PRED_PATH,
        index=False,
    )

    save_dower_matrix(
        model
    )

    vcg_test_metrics = metrics_at_threshold(
        vcg_test[
            "sex_true_binary"
        ].to_numpy(
            dtype=int
        ),
        vcg_test[
            "prob_male"
        ].to_numpy(
            dtype=float
        ),
        vcg_threshold,
    )

    # Exact matched ResNet baseline from already-completed 5-seed confirmation.
    resnet_model = load_resnet_seed_model(
        SEED
    )

    resnet_val, resnet_val_fail = predict_resnet(
        resnet_model,
        val_loader,
    )

    resnet_threshold = select_f1_threshold(
        resnet_val[
            "sex_true_binary"
        ].to_numpy(
            dtype=int
        ),
        resnet_val[
            "prob_male"
        ].to_numpy(
            dtype=float
        ),
    )

    resnet_val.to_csv(
        RESNET_VAL_PRED_PATH,
        index=False,
    )

    resnet_test, resnet_test_fail = predict_resnet(
        resnet_model,
        test_loader,
    )

    resnet_test.to_csv(
        RESNET_TEST_PRED_PATH,
        index=False,
    )

    resnet_test_metrics = metrics_at_threshold(
        resnet_test[
            "sex_true_binary"
        ].to_numpy(
            dtype=int
        ),
        resnet_test[
            "prob_male"
        ].to_numpy(
            dtype=float
        ),
        resnet_threshold,
    )

    summary = {
        "seed":
            SEED,

        "VCG_best_epoch":
            int(
                checkpoint["epoch"]
            ),

        "VCG_validation_threshold":
            float(
                vcg_threshold
            ),

        "ResNet_validation_threshold":
            float(
                resnet_threshold
            ),

        "VCG_locked_test":
            vcg_test_metrics,

        "ResNet_locked_test":
            resnet_test_metrics,

        "delta_VCG_minus_ResNet":
            {
                metric:
                    float(
                        vcg_test_metrics[metric]
                        -
                        resnet_test_metrics[metric]
                    )
                for metric in [
                    "AUROC",
                    "AUPRC",
                    "F1",
                    "Brier",
                ]
            },

        "VCG_mean_gate_validation":
            float(
                vcg_val[
                    "vcg_gate"
                ].mean()
            ),

        "VCG_mean_gate_test":
            float(
                vcg_test[
                    "vcg_gate"
                ].mean()
            ),

        "validation_failures":
            int(
                len(val_fail)
            ),

        "test_failures":
            int(
                len(test_fail)
            ),

        "resnet_validation_failures":
            int(
                len(resnet_val_fail)
            ),

        "resnet_test_failures":
            int(
                len(resnet_test_fail)
            ),
    }

    with open(
        SUMMARY_PATH,
        "w",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print(
        "\n"
        +
        "=" * 120
    )

    print(
        "SEED LOCKED-TEST RESULT"
    )

    print(
        "=" * 120
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    del model
    del resnet_model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ==========================================================================================
# FINAL 5-SEED ENSEMBLE
# ==========================================================================================

def load_seed_prediction(
    seed,
    model_name,
    split,
):
    root = (
        OUT_ROOT
        /
        "seeds"
        /
        f"seed_{seed}"
    )

    if model_name == "VCG":
        filename = (
            "VCGGeometryResNet_validation_predictions.csv"
            if split == "validation"
            else
            "VCGGeometryResNet_locked_test_predictions.csv"
        )

    elif model_name == "ResNet":
        filename = (
            "ResNet_validation_predictions.csv"
            if split == "validation"
            else
            "ResNet_locked_test_predictions.csv"
        )

    else:
        raise ValueError(model_name)

    path = root / filename

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    for col in [
        "patient_key",
        "ecg_key",
    ]:
        df[col] = df[col].astype(str)

    return df


def build_ensemble(
    model_name,
    split,
):
    merged = None

    for seed in SEEDS:
        df = load_seed_prediction(
            seed,
            model_name,
            split,
        )

        keep = df[
            [
                "patient_key",
                "ecg_key",
                "sex_true_binary",
                "prob_male",
            ]
        ].copy()

        keep = keep.rename(
            columns={
                "prob_male":
                    f"prob_seed{seed}",
            }
        )

        if merged is None:
            merged = keep

        else:
            keep = keep.rename(
                columns={
                    "sex_true_binary":
                        f"sex_seed{seed}",
                }
            )

            merged = merged.merge(
                keep,
                on=[
                    "patient_key",
                    "ecg_key",
                ],
                how="inner",
                validate="one_to_one",
            )

            if not np.array_equal(
                merged[
                    "sex_true_binary"
                ].to_numpy(
                    dtype=int
                ),
                merged[
                    f"sex_seed{seed}"
                ].to_numpy(
                    dtype=int
                ),
            ):
                raise RuntimeError(
                    f"{model_name} label mismatch across seeds."
                )

            merged = merged.drop(
                columns=[
                    f"sex_seed{seed}"
                ]
            )

    prob_cols = [
        f"prob_seed{seed}"
        for seed in SEEDS
    ]

    merged[
        "prob_ensemble"
    ] = merged[
        prob_cols
    ].mean(
        axis=1
    )

    return merged


def align_models(
    vcg,
    resnet,
):
    a = vcg[
        [
            "patient_key",
            "ecg_key",
            "sex_true_binary",
            "prob_ensemble",
        ]
    ].rename(
        columns={
            "prob_ensemble":
                "prob_vcg",
        }
    )

    b = resnet[
        [
            "patient_key",
            "ecg_key",
            "sex_true_binary",
            "prob_ensemble",
        ]
    ].rename(
        columns={
            "sex_true_binary":
                "sex_resnet",

            "prob_ensemble":
                "prob_resnet",
        }
    )

    out = a.merge(
        b,
        on=[
            "patient_key",
            "ecg_key",
        ],
        validate="one_to_one",
    )

    if not np.array_equal(
        out[
            "sex_true_binary"
        ].to_numpy(
            dtype=int
        ),
        out[
            "sex_resnet"
        ].to_numpy(
            dtype=int
        ),
    ):
        raise RuntimeError(
            "VCG/ResNet labels are misaligned."
        )

    return out.drop(
        columns=[
            "sex_resnet"
        ]
    )


def bootstrap_ci_and_difference(
    aligned,
    vcg_threshold,
    resnet_threshold,
):
    y = aligned[
        "sex_true_binary"
    ].to_numpy(
        dtype=int
    )

    p_vcg = aligned[
        "prob_vcg"
    ].to_numpy(
        dtype=float
    )

    p_res = aligned[
        "prob_resnet"
    ].to_numpy(
        dtype=float
    )

    point_vcg = metrics_at_threshold(
        y,
        p_vcg,
        vcg_threshold,
    )

    point_res = metrics_at_threshold(
        y,
        p_res,
        resnet_threshold,
    )

    rng = np.random.default_rng(
        42
    )

    rows = []

    n = len(y)

    for b in range(N_BOOT):
        idx = rng.integers(
            0,
            n,
            n,
        )

        yy = y[idx]

        if np.unique(yy).size < 2:
            continue

        pv = p_vcg[idx]
        pr = p_res[idx]

        mv = metrics_at_threshold(
            yy,
            pv,
            vcg_threshold,
        )

        mr = metrics_at_threshold(
            yy,
            pr,
            resnet_threshold,
        )

        row = {
            "bootstrap_id":
                b + 1,
        }

        for metric in [
            "AUROC",
            "AUPRC",
            "Accuracy",
            "Balanced_Accuracy",
            "Precision",
            "Sensitivity",
            "Specificity",
            "F1",
            "Brier",
        ]:
            row[
                f"VCG__{metric}"
            ] = mv[metric]

            row[
                f"ResNet__{metric}"
            ] = mr[metric]

            row[
                f"Delta_VCG_minus_ResNet__{metric}"
            ] = (
                mv[metric]
                -
                mr[metric]
            )

        rows.append(row)

        if (
            b + 1
        ) % 100 == 0:
            print(
                f"bootstrap {b+1:,}/{N_BOOT:,}"
            )

    boot = pd.DataFrame(rows)

    return (
        point_vcg,
        point_res,
        boot,
    )


def p_two_sided(
    delta,
):
    delta = np.asarray(
        delta,
        dtype=float,
    )

    p = (
        2.0
        *
        min(
            np.mean(
                delta <= 0
            ),
            np.mean(
                delta >= 0
            ),
        )
    )

    return max(
        float(p),
        1.0
        /
        (
            len(delta)
            +
            1
        ),
    )


def final_ensemble():
    FINAL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Refuse partial finalization.
    missing = []

    for seed in SEEDS:
        summary_path = (
            OUT_ROOT
            /
            "seeds"
            /
            f"seed_{seed}"
            /
            "seed_summary.json"
        )

        if not summary_path.exists():
            missing.append(
                str(summary_path)
            )

    if missing:
        raise RuntimeError(
            "Missing seed runs; refusing partial finalization:\n"
            +
            "\n".join(missing)
        )

    print(
        "=" * 120
    )

    print(
        "VCG-GEOMETRY RESNET — FINAL 5-SEED WCMC ENSEMBLE"
    )

    print(
        "=" * 120
    )

    vcg_val = build_ensemble(
        "VCG",
        "validation",
    )

    res_val = build_ensemble(
        "ResNet",
        "validation",
    )

    val_aligned = align_models(
        vcg_val,
        res_val,
    )

    y_val = val_aligned[
        "sex_true_binary"
    ].to_numpy(
        dtype=int
    )

    vcg_threshold = select_f1_threshold(
        y_val,
        val_aligned[
            "prob_vcg"
        ].to_numpy(
            dtype=float
        ),
    )

    resnet_threshold = select_f1_threshold(
        y_val,
        val_aligned[
            "prob_resnet"
        ].to_numpy(
            dtype=float
        ),
    )

    print(
        f"VCG validation threshold: {vcg_threshold:.6f}"
    )

    print(
        f"ResNet validation threshold: {resnet_threshold:.6f}"
    )

    val_aligned.to_csv(
        FINAL_DIR
        /
        "WCMC_Validation_Aligned_5SeedEnsemble_Predictions.csv",
        index=False,
    )

    vcg_test = build_ensemble(
        "VCG",
        "test",
    )

    res_test = build_ensemble(
        "ResNet",
        "test",
    )

    test_aligned = align_models(
        vcg_test,
        res_test,
    )

    test_aligned.to_csv(
        FINAL_DIR
        /
        "WCMC_LockedTest_Aligned_5SeedEnsemble_Predictions.csv",
        index=False,
    )

    point_vcg, point_res, boot = bootstrap_ci_and_difference(
        test_aligned,
        vcg_threshold,
        resnet_threshold,
    )

    boot.to_csv(
        FINAL_DIR
        /
        "WCMC_VCG_vs_ResNet_5SeedEnsemble_PairedBootstrap_Distributions.csv",
        index=False,
    )

    perf_rows = []

    for model_name, point in [
        (
            "VCG-Geometry ResNet 5-seed ensemble",
            point_vcg,
        ),
        (
            "ResNet 5-seed ensemble",
            point_res,
        ),
    ]:
        prefix = (
            "VCG"
            if model_name.startswith("VCG")
            else
            "ResNet"
        )

        row = {
            "model":
                model_name,

            "N":
                len(
                    test_aligned
                ),

            "threshold":
                vcg_threshold
                if prefix == "VCG"
                else
                resnet_threshold,
        }

        for metric in [
            "AUROC",
            "AUPRC",
            "Accuracy",
            "Balanced_Accuracy",
            "Precision",
            "Sensitivity",
            "Specificity",
            "F1",
            "Brier",
        ]:
            vals = boot[
                f"{prefix}__{metric}"
            ].to_numpy(
                dtype=float
            )

            row[metric] = point[metric]

            row[
                f"{metric}_CI_low"
            ] = float(
                np.percentile(
                    vals,
                    2.5,
                )
            )

            row[
                f"{metric}_CI_high"
            ] = float(
                np.percentile(
                    vals,
                    97.5,
                )
            )

        perf_rows.append(row)

    perf_df = pd.DataFrame(
        perf_rows
    )

    perf_df.to_csv(
        FINAL_DIR
        /
        "WCMC_VCG_vs_ResNet_5SeedEnsemble_Performance_With95CI.csv",
        index=False,
    )

    diff_rows = []

    for metric in [
        "AUROC",
        "AUPRC",
        "Accuracy",
        "Balanced_Accuracy",
        "Precision",
        "Sensitivity",
        "Specificity",
        "F1",
        "Brier",
    ]:
        col = (
            f"Delta_VCG_minus_ResNet__{metric}"
        )

        delta = boot[
            col
        ].to_numpy(
            dtype=float
        )

        point_delta = (
            point_vcg[metric]
            -
            point_res[metric]
        )

        diff_rows.append(
            {
                "comparison":
                    "VCG-Geometry ResNet minus ResNet",

                "metric":
                    metric,

                "VCG_point":
                    point_vcg[metric],

                "ResNet_point":
                    point_res[metric],

                "delta_VCG_minus_ResNet":
                    point_delta,

                "CI_low":
                    float(
                        np.percentile(
                            delta,
                            2.5,
                        )
                    ),

                "CI_high":
                    float(
                        np.percentile(
                            delta,
                            97.5,
                        )
                    ),

                "P":
                    p_two_sided(
                        delta
                    ),
            }
        )

    diff_df = pd.DataFrame(
        diff_rows
    )

    diff_df.to_csv(
        FINAL_DIR
        /
        "WCMC_VCG_vs_ResNet_5SeedEnsemble_PairedDifferences.csv",
        index=False,
    )

    print(
        "\n"
        +
        "=" * 120
    )

    print(
        "LOCKED WCMC TEST — PERFORMANCE"
    )

    print(
        "=" * 120
    )

    print(
        perf_df[
            [
                "model",
                "N",
                "threshold",
                "AUROC",
                "AUROC_CI_low",
                "AUROC_CI_high",
                "AUPRC",
                "AUPRC_CI_low",
                "AUPRC_CI_high",
                "F1",
                "Brier",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\n"
        +
        "=" * 120
    )

    print(
        "PAIRED DIFFERENCES — VCG MINUS RESNET"
    )

    print(
        "=" * 120
    )

    print(
        diff_df.to_string(
            index=False
        )
    )

    primary = diff_df.loc[
        diff_df[
            "metric"
        ].isin(
            [
                "AUROC",
                "AUPRC",
                "F1",
                "Brier",
            ]
        )
    ].copy()

    summary = {
        "architecture":
            "VCG-Geometry ResNet",

        "seeds":
            SEEDS,

        "N_bootstrap":
            N_BOOT,

        "VCG_validation_threshold":
            vcg_threshold,

        "ResNet_validation_threshold":
            resnet_threshold,

        "VCG_locked_test":
            point_vcg,

        "ResNet_locked_test":
            point_res,

        "paired_primary_metrics":
            primary.to_dict(
                orient="records"
            ),
    }

    with open(
        FINAL_DIR
        /
        "WCMC_VCG_vs_ResNet_5SeedEnsemble_FinalSummary.json",
        "w",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print(
        "\nSaved to:",
        FINAL_DIR,
    )


# ==========================================================================================
# MAIN
# ==========================================================================================

if __name__ == "__main__":
    if MODE == "RUN":
        run_one_seed()

    elif MODE == "FINAL":
        final_ensemble()

    else:
        raise ValueError(
            f"Unknown VCGFINAL_MODE={MODE}"
        )
