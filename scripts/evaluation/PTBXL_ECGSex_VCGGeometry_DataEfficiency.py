#!/usr/bin/env python
# ==========================================================================================
# PTB-XL ECG-SEX — FINAL VCG-GEOMETRY RESNET DATA-EFFICIENCY PIPELINE
# ==========================================================================================
#
# FINAL PRIMARY ARCHITECTURE
# --------------------------
# VCG-Geometry ResNet
#
# GOAL
# ----
# Quantify local labeled-data efficiency of the FINAL primary architecture.
#
# Strategy 1: PTB-XL-only VCG-Geometry ResNet
#     random initialization -> PTB-XL
#
# Strategy 2: MIMIC→WCMC→PTB-XL VCG-Geometry ResNet
#     seed-matched FINAL WCMC VCG checkpoint -> PTB-XL
#
# FRACTIONS
# ---------
# 10%, 25%, 50%, 100% of official PTB-XL folds 1-8.
#
# SEEDS
# -----
# 11, 22, 42.
#
# OFFICIAL SPLIT
# --------------
# folds 1-8 : local training pool
# fold 9    : validation
# fold 10   : untouched test
#
# IMPORTANT
# ---------
# - One locked ECG per patient.
# - Wherever present, this script REUSES the exact selected_training_subset.csv
#   from the completed ResNet data-efficiency run for the same fraction/seed.
#   This guarantees matched local training examples across architectures.
# - Same PTB subset is used for local-only and transfer within fraction/seed.
# - Checkpoint selection: fold-9 validation AUPRC.
# - Threshold: fold-9 Youden J, matching the prior PTB data-efficiency protocol.
# - Fold 10 is never used for training, early stopping, checkpoint selection,
#   or threshold selection.
# - Female=0, Male=1.
#
# FINAL VCG INPUT
# ---------------
# Base branch:
#   12 per-lead z-scored waveforms + 12 first derivatives = 24 channels.
#
# VCG branch:
#   raw I, II, V1-V6 -> inverse Dower -> XYZ -> geometry encoder.
#   Inter-lead amplitudes are preserved using a single recording-level RMS scale.
#
# TRAINING
# --------
# Local-only:
#   all parameters LR=3e-4
#
# Transfer:
#   all already-trained VCG parameters LR=1e-5
#
# Both:
#   AdamW, WD=1e-4, unweighted BCE, no scheduler,
#   grad clip=5, max 30 epochs, patience 5.
#
# MODES
# -----
#   PTB_VCG_MODE=RUN
#   PTB_VCG_MODE=AGGREGATE
#
# ENVIRONMENT
# -----------
#   PTB_VCG_FRACTION=0.10|0.25|0.50|1.00
#   PTB_VCG_SEED=11|22|42
# ==========================================================================================

import os
import gc
import json
import math
import random
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

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
    roc_curve,
    brier_score_loss,
    matthews_corrcoef,
)

warnings.filterwarnings("ignore")


# ==========================================================================================
# CONFIG
# ==========================================================================================

MODE = os.environ.get(
    "PTB_VCG_MODE",
    "RUN",
).strip().upper()

TRAIN_FRACTION = float(
    os.environ.get(
        "PTB_VCG_FRACTION",
        "0.10",
    )
)

EXPERIMENT_SEED = int(
    os.environ.get(
        "PTB_VCG_SEED",
        "42",
    )
)

ALLOWED_FRACTIONS = [
    0.10,
    0.25,
    0.50,
    1.00,
]

ALLOWED_SEEDS = [
    11,
    22,
    42,
]

if MODE == "RUN":
    if not any(
        abs(
            TRAIN_FRACTION
            -
            x
        )
        <
        1e-10
        for x in ALLOWED_FRACTIONS
    ):
        raise ValueError(
            f"Invalid PTB_VCG_FRACTION={TRAIN_FRACTION}"
        )

    if EXPERIMENT_SEED not in ALLOWED_SEEDS:
        raise ValueError(
            f"Invalid PTB_VCG_SEED={EXPERIMENT_SEED}"
        )

TARGET_LEN = 2500
N_LEADS = 12
IN_CH = 24
DROPOUT = 0.08

ECG_LEADS = [
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]

DOWER_INPUT_LEADS = [
    "I",
    "II",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]

DOWER_LEAD_INDICES = [
    0,
    1,
    6,
    7,
    8,
    9,
    10,
    11,
]

VCG_DIM = 192
N_TRAJECTORY_SLOTS = 3
MAX_DOWER_DELTA = 0.03

BATCH_SIZE = int(
    os.environ.get(
        "PTB_VCG_BATCH_SIZE",
        "96",
    )
)

NUM_WORKERS = int(
    os.environ.get(
        "PTB_VCG_NUM_WORKERS",
        "8",
    )
)

MAX_EPOCHS = int(
    os.environ.get(
        "PTB_VCG_MAX_EPOCHS",
        "30",
    )
)

PATIENCE = int(
    os.environ.get(
        "PTB_VCG_PATIENCE",
        "5",
    )
)

LR_LOCAL = float(
    os.environ.get(
        "PTB_VCG_LR_LOCAL",
        "3e-4",
    )
)

LR_TRANSFER = float(
    os.environ.get(
        "PTB_VCG_LR_TRANSFER",
        "1e-5",
    )
)

WEIGHT_DECAY = 1e-4
GRAD_CLIP = 5.0

N_BOOT = int(
    os.environ.get(
        "PTB_VCG_N_BOOT",
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

PTBXL_PROJECT_ROOT = Path(
    "/athena/madelab/scratch/iqh4001/ECG/PTB-XL-ECG"
)

PTBXL_DATA_ROOT = (
    PTBXL_PROJECT_ROOT
    /
    "Data"
)

PTB_ROOT = (
    RUN_DIR
    /
    "ptbxl_ecgsex_from_scratch"
)

COHORT_CSV = (
    PTB_ROOT
    /
    "01_cohort"
    /
    "PTBXL_ECGSex_Locked_OneECGPerPatient_AgeQC.csv"
)

WCMC_VCG_ROOT = (
    RUN_DIR
    /
    "wcmc_ecgsex_from_scratch"
    /
    "19_vcg_geometry_resnet_5seed_locked_wcmc"
    /
    "seeds"
)

RESNET_EFF_ROOT = (
    PTB_ROOT
    /
    "11_resnet1d_data_efficiency"
)

RESNET_RUNS_ROOT = (
    RESNET_EFF_ROOT
    /
    "runs"
)

RESNET_FINAL_ROOT = (
    RESNET_EFF_ROOT
    /
    "final_seedensemble_bootstrap"
)

OUT_ROOT = (
    PTB_ROOT
    /
    "12_vcg_geometry_data_efficiency"
)

RUNS_ROOT = (
    OUT_ROOT
    /
    "runs"
)

FINAL_ROOT = (
    OUT_ROOT
    /
    "final_seedensemble_bootstrap"
)

if os.environ.get(
    "PTB_VCG_SKIP_MKDIR",
    "0",
) != "1":
    RUNS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    FINAL_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


def fraction_tag(
    fraction,
):
    return (
        f"{int(round(fraction * 100)):03d}pct"
    )


if MODE == "RUN":
    RUN_TAG = (
        f"{fraction_tag(TRAIN_FRACTION)}"
        f"_seed{EXPERIMENT_SEED}"
    )

    OUT_DIR = (
        RUNS_ROOT
        /
        RUN_TAG
    )

    CKPT_DIR = (
        OUT_DIR
        /
        "checkpoints"
    )

    PRED_DIR = (
        OUT_DIR
        /
        "predictions"
    )

    if os.environ.get(
        "PTB_VCG_SKIP_MKDIR",
        "0",
    ) != "1":
        for d in [
            OUT_DIR,
            CKPT_DIR,
            PRED_DIR,
        ]:
            d.mkdir(
                parents=True,
                exist_ok=True,
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
# ECG PREPROCESSING
# ==========================================================================================

def orient_resample_raw(
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

    return np.nan_to_num(
        x,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(
        np.float32,
        copy=False,
    )


def make_base_input(
    x12,
):
    mean = x12.mean(
        axis=1,
        keepdims=True,
    )

    std = x12.std(
        axis=1,
        keepdims=True,
    )

    std[
        std < 1e-5
    ] = 1.0

    z = (
        x12
        -
        mean
    ) / std

    dz = np.diff(
        z,
        axis=1,
        prepend=z[:, :1],
    )

    return torch.tensor(
        np.concatenate(
            [
                z,
                dz,
            ],
            axis=0,
        ),
        dtype=torch.float32,
    )


def make_vcg_input(
    x12,
):
    # Preserve inter-lead relative amplitude before inverse-Dower transform.
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

    return torch.tensor(
        x[
            DOWER_LEAD_INDICES,
            :
        ],
        dtype=torch.float32,
    )


# ==========================================================================================
# FINAL VCG-GEOMETRY RESNET
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
            self.conv(
                x
            )
            +
            self.skip(
                x
            )
        )


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
        xyz = torch.einsum(
            "ce,bet->bct",
            self.effective_matrix(),
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


def first_difference(
    x,
):
    return torch.diff(
        x,
        dim=-1,
        prepend=x[
            ...,
            :1
        ],
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

    return torch.cat(
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
                    slots.shape[
                        0
                    ],
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

        # Same residual initialization used in final WCMC development.
        nn.init.zeros_(
            self.vcg_delta_head[
                -1
            ].weight
        )

        nn.init.zeros_(
            self.vcg_delta_head[
                -1
            ].bias
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
        h = self.stem(
            x
        )

        h = self.block1(
            h
        )

        h = self.block2(
            h
        )

        h = self.block3(
            h
        )

        h = self.block4(
            h
        )

        h = self.block5(
            h
        )

        pooled = self.base_pool(
            h
        )

        z_base = torch.flatten(
            pooled,
            1,
        )

        base_logit = self.base_head(
            pooled
        ).squeeze(
            1
        )

        return (
            base_logit,
            z_base,
        )

    def forward(
        self,
        x_base,
        x_vcg8,
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
        ).squeeze(
            1
        )

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
        ).squeeze(
            1
        )

        return (
            base_logit
            +
            gate
            *
            delta
        )


# ==========================================================================================
# CHECKPOINT
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

    else:
        state = checkpoint

    clean = {}

    for key, value in state.items():
        key = str(
            key
        )

        if key.startswith(
            "module."
        ):
            key = key[
                len(
                    "module."
                ):
            ]

        clean[
            key
        ] = value

    return clean


def transfer_checkpoint_for_seed(
    seed,
):
    path = (
        WCMC_VCG_ROOT
        /
        f"seed_{seed}"
        /
        "best_VCGGeometryResNet_WCMC.pt"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing seed-matched final WCMC VCG checkpoint:\n{path}"
        )

    return path


def initialize_model(
    init_checkpoint=None,
):
    model = VCGGeometryResNet().to(
        DEVICE
    )

    if init_checkpoint is None:
        return model

    checkpoint = torch.load(
        init_checkpoint,
        map_location=DEVICE,
        weights_only=False,
    )

    state = extract_state_dict(
        checkpoint
    )

    model.load_state_dict(
        state,
        strict=True,
    )

    print(
        "Loaded final seed-matched WCMC VCG checkpoint:",
        init_checkpoint,
    )

    return model


# ==========================================================================================
# PTB-XL WFDB
# ==========================================================================================

def normalize_lead_name(
    name,
):
    s = str(
        name
    ).strip()

    upper = s.upper()

    lookup = {
        "I":
            "I",

        "II":
            "II",

        "III":
            "III",

        "AVR":
            "aVR",

        "AVL":
            "aVL",

        "AVF":
            "aVF",

        "V1":
            "V1",

        "V2":
            "V2",

        "V3":
            "V3",

        "V4":
            "V4",

        "V5":
            "V5",

        "V6":
            "V6",
    }

    if upper not in lookup:
        raise ValueError(
            f"Unknown lead={name}"
        )

    return lookup[
        upper
    ]


def record_exists(
    path,
):
    return (
        Path(
            str(
                path
            )
            +
            ".hea"
        ).is_file()
        and
        Path(
            str(
                path
            )
            +
            ".dat"
        ).is_file()
    )


def resolve_wfdb_path(
    row,
):
    for col in [
        "wfdb_path_hr",
        "hr_path",
        "record_path_hr",
        "record_path",
    ]:
        if (
            col in row.index
            and
            pd.notna(
                row[
                    col
                ]
            )
        ):
            p = str(
                row[
                    col
                ]
            ).strip()

            if p.endswith(
                ".hea"
            ) or p.endswith(
                ".dat"
            ):
                p = p[:-4]

            if record_exists(
                p
            ):
                return p

    if (
        "filename_hr"
        not in row.index
        or
        pd.isna(
            row[
                "filename_hr"
            ]
        )
    ):
        raise ValueError(
            "No usable PTB-XL HR path."
        )

    rel = str(
        row[
            "filename_hr"
        ]
    ).strip()

    if rel.endswith(
        ".hea"
    ) or rel.endswith(
        ".dat"
    ):
        rel = rel[:-4]

    candidates = [
        PTBXL_DATA_ROOT
        /
        rel
    ]

    if rel.startswith(
        "records500/"
    ):
        candidates.append(
            PTBXL_DATA_ROOT
            /
            rel[
                len(
                    "records500/"
                ):
            ]
        )

    parts = Path(
        rel
    ).parts

    if len(
        parts
    ) >= 2:
        candidates.append(
            PTBXL_DATA_ROOT
            /
            parts[
                -2
            ]
            /
            parts[
                -1
            ]
        )

    for candidate in candidates:
        if record_exists(
            candidate
        ):
            return str(
                candidate
            )

    raise FileNotFoundError(
        f"Could not resolve PTB-XL ECG: {rel}"
    )


def load_ptbxl_ecg(
    record_path,
):
    signal, fields = wfdb.rdsamp(
        str(
            record_path
        )
    )

    signal = np.asarray(
        signal,
        dtype=np.float32,
    )

    names = [
        normalize_lead_name(
            x
        )
        for x in fields[
            "sig_name"
        ]
    ]

    missing = [
        lead
        for lead in ECG_LEADS
        if lead not in names
    ]

    if missing:
        raise ValueError(
            f"Missing leads={missing}"
        )

    idx = [
        names.index(
            lead
        )
        for lead in ECG_LEADS
    ]

    return signal[
        :,
        idx
    ].T


# ==========================================================================================
# OFFICIAL FOLDS
# ==========================================================================================

def find_metadata():
    candidates = [
        PTBXL_DATA_ROOT
        /
        "ptbxl_database.csv",

        PTBXL_PROJECT_ROOT
        /
        "ptbxl_database.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    matches = sorted(
        PTBXL_PROJECT_ROOT.glob(
            "**/ptbxl_database.csv"
        )
    )

    if not matches:
        raise FileNotFoundError(
            "ptbxl_database.csv not found."
        )

    return matches[
        0
    ]


def ensure_strat_fold(
    df,
):
    out = df.copy()

    if (
        "strat_fold"
        in out.columns
        and
        out[
            "strat_fold"
        ].notna().all()
    ):
        out[
            "strat_fold"
        ] = pd.to_numeric(
            out[
                "strat_fold"
            ],
            errors="raise",
        ).astype(
            int
        )

        return out

    meta = pd.read_csv(
        find_metadata(),
        low_memory=False,
    )

    out[
        "ecg_id"
    ] = pd.to_numeric(
        out[
            "ecg_id"
        ],
        errors="raise",
    )

    meta[
        "ecg_id"
    ] = pd.to_numeric(
        meta[
            "ecg_id"
        ],
        errors="raise",
    )

    if "strat_fold" in out.columns:
        out = out.drop(
            columns=[
                "strat_fold"
            ]
        )

    out = out.merge(
        meta[
            [
                "ecg_id",
                "strat_fold",
            ]
        ].drop_duplicates(
            "ecg_id"
        ),
        on="ecg_id",
        how="left",
        validate="one_to_one",
    )

    if out[
        "strat_fold"
    ].isna().any():
        raise RuntimeError(
            "Could not recover all official PTB-XL folds."
        )

    out[
        "strat_fold"
    ] = out[
        "strat_fold"
    ].astype(
        int
    )

    return out


# ==========================================================================================
# EXACT MATCHED TRAINING SUBSET
# ==========================================================================================

def select_training_subset(
    full_train,
):
    matched_path = (
        RESNET_RUNS_ROOT
        /
        (
            f"{fraction_tag(TRAIN_FRACTION)}"
            f"_seed{EXPERIMENT_SEED}"
        )
        /
        "selected_training_subset.csv"
    )

    if matched_path.exists():
        selected = pd.read_csv(
            matched_path,
            low_memory=False,
        )

        selected_ids = set(
            selected[
                "ecg_id"
            ].astype(
                str
            )
        )

        out = full_train.loc[
            full_train[
                "ecg_id"
            ].astype(
                str
            ).isin(
                selected_ids
            )
        ].copy()

        if len(
            out
        ) != len(
            selected_ids
        ):
            raise RuntimeError(
                "Could not recover exact matched ResNet training subset."
            )

        print(
            "Training subset source: EXACT completed ResNet subset"
        )

        print(
            "Matched subset:",
            matched_path,
        )

        return (
            out.reset_index(
                drop=True
            ),
            str(
                matched_path
            ),
        )

    # Fallback only if old matched subset is unavailable.
    if TRAIN_FRACTION < 1.0:
        out, _ = train_test_split(
            full_train,
            train_size=TRAIN_FRACTION,
            random_state=EXPERIMENT_SEED,
            stratify=full_train[
                "sex_binary"
            ],
        )

    else:
        out = full_train.copy()

    print(
        "WARNING: matched ResNet subset not found; "
        "recreated deterministically from locked cohort."
    )

    return (
        out.reset_index(
            drop=True
        ),
        "deterministic fallback",
    )


# ==========================================================================================
# DATASET
# ==========================================================================================

class PTBXLDataset(
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

        record_path = resolve_wfdb_path(
            row
        )

        x12 = orient_resample_raw(
            load_ptbxl_ecg(
                record_path
            )
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
                torch.tensor(
                    float(
                        row[
                            "sex_binary"
                        ]
                    ),
                    dtype=torch.float32,
                ),

            "patient_id":
                str(
                    row[
                        "patient_id"
                    ]
                ),

            "ecg_id":
                str(
                    row[
                        "ecg_id"
                    ]
                ),

            "fold":
                int(
                    row[
                        "strat_fold"
                    ]
                ),
        }


def make_loader(
    df,
    shuffle,
):
    kwargs = {
        "dataset":
            PTBXLDataset(
                df
            ),

        "batch_size":
            BATCH_SIZE,

        "shuffle":
            shuffle,

        "num_workers":
            NUM_WORKERS,

        "pin_memory":
            torch.cuda.is_available(),

        "drop_last":
            False,
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


# ==========================================================================================
# METRICS
# ==========================================================================================

def discrimination_metrics(
    y,
    p,
):
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

        "Brier":
            float(
                brier_score_loss(
                    y,
                    p,
                )
            ),
    }


def select_threshold_youden(
    y,
    p,
):
    fpr, tpr, thresholds = roc_curve(
        y,
        p,
    )

    finite = np.isfinite(
        thresholds
    )

    fpr = fpr[
        finite
    ]

    tpr = tpr[
        finite
    ]

    thresholds = thresholds[
        finite
    ]

    if not len(
        thresholds
    ):
        return 0.5

    idx = int(
        np.argmax(
            tpr
            -
            fpr
        )
    )

    return float(
        np.clip(
            thresholds[
                idx
            ],
            0.0,
            1.0,
        )
    )


def threshold_metrics(
    y,
    p,
    threshold,
):
    pred = (
        p
        >=
        threshold
    ).astype(
        int
    )

    tn, fp, fn, tp = confusion_matrix(
        y,
        pred,
        labels=[
            0,
            1,
        ],
    ).ravel()

    return {
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
                    tn
                    +
                    fp,
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

        "MCC":
            float(
                matthews_corrcoef(
                    y,
                    pred,
                )
            ),
    }


# ==========================================================================================
# TRAIN / PREDICT
# ==========================================================================================

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

    for batch in loader:
        x_base = batch[
            "x_base"
        ].to(
            DEVICE,
            non_blocking=True,
        )

        x_vcg8 = batch[
            "x_vcg8"
        ].to(
            DEVICE,
            non_blocking=True,
        )

        y = batch[
            "y"
        ].to(
            DEVICE,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with autocast(
            enabled=torch.cuda.is_available()
        ):
            logits = model(
                x_base,
                x_vcg8,
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

        n = int(
            y.numel()
        )

        total_loss += (
            float(
                loss.detach().cpu()
            )
            *
            n
        )

        total_n += n

    return (
        total_loss
        /
        max(
            total_n,
            1,
        )
    )


@torch.no_grad()
def predict_loader(
    model,
    loader,
):
    model.eval()

    y_all = []
    p_all = []
    patients = []
    ecgs = []
    folds = []

    for batch in loader:
        x_base = batch[
            "x_base"
        ].to(
            DEVICE,
            non_blocking=True,
        )

        x_vcg8 = batch[
            "x_vcg8"
        ].to(
            DEVICE,
            non_blocking=True,
        )

        with autocast(
            enabled=torch.cuda.is_available()
        ):
            logits = model(
                x_base,
                x_vcg8,
            )

        probs = torch.sigmoid(
            logits
        )

        y_all.append(
            batch[
                "y"
            ].numpy()
        )

        p_all.append(
            probs.float().cpu().numpy()
        )

        patients.extend(
            batch[
                "patient_id"
            ]
        )

        ecgs.extend(
            batch[
                "ecg_id"
            ]
        )

        folds.extend(
            [
                int(
                    x
                )
                for x in batch[
                    "fold"
                ]
            ]
        )

    return {
        "y":
            np.concatenate(
                y_all
            ).astype(
                int
            ),

        "p":
            np.concatenate(
                p_all
            ).astype(
                float
            ),

        "patient_id":
            patients,

        "ecg_id":
            ecgs,

        "fold":
            folds,
    }


def fit_strategy(
    strategy,
    safe_name,
    train_loader,
    val_loader,
    test_loader,
    lr,
    init_checkpoint,
):
    set_seed(
        EXPERIMENT_SEED
    )

    model = initialize_model(
        init_checkpoint
    )

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=WEIGHT_DECAY,
    )

    scaler = GradScaler(
        enabled=torch.cuda.is_available()
    )

    ckpt_path = (
        CKPT_DIR
        /
        f"best_VCGGeometryResNet_{safe_name}.pt"
    )

    best = -np.inf
    best_epoch = -1
    no_improve = 0
    history = []

    for epoch in range(
        1,
        MAX_EPOCHS
        +
        1
    ):
        t0 = time.time()

        loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
        )

        val = predict_loader(
            model,
            val_loader,
        )

        disc = discrimination_metrics(
            val[
                "y"
            ],
            val[
                "p"
            ],
        )

        history.append(
            {
                "strategy":
                    strategy,

                "epoch":
                    epoch,

                "train_loss":
                    loss,

                "val_AUROC":
                    disc[
                        "AUROC"
                    ],

                "val_AUPRC":
                    disc[
                        "AUPRC"
                    ],

                "val_Brier":
                    disc[
                        "Brier"
                    ],

                "seconds":
                    time.time()
                    -
                    t0,
            }
        )

        print(
            f"{strategy} | epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"loss={loss:.6f} | "
            f"val AUROC={disc['AUROC']:.6f} | "
            f"val AUPRC={disc['AUPRC']:.6f} | "
            f"Brier={disc['Brier']:.6f}"
        )

        if (
            disc[
                "AUPRC"
            ]
            >
            best
            +
            1e-7
        ):
            best = disc[
                "AUPRC"
            ]

            best_epoch = epoch

            no_improve = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "strategy":
                        strategy,

                    "epoch":
                        epoch,

                    "best_val_AUPRC":
                        best,

                    "experiment_seed":
                        EXPERIMENT_SEED,

                    "train_fraction":
                        TRAIN_FRACTION,
                },
                ckpt_path,
            )

        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            print(
                f"Early stop at epoch {epoch}; best epoch={best_epoch}"
            )

            break

    pd.DataFrame(
        history
    ).to_csv(
        OUT_DIR
        /
        f"{safe_name}_history.csv",
        index=False,
    )

    checkpoint = torch.load(
        ckpt_path,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    val = predict_loader(
        model,
        val_loader,
    )

    threshold = select_threshold_youden(
        val[
            "y"
        ],
        val[
            "p"
        ],
    )

    test = predict_loader(
        model,
        test_loader,
    )

    val_disc = discrimination_metrics(
        val[
            "y"
        ],
        val[
            "p"
        ],
    )

    test_disc = discrimination_metrics(
        test[
            "y"
        ],
        test[
            "p"
        ],
    )

    test_thr = threshold_metrics(
        test[
            "y"
        ],
        test[
            "p"
        ],
        threshold,
    )

    pred_df = pd.DataFrame(
        {
            "patient_id":
                test[
                    "patient_id"
                ],

            "ecg_id":
                test[
                    "ecg_id"
                ],

            "strat_fold":
                test[
                    "fold"
                ],

            "sex_true_binary":
                test[
                    "y"
                ],

            "prob_male":
                test[
                    "p"
                ],

            "pred_binary":
                (
                    test[
                        "p"
                    ]
                    >=
                    threshold
                ).astype(
                    int
                ),

            "strategy":
                strategy,

            "train_fraction":
                TRAIN_FRACTION,

            "experiment_seed":
                EXPERIMENT_SEED,
        }
    )

    pred_path = (
        PRED_DIR
        /
        f"{safe_name}_fold10_predictions.csv"
    )

    pred_df.to_csv(
        pred_path,
        index=False,
    )

    result = {
        "strategy":
            strategy,

        "safe_name":
            safe_name,

        "train_fraction":
            TRAIN_FRACTION,

        "train_percent":
            int(
                round(
                    TRAIN_FRACTION
                    *
                    100
                )
            ),

        "experiment_seed":
            EXPERIMENT_SEED,

        "best_epoch":
            int(
                best_epoch
            ),

        "best_val_AUPRC":
            float(
                best
            ),

        "validation_threshold":
            float(
                threshold
            ),

        "validation_AUROC":
            val_disc[
                "AUROC"
            ],

        "validation_AUPRC":
            val_disc[
                "AUPRC"
            ],

        "validation_Brier":
            val_disc[
                "Brier"
            ],

        "test_AUROC":
            test_disc[
                "AUROC"
            ],

        "test_AUPRC":
            test_disc[
                "AUPRC"
            ],

        "test_Brier":
            test_disc[
                "Brier"
            ],

        "test_Accuracy":
            test_thr[
                "Accuracy"
            ],

        "test_Balanced_Accuracy":
            test_thr[
                "Balanced_Accuracy"
            ],

        "test_Precision":
            test_thr[
                "Precision"
            ],

        "test_Sensitivity":
            test_thr[
                "Sensitivity"
            ],

        "test_Specificity":
            test_thr[
                "Specificity"
            ],

        "test_F1":
            test_thr[
                "F1"
            ],

        "test_MCC":
            test_thr[
                "MCC"
            ],

        "checkpoint":
            str(
                ckpt_path
            ),

        "prediction_file":
            str(
                pred_path
            ),
    }

    del model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ==========================================================================================
# RUN ONE FRACTION × SEED
# ==========================================================================================

def run_one():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "RUN mode requires CUDA."
        )

    print(
        "=" * 120
    )

    print(
        "PTB-XL ECG-SEX — FINAL VCG-GEOMETRY RESNET DATA EFFICIENCY"
    )

    print(
        "=" * 120
    )

    print(
        "GPU:",
        torch.cuda.get_device_name(
            0
        ),
    )

    print(
        "Fraction:",
        TRAIN_FRACTION,
    )

    print(
        "Seed:",
        EXPERIMENT_SEED,
    )

    transfer_ckpt = transfer_checkpoint_for_seed(
        EXPERIMENT_SEED
    )

    cohort = pd.read_csv(
        COHORT_CSV,
        low_memory=False,
    )

    cohort = ensure_strat_fold(
        cohort
    )

    cohort[
        "sex_binary"
    ] = pd.to_numeric(
        cohort[
            "sex_binary"
        ],
        errors="raise",
    ).astype(
        int
    )

    full_train = cohort.loc[
        cohort[
            "strat_fold"
        ].isin(
            range(
                1,
                9,
            )
        )
    ].copy()

    val_df = cohort.loc[
        cohort[
            "strat_fold"
        ]
        ==
        9
    ].copy()

    test_df = cohort.loc[
        cohort[
            "strat_fold"
        ]
        ==
        10
    ].copy()

    (
        train_df,
        subset_source,
    ) = select_training_subset(
        full_train
    )

    val_df = val_df.reset_index(
        drop=True
    )

    test_df = test_df.reset_index(
        drop=True
    )

    train_ids = set(
        train_df[
            "patient_id"
        ].astype(
            str
        )
    )

    val_ids = set(
        val_df[
            "patient_id"
        ].astype(
            str
        )
    )

    test_ids = set(
        test_df[
            "patient_id"
        ].astype(
            str
        )
    )

    if (
        train_ids
        &
        val_ids
        or
        train_ids
        &
        test_ids
        or
        val_ids
        &
        test_ids
    ):
        raise RuntimeError(
            "Patient overlap detected."
        )

    train_df.to_csv(
        OUT_DIR
        /
        "selected_training_subset.csv",
        index=False,
    )

    print(
        f"Full folds 1-8 : {len(full_train):,}"
    )

    print(
        f"Selected train : {len(train_df):,}"
    )

    print(
        f"Validation f9  : {len(val_df):,}"
    )

    print(
        f"Test f10       : {len(test_df):,}"
    )

    train_loader = make_loader(
        train_df,
        True,
    )

    val_loader = make_loader(
        val_df,
        False,
    )

    test_loader = make_loader(
        test_df,
        False,
    )

    results = []

    results.append(
        fit_strategy(
            strategy="PTB-XL-only VCG-Geometry ResNet",
            safe_name="PTBXL_Only_VCG",
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            lr=LR_LOCAL,
            init_checkpoint=None,
        )
    )

    results.append(
        fit_strategy(
            strategy="MIMIC→WCMC→PTB-XL VCG-Geometry ResNet",
            safe_name="MIMIC_WCMC_PTBXL_VCG",
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            lr=LR_TRANSFER,
            init_checkpoint=transfer_ckpt,
        )
    )

    results_df = pd.DataFrame(
        results
    )

    results_df[
        "train_N"
    ] = len(
        train_df
    )

    results_path = (
        OUT_DIR
        /
        f"PTBXL_VCG_DataEfficiency_{RUN_TAG}_Performance.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    with open(
        OUT_DIR
        /
        f"PTBXL_VCG_DataEfficiency_{RUN_TAG}_Summary.json",
        "w",
    ) as f:
        json.dump(
            {
                "run_tag":
                    RUN_TAG,

                "fraction":
                    TRAIN_FRACTION,

                "seed":
                    EXPERIMENT_SEED,

                "full_train_N":
                    int(
                        len(
                            full_train
                        )
                    ),

                "selected_train_N":
                    int(
                        len(
                            train_df
                        )
                    ),

                "validation_N":
                    int(
                        len(
                            val_df
                        )
                    ),

                "test_N":
                    int(
                        len(
                            test_df
                        )
                    ),

                "training_subset_source":
                    subset_source,

                "same_training_subset_for_local_and_transfer":
                    True,

                "validation_fold":
                    9,

                "test_fold":
                    10,

                "test_used_for_training_or_selection":
                    False,

                "threshold_source":
                    "PTB-XL fold-9 validation only; Youden J",

                "checkpoint_selection":
                    "PTB-XL fold-9 validation AUPRC",

                "transfer_checkpoint":
                    str(
                        transfer_ckpt
                    ),

                "results":
                    results,
            },
            f,
            indent=2,
        )

    print(
        "\n"
        +
        "=" * 120
    )

    print(
        "FINAL RESULT"
    )

    print(
        "=" * 120
    )

    print(
        results_df[
            [
                "strategy",
                "train_fraction",
                "experiment_seed",
                "train_N",
                "best_epoch",
                "test_AUROC",
                "test_AUPRC",
                "test_F1",
                "test_Brier",
            ]
        ].to_string(
            index=False
        )
    )


# ==========================================================================================
# AGGREGATION
# ==========================================================================================

def expected_run_dir(
    fraction,
    seed,
):
    return (
        RUNS_ROOT
        /
        (
            f"{fraction_tag(fraction)}"
            f"_seed{seed}"
        )
    )


def load_all_run_results():
    rows = []
    missing = []

    for fraction in ALLOWED_FRACTIONS:
        for seed in ALLOWED_SEEDS:
            tag = (
                f"{fraction_tag(fraction)}"
                f"_seed{seed}"
            )

            path = (
                expected_run_dir(
                    fraction,
                    seed,
                )
                /
                f"PTBXL_VCG_DataEfficiency_{tag}_Performance.csv"
            )

            if not path.exists():
                missing.append(
                    str(
                        path
                    )
                )

                continue

            df = pd.read_csv(
                path,
                low_memory=False,
            )

            if len(
                df
            ) != 2:
                raise RuntimeError(
                    f"Expected two strategies in {path}"
                )

            rows.append(
                df
            )

    if missing:
        raise RuntimeError(
            "Missing completed VCG data-efficiency runs:\n"
            +
            "\n".join(
                missing
            )
        )

    return pd.concat(
        rows,
        ignore_index=True,
    )


def load_prediction(
    fraction,
    seed,
    safe_name,
):
    path = (
        expected_run_dir(
            fraction,
            seed,
        )
        /
        "predictions"
        /
        f"{safe_name}_fold10_predictions.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    df[
        "patient_id"
    ] = df[
        "patient_id"
    ].astype(
        str
    )

    df[
        "ecg_id"
    ] = df[
        "ecg_id"
    ].astype(
        str
    )

    return df


def seed_ensemble_prediction(
    fraction,
    strategy,
):
    safe_name = (
        "PTBXL_Only_VCG"
        if strategy
        ==
        "PTB-XL-only VCG-Geometry ResNet"
        else
        "MIMIC_WCMC_PTBXL_VCG"
    )

    merged = None

    for seed in ALLOWED_SEEDS:
        df = load_prediction(
            fraction,
            seed,
            safe_name,
        )

        col = (
            f"prob_seed{seed}"
        )

        this = df[
            [
                "patient_id",
                "ecg_id",
                "sex_true_binary",
                "prob_male",
            ]
        ].rename(
            columns={
                "prob_male":
                    col
            }
        )

        if merged is None:
            merged = this

        else:
            this = this.rename(
                columns={
                    "sex_true_binary":
                        f"sex_seed{seed}"
                }
            )

            merged = merged.merge(
                this,
                on=[
                    "patient_id",
                    "ecg_id",
                ],
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
                    "Seed prediction label mismatch."
                )

            merged = merged.drop(
                columns=[
                    f"sex_seed{seed}"
                ]
            )

    prob_cols = [
        f"prob_seed{x}"
        for x in ALLOWED_SEEDS
    ]

    merged[
        "prob_ensemble"
    ] = merged[
        prob_cols
    ].mean(
        axis=1
    )

    merged[
        "strategy"
    ] = strategy

    merged[
        "train_fraction"
    ] = fraction

    return merged


def bootstrap_metrics(
    aligned,
    columns,
):
    y = aligned[
        "sex_true_binary"
    ].to_numpy(
        dtype=int
    )

    point = {}

    for name, col in columns.items():
        p = aligned[
            col
        ].to_numpy(
            dtype=float
        )

        point[
            name
        ] = {
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

            "Brier":
                float(
                    brier_score_loss(
                        y,
                        p,
                    )
                ),
        }

    rng = np.random.default_rng(
        42
    )

    rows = []
    n = len(
        y
    )

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

        if np.unique(
            yy
        ).size < 2:
            continue

        row = {
            "bootstrap_id":
                b
                +
                1,
        }

        for name, col in columns.items():
            pp = aligned[
                col
            ].to_numpy(
                dtype=float
            )[
                idx
            ]

            row[
                f"{name}__AUROC"
            ] = roc_auc_score(
                yy,
                pp,
            )

            row[
                f"{name}__AUPRC"
            ] = average_precision_score(
                yy,
                pp,
            )

            row[
                f"{name}__Brier"
            ] = brier_score_loss(
                yy,
                pp,
            )

        rows.append(
            row
        )

        if (
            b
            +
            1
        ) % 100 == 0:
            print(
                f"bootstrap {b+1:,}/{N_BOOT:,}"
            )

    return (
        point,
        pd.DataFrame(
            rows
        ),
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
        float(
            p
        ),
        1.0
        /
        (
            len(
                delta
            )
            +
            1
        ),
    )


def aggregate():
    print(
        "=" * 120
    )

    print(
        "PTB-XL FINAL VCG-GEOMETRY RESNET DATA EFFICIENCY — AGGREGATE"
    )

    print(
        "=" * 120
    )

    all_results = load_all_run_results()

    all_results.to_csv(
        FINAL_ROOT
        /
        "PTBXL_VCG_DataEfficiency_AllRuns.csv",
        index=False,
    )

    summary = (
        all_results
        .groupby(
            [
                "train_fraction",
                "strategy",
            ],
            as_index=False,
        )
        .agg(
            train_N=(
                "train_N",
                "mean",
            ),

            AUROC_mean=(
                "test_AUROC",
                "mean",
            ),

            AUROC_SD=(
                "test_AUROC",
                "std",
            ),

            AUPRC_mean=(
                "test_AUPRC",
                "mean",
            ),

            AUPRC_SD=(
                "test_AUPRC",
                "std",
            ),

            F1_mean=(
                "test_F1",
                "mean",
            ),

            F1_SD=(
                "test_F1",
                "std",
            ),

            Brier_mean=(
                "test_Brier",
                "mean",
            ),

            Brier_SD=(
                "test_Brier",
                "std",
            ),
        )
    )

    summary.to_csv(
        FINAL_ROOT
        /
        "PTBXL_VCG_DataEfficiency_MeanSD.csv",
        index=False,
    )

    print(
        "\nMEAN ± SD ACROSS SEEDS"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    ensembles = {}

    strategies = [
        "PTB-XL-only VCG-Geometry ResNet",
        "MIMIC→WCMC→PTB-XL VCG-Geometry ResNet",
    ]

    for fraction in ALLOWED_FRACTIONS:
        for strategy in strategies:
            pct = int(
                round(
                    fraction
                    *
                    100
                )
            )

            key = (
                pct,
                strategy,
            )

            ensembles[
                key
            ] = seed_ensemble_prediction(
                fraction,
                strategy,
            )

            ensembles[
                key
            ].to_csv(
                FINAL_ROOT
                /
                (
                    f"PTBXL_VCG_"
                    f"{fraction_tag(fraction)}_"
                    f"{'local' if 'only' in strategy else 'transfer'}_"
                    f"3SeedEnsemblePredictions.csv"
                ),
                index=False,
            )

    ci_rows = []
    comparison_rows = []

    for fraction in ALLOWED_FRACTIONS:
        pct = int(
            round(
                fraction
                *
                100
            )
        )

        local = ensembles[
            (
                pct,
                strategies[
                    0
                ],
            )
        ][
            [
                "patient_id",
                "ecg_id",
                "sex_true_binary",
                "prob_ensemble",
            ]
        ].rename(
            columns={
                "prob_ensemble":
                    "local"
            }
        )

        transfer = ensembles[
            (
                pct,
                strategies[
                    1
                ],
            )
        ][
            [
                "patient_id",
                "ecg_id",
                "sex_true_binary",
                "prob_ensemble",
            ]
        ].rename(
            columns={
                "sex_true_binary":
                    "sex_transfer",

                "prob_ensemble":
                    "transfer",
            }
        )

        aligned = local.merge(
            transfer,
            on=[
                "patient_id",
                "ecg_id",
            ],
            validate="one_to_one",
        )

        if not np.array_equal(
            aligned[
                "sex_true_binary"
            ].to_numpy(
                dtype=int
            ),
            aligned[
                "sex_transfer"
            ].to_numpy(
                dtype=int
            ),
        ):
            raise RuntimeError(
                "Local/transfer labels mismatch."
            )

        aligned = aligned.drop(
            columns=[
                "sex_transfer"
            ]
        )

        point, boot = bootstrap_metrics(
            aligned,
            {
                "local":
                    "local",

                "transfer":
                    "transfer",
            },
        )

        boot.to_csv(
            FINAL_ROOT
            /
            f"PTBXL_VCG_{pct:03d}pct_PairedBootstrap.csv",
            index=False,
        )

        for model in [
            "local",
            "transfer",
        ]:
            strategy = (
                strategies[
                    0
                ]
                if model == "local"
                else
                strategies[
                    1
                ]
            )

            row = {
                "train_percent":
                    pct,

                "strategy":
                    strategy,
            }

            for metric in [
                "AUROC",
                "AUPRC",
                "Brier",
            ]:
                vals = boot[
                    f"{model}__{metric}"
                ].to_numpy(
                    dtype=float
                )

                row[
                    metric
                ] = float(
                    point[
                        model
                    ][
                        metric
                    ]
                )

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

            ci_rows.append(
                row
            )

        for metric in [
            "AUROC",
            "AUPRC",
            "Brier",
        ]:
            delta = (
                boot[
                    f"transfer__{metric}"
                ].to_numpy(
                    dtype=float
                )
                -
                boot[
                    f"local__{metric}"
                ].to_numpy(
                    dtype=float
                )
            )

            comparison_rows.append(
                {
                    "comparison_type":
                        "same-fraction transfer vs local",

                    "train_percent":
                        pct,

                    "metric":
                        metric,

                    "A":
                        strategies[
                            1
                        ],

                    "B":
                        strategies[
                            0
                        ],

                    "A_point":
                        float(
                            point[
                                "transfer"
                            ][
                                metric
                            ]
                        ),

                    "B_point":
                        float(
                            point[
                                "local"
                            ][
                                metric
                            ]
                        ),

                    "delta_A_minus_B":
                        float(
                            point[
                                "transfer"
                            ][
                                metric
                            ]
                            -
                            point[
                                "local"
                            ][
                                metric
                            ]
                        ),

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

    ci_df = pd.DataFrame(
        ci_rows
    )

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    # PRIMARY: 10% transfer vs 100% local-only, SAME FINAL ARCHITECTURE.
    transfer10 = ensembles[
        (
            10,
            strategies[
                1
            ],
        )
    ][
        [
            "patient_id",
            "ecg_id",
            "sex_true_binary",
            "prob_ensemble",
        ]
    ].rename(
        columns={
            "prob_ensemble":
                "transfer10"
        }
    )

    local100 = ensembles[
        (
            100,
            strategies[
                0
            ],
        )
    ][
        [
            "patient_id",
            "ecg_id",
            "sex_true_binary",
            "prob_ensemble",
        ]
    ].rename(
        columns={
            "sex_true_binary":
                "sex_local100",

            "prob_ensemble":
                "local100",
        }
    )

    aligned = transfer10.merge(
        local100,
        on=[
            "patient_id",
            "ecg_id",
        ],
        validate="one_to_one",
    )

    if not np.array_equal(
        aligned[
            "sex_true_binary"
        ].to_numpy(
            dtype=int
        ),
        aligned[
            "sex_local100"
        ].to_numpy(
            dtype=int
        ),
    ):
        raise RuntimeError(
            "Primary comparison labels mismatch."
        )

    aligned = aligned.drop(
        columns=[
            "sex_local100"
        ]
    )

    point, boot = bootstrap_metrics(
        aligned,
        {
            "transfer10":
                "transfer10",

            "local100":
                "local100",
        },
    )

    boot.to_csv(
        FINAL_ROOT
        /
        "PTBXL_VCG_PRIMARY_10pctTransfer_vs_100pctLocal_Bootstrap.csv",
        index=False,
    )

    primary_rows = []

    for metric in [
        "AUROC",
        "AUPRC",
        "Brier",
    ]:
        delta = (
            boot[
                f"transfer10__{metric}"
            ].to_numpy(
                dtype=float
            )
            -
            boot[
                f"local100__{metric}"
            ].to_numpy(
                dtype=float
            )
        )

        primary_rows.append(
            {
                "comparison_type":
                    "PRIMARY data-efficiency",

                "comparison":
                    (
                        "10% MIMIC→WCMC→PTB-XL VCG-Geometry ResNet "
                        "vs 100% PTB-XL-only VCG-Geometry ResNet"
                    ),

                "metric":
                    metric,

                "A_point":
                    float(
                        point[
                            "transfer10"
                        ][
                            metric
                        ]
                    ),

                "B_point":
                    float(
                        point[
                            "local100"
                        ][
                            metric
                        ]
                    ),

                "delta_A_minus_B":
                    float(
                        point[
                            "transfer10"
                        ][
                            metric
                        ]
                        -
                        point[
                            "local100"
                        ][
                            metric
                        ]
                    ),

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

    primary_df = pd.DataFrame(
        primary_rows
    )

    comparison_df = pd.concat(
        [
            primary_df,
            comparison_df,
        ],
        ignore_index=True,
    )

    ci_df.to_csv(
        FINAL_ROOT
        /
        "PTBXL_VCG_DataEfficiency_SeedEnsemble_Performance_With95CI.csv",
        index=False,
    )

    comparison_df.to_csv(
        FINAL_ROOT
        /
        "PTBXL_VCG_DataEfficiency_PairedComparisons.csv",
        index=False,
    )

    # ----------------------------------------------------------------------
    # SECONDARY: matched VCG-transfer vs completed ResNet-transfer ensembles.
    # ----------------------------------------------------------------------
    arch_rows = []

    for fraction in ALLOWED_FRACTIONS:
        pct = int(
            round(
                fraction
                *
                100
            )
        )

        vcg = ensembles[
            (
                pct,
                strategies[
                    1
                ],
            )
        ][
            [
                "patient_id",
                "ecg_id",
                "sex_true_binary",
                "prob_ensemble",
            ]
        ].rename(
            columns={
                "prob_ensemble":
                    "vcg"
            }
        )

        res_path = (
            RESNET_FINAL_ROOT
            /
            (
                f"PTBXL_ResNet_"
                f"{fraction_tag(fraction)}_"
                f"transfer_3SeedEnsemblePredictions.csv"
            )
        )

        if not res_path.exists():
            print(
                "WARNING: ResNet ensemble missing, skipping architecture comparison:",
                res_path,
            )

            continue

        res = pd.read_csv(
            res_path,
            low_memory=False,
        )

        res[
            "patient_id"
        ] = res[
            "patient_id"
        ].astype(
            str
        )

        res[
            "ecg_id"
        ] = res[
            "ecg_id"
        ].astype(
            str
        )

        res = res[
            [
                "patient_id",
                "ecg_id",
                "sex_true_binary",
                "prob_ensemble",
            ]
        ].rename(
            columns={
                "sex_true_binary":
                    "sex_resnet",

                "prob_ensemble":
                    "resnet",
            }
        )

        aligned_arch = vcg.merge(
            res,
            on=[
                "patient_id",
                "ecg_id",
            ],
            validate="one_to_one",
        )

        if not np.array_equal(
            aligned_arch[
                "sex_true_binary"
            ].to_numpy(
                dtype=int
            ),
            aligned_arch[
                "sex_resnet"
            ].to_numpy(
                dtype=int
            ),
        ):
            raise RuntimeError(
                "VCG/ResNet architecture comparison label mismatch."
            )

        aligned_arch = aligned_arch.drop(
            columns=[
                "sex_resnet"
            ]
        )

        arch_point, arch_boot = bootstrap_metrics(
            aligned_arch,
            {
                "VCG":
                    "vcg",

                "ResNet":
                    "resnet",
            },
        )

        for metric in [
            "AUROC",
            "AUPRC",
            "Brier",
        ]:
            delta = (
                arch_boot[
                    f"VCG__{metric}"
                ].to_numpy(
                    dtype=float
                )
                -
                arch_boot[
                    f"ResNet__{metric}"
                ].to_numpy(
                    dtype=float
                )
            )

            arch_rows.append(
                {
                    "train_percent":
                        pct,

                    "comparison":
                        "VCG transfer minus ResNet transfer",

                    "metric":
                        metric,

                    "VCG_point":
                        arch_point[
                            "VCG"
                        ][
                            metric
                        ],

                    "ResNet_point":
                        arch_point[
                            "ResNet"
                        ][
                            metric
                        ],

                    "delta_VCG_minus_ResNet":
                        (
                            arch_point[
                                "VCG"
                            ][
                                metric
                            ]
                            -
                            arch_point[
                                "ResNet"
                            ][
                                metric
                            ]
                        ),

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

    pd.DataFrame(
        arch_rows
    ).to_csv(
        FINAL_ROOT
        /
        "PTBXL_VCG_vs_ResNet_Transfer_DataEfficiency_PairedComparisons.csv",
        index=False,
    )

    print(
        "\n"
        +
        "=" * 120
    )

    print(
        "SEED-ENSEMBLE PERFORMANCE"
    )

    print(
        "=" * 120
    )

    print(
        ci_df.to_string(
            index=False
        )
    )

    print(
        "\n"
        +
        "=" * 120
    )

    print(
        "PRIMARY: 10% VCG TRANSFER vs 100% VCG LOCAL-ONLY"
    )

    print(
        "=" * 120
    )

    print(
        primary_df.to_string(
            index=False
        )
    )

    if arch_rows:
        print(
            "\n"
            +
            "=" * 120
        )

        print(
            "SECONDARY: TRANSFERRED VCG vs TRANSFERRED RESNET"
        )

        print(
            "=" * 120
        )

        print(
            pd.DataFrame(
                arch_rows
            ).to_string(
                index=False
            )
        )

    print(
        "\nOutput:",
        FINAL_ROOT,
    )


# ==========================================================================================
# MAIN
# ==========================================================================================

if __name__ == "__main__":
    if MODE == "RUN":
        run_one()

    elif MODE == "AGGREGATE":
        aggregate()

    else:
        raise ValueError(
            f"Unknown PTB_VCG_MODE={MODE}"
        )
