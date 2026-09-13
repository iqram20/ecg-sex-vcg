#!/usr/bin/env python
# ==========================================================================================
# ECG-SEX — VCG-GEOMETRY RESNET 5-SEED ZERO-SHOT EXTERNAL CONFIRMATION
# ==========================================================================================
#
# MODES
# -----
#   VCGEXT_MODE=PTBXL
#   VCGEXT_MODE=CHAPMAN
#   VCGEXT_MODE=FINAL
#
# DESIGN
# ------
# - Final VCG-Geometry ResNet architecture is FROZEN.
# - Five completed WCMC VCG checkpoints:
#       seeds 11, 22, 33, 42, 55
# - Five completed matched ResNet checkpoints:
#       seeds 11, 22, 33, 42, 55
# - Equal-probability 5-seed ensembles.
# - Frozen thresholds are read from the completed WCMC locked-test summary:
#       VCG threshold    ~ 0.484619
#       ResNet threshold ~ 0.453711
# - ZERO external training.
# - ZERO external recalibration.
# - ZERO external threshold optimization.
# - External cohorts:
#       PTB-XL full locked cohort
#       Chapman/Shaoxing-Ningbo full locked cohort
# - FINAL performs 2,000 paired bootstrap replicates:
#       VCG-Geometry ResNet minus ResNet
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
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast

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
)

warnings.filterwarnings("ignore")


# ==========================================================================================
# CONFIG
# ==========================================================================================

MODE = os.environ.get(
    "VCGEXT_MODE",
    "PTBXL",
).strip().upper()

if MODE not in {
    "PTBXL",
    "CHAPMAN",
    "FINAL",
}:
    raise RuntimeError(
        f"VCGEXT_MODE must be PTBXL, CHAPMAN, or FINAL; got {MODE}"
    )

SEEDS = [
    11,
    22,
    33,
    42,
    55,
]

TARGET_LEN = 2500
N_LEADS = 12
IN_CH = 24
DROPOUT = 0.08

DOWER_LEAD_INDICES = [
    0,   # I
    1,   # II
    6,   # V1
    7,   # V2
    8,   # V3
    9,   # V4
    10,  # V5
    11,  # V6
]

VCG_DIM = 192
N_TRAJECTORY_SLOTS = 3

MAX_DOWER_DELTA = 0.03

BATCH_SIZE = int(
    os.environ.get(
        "VCGEXT_BATCH_SIZE",
        "128",
    )
)

NUM_WORKERS = int(
    os.environ.get(
        "VCGEXT_NUM_WORKERS",
        "8",
    )
)

N_BOOT = int(
    os.environ.get(
        "VCGEXT_N_BOOT",
        "2000",
    )
)

LEADS = [
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

VCG_FINAL_ROOT = (
    WCMC_ROOT
    /
    "19_vcg_geometry_resnet_5seed_locked_wcmc"
)

VCG_SEED_ROOT = (
    VCG_FINAL_ROOT
    /
    "seeds"
)

WCMC_FINAL_SUMMARY = (
    VCG_FINAL_ROOT
    /
    "final_5seed_ensemble_bootstrap"
    /
    "WCMC_VCG_vs_ResNet_5SeedEnsemble_FinalSummary.json"
)

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
    "20_vcg_geometry_resnet_external_confirmation"
)

PTB_OUT = (
    OUT_ROOT
    /
    "ptbxl"
)

CHAPMAN_OUT = (
    OUT_ROOT
    /
    "chapman_ningbo"
)

FINAL_OUT = (
    OUT_ROOT
    /
    "final_paired_bootstrap"
)

# Create directories only when actually running on Athena.
if MODE in {
    "PTBXL",
    "CHAPMAN",
    "FINAL",
}:
    # This block is harmless on cluster; local smoke-test imports set a special env below.
    if os.environ.get(
        "VCGEXT_SKIP_MKDIR",
        "0",
    ) != "1":
        for d in [
            OUT_ROOT,
            PTB_OUT,
            CHAPMAN_OUT,
            FINAL_OUT,
        ]:
            d.mkdir(
                parents=True,
                exist_ok=True,
            )


# ---------------- PTB-XL ----------------

PTB_COHORT = (
    RUN_DIR
    /
    "ptbxl_ecgsex_from_scratch"
    /
    "01_cohort"
    /
    "PTBXL_ECGSex_Locked_OneECGPerPatient_AgeQC.csv"
)

PTB_DATA_ROOT = Path(
    "/athena/madelab/scratch/iqh4001/ECG/PTB-XL-ECG/Data"
)


# ---------------- Chapman / Ningbo ----------------

CHAPMAN_COHORT = (
    RUN_DIR
    /
    "chapman_ningbo_ecgsex_from_scratch"
    /
    "01_cohort"
    /
    "ChapmanNingbo_ECGSex_Locked_Adult_KnownSex.csv"
)

CHAPMAN_DATA_ROOT = Path(
    "/athena/madelab/scratch/iqh4001/ECG/Arrhymia_ECG/Data/"
    "physionet.org/files/ecg-arrhythmia/1.0.0"
)


# ==========================================================================================
# REPRODUCIBILITY
# ==========================================================================================

def set_seed(
    seed=42,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True

        try:
            torch.set_float32_matmul_precision(
                "high"
            )
        except Exception:
            pass


set_seed(42)


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

        for lead_idx in range(N_LEADS):
            out[lead_idx] = np.interp(
                new_grid,
                old_grid,
                x[lead_idx],
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
    # Exact ResNet representation:
    # 12 per-lead z scores + 12 first derivatives.
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

    x24 = np.concatenate(
        [
            z,
            dz,
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
    # Preserve inter-lead relative amplitudes:
    # per-lead DC removal + ONE recording-level RMS scale.
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
# MODEL DEFINITIONS
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
            f"Expected 13 VCG channels; got {features.shape}"
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

        self.gate = nn.Linear(
            512,
            1,
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

        logit = (
            base_logit
            +
            gate
            *
            delta
        )

        if return_components:
            return {
                "logit":
                    logit,

                "base_logit":
                    base_logit,

                "gate":
                    gate,

                "delta":
                    delta,
            }

        return logit


# ==========================================================================================
# CHECKPOINT LOADERS
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

    clean = {}

    for key, value in state.items():
        key = str(key)

        if key.startswith(
            "module."
        ):
            key = key[
                len("module.") :
            ]

        clean[key] = value

    return clean


def load_vcg_model(
    seed,
):
    path = (
        VCG_SEED_ROOT
        /
        f"seed_{seed}"
        /
        "best_VCGGeometryResNet_WCMC.pt"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing VCG seed checkpoint:\n{path}"
        )

    model = VCGGeometryResNet().to(
        DEVICE
    )

    checkpoint = torch.load(
        path,
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

    model.eval()

    return model


def load_resnet_model(
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
            f"Missing ResNet seed checkpoint:\n{path}"
        )

    checkpoint = torch.load(
        path,
        map_location=DEVICE,
        weights_only=False,
    )

    source = extract_state_dict(
        checkpoint
    )

    model = ResNet1D().to(
        DEVICE
    )

    target = model.state_dict()

    transferred = 0

    for key, value in source.items():
        target_key = None

        if key.startswith(
            "stem."
        ):
            target_key = key

        elif key.startswith(
            "block1."
        ):
            target_key = key

        elif key.startswith(
            "block2."
        ):
            target_key = key

        elif key.startswith(
            "block3."
        ):
            target_key = key

        elif key.startswith(
            "block4."
        ):
            target_key = key

        elif key.startswith(
            "block5."
        ):
            target_key = key

        elif key.startswith(
            "base_head."
        ):
            target_key = (
                "head."
                +
                key[
                    len("base_head.") :
                ]
            )

        elif key.startswith(
            "head."
        ):
            target_key = key

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
            f"Only {transferred} tensors loaded from ResNet seed {seed}."
        )

    model.eval()

    return model


def load_frozen_thresholds():
    if not WCMC_FINAL_SUMMARY.exists():
        raise FileNotFoundError(
            f"Missing WCMC final summary:\n{WCMC_FINAL_SUMMARY}"
        )

    with open(
        WCMC_FINAL_SUMMARY,
        "r",
    ) as f:
        summary = json.load(f)

    vcg_threshold = float(
        summary[
            "VCG_validation_threshold"
        ]
    )

    resnet_threshold = float(
        summary[
            "ResNet_validation_threshold"
        ]
    )

    return (
        vcg_threshold,
        resnet_threshold,
        summary,
    )


# ==========================================================================================
# METRICS
# ==========================================================================================

def calc_metrics(
    y,
    p,
    threshold,
):
    y = np.asarray(
        y,
        dtype=int,
    )

    p = np.asarray(
        p,
        dtype=float,
    )

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
# COMMON EXTERNAL COLLATE
# ==========================================================================================

def external_collate(
    batch,
):
    valid = [
        item
        for item in batch
        if item[
            "x_base"
        ] is not None
    ]

    invalid = [
        item
        for item in batch
        if item[
            "x_base"
        ] is None
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
                    item[
                        "x_base"
                    ]
                    for item in valid
                ],
                dim=0,
            ),

        "x_vcg8":
            torch.stack(
                [
                    item[
                        "x_vcg8"
                    ]
                    for item in valid
                ],
                dim=0,
            ),

        "valid":
            valid,

        "invalid":
            invalid,
    }


# ==========================================================================================
# PTB-XL
# ==========================================================================================

def normalize_lead_name(
    name,
):
    s = str(name).strip()
    upper = s.upper()

    mapping = {
        "AVR":
            "aVR",

        "AVL":
            "aVL",

        "AVF":
            "aVF",
    }

    if upper in mapping:
        return mapping[upper]

    if upper.startswith("V"):
        return upper

    return upper


def record_exists(
    path,
):
    path = str(path)

    return (
        Path(
            path + ".hea"
        ).is_file()
        and
        Path(
            path + ".dat"
        ).is_file()
    )


def resolve_ptb_path(
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
                row[col]
            )
            and
            str(
                row[col]
            ).strip()
        ):
            p = str(
                row[col]
            ).strip()

            if (
                p.endswith(".hea")
                or
                p.endswith(".dat")
            ):
                p = p.rsplit(
                    ".",
                    1,
                )[0]

            if record_exists(p):
                return p

    if (
        "filename_hr"
        not in row.index
        or
        pd.isna(
            row["filename_hr"]
        )
    ):
        raise ValueError(
            "PTB-XL row has no usable waveform path."
        )

    rel = str(
        row["filename_hr"]
    ).strip()

    if (
        rel.endswith(".hea")
        or
        rel.endswith(".dat")
    ):
        rel = rel.rsplit(
            ".",
            1,
        )[0]

    candidates = [
        PTB_DATA_ROOT
        /
        rel
    ]

    if rel.startswith(
        "records500/"
    ):
        candidates.append(
            PTB_DATA_ROOT
            /
            rel[
                len("records500/") :
            ]
        )

    parts = Path(rel).parts

    if len(parts) >= 2:
        candidates.append(
            PTB_DATA_ROOT
            /
            parts[-2]
            /
            parts[-1]
        )

    for candidate in candidates:
        if record_exists(
            candidate
        ):
            return str(candidate)

    raise FileNotFoundError(
        f"Could not resolve PTB-XL waveform: {rel}"
    )


def load_ptb_waveform(
    record_path,
):
    import wfdb

    signal, fields = wfdb.rdsamp(
        str(record_path)
    )

    signal = np.asarray(
        signal,
        dtype=np.float32,
    )

    names = fields.get(
        "sig_name",
        None,
    )

    if names is None:
        raise ValueError(
            "PTB record missing lead names."
        )

    normalized = [
        normalize_lead_name(x)
        for x in names
    ]

    missing = [
        lead
        for lead in LEADS
        if lead not in normalized
    ]

    if missing:
        raise ValueError(
            f"Missing PTB leads: {missing}"
        )

    indices = [
        normalized.index(lead)
        for lead in LEADS
    ]

    return signal[
        :,
        indices
    ].T


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
        return len(self.df)

    def __getitem__(
        self,
        idx,
    ):
        row = self.df.iloc[idx]

        patient_key = str(
            row["patient_id"]
        )

        ecg_key = str(
            row["ecg_id"]
        )

        try:
            path = resolve_ptb_path(
                row
            )

            x12 = orient_and_resample(
                load_ptb_waveform(
                    path
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
                    int(
                        row["sex_binary"]
                    ),

                "patient_key":
                    patient_key,

                "ecg_key":
                    ecg_key,

                "record_path":
                    str(path),

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
                    int(
                        row["sex_binary"]
                    ),

                "patient_key":
                    patient_key,

                "ecg_key":
                    ecg_key,

                "record_path":
                    "",

                "error":
                    repr(exc),
            }


def load_ptb_cohort():
    if not PTB_COHORT.exists():
        raise FileNotFoundError(
            PTB_COHORT
        )

    df = pd.read_csv(
        PTB_COHORT,
        low_memory=False,
    )

    required = [
        "patient_id",
        "ecg_id",
        "sex_binary",
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"PTB cohort missing columns {missing}"
        )

    df[
        "sex_binary"
    ] = pd.to_numeric(
        df[
            "sex_binary"
        ],
        errors="raise",
    ).astype(int)

    return df.reset_index(
        drop=True
    )


# ==========================================================================================
# CHAPMAN / SHAOXING-NINGBO
# ==========================================================================================

CHAPMAN_PATH_COLUMNS = [
    "mat_path",
    "waveform_path",
    "ecg_path",
    "path",
    "file_path",
    "filename",
]


def first_existing_column(
    df,
    candidates,
    required=True,
):
    lower = {
        str(c).lower():
            c
        for c in df.columns
    }

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

        if candidate.lower() in lower:
            return lower[
                candidate.lower()
            ]

    if required:
        raise RuntimeError(
            f"Could not resolve column from {candidates}. "
            f"Available={list(df.columns)}"
        )

    return None


def resolve_chapman_path(
    row,
    ecg_key,
):
    for col in CHAPMAN_PATH_COLUMNS:
        if (
            col in row.index
            and
            pd.notna(
                row[col]
            )
        ):
            raw = str(
                row[col]
            ).strip()

            if not raw:
                continue

            for candidate in [
                Path(raw),
                CHAPMAN_DATA_ROOT
                /
                raw,
            ]:
                if candidate.exists():
                    return str(candidate)

    stem = Path(
        str(ecg_key)
    ).stem

    matches = list(
        CHAPMAN_DATA_ROOT.glob(
            f"**/{stem}.mat"
        )
    )

    if len(matches) == 1:
        return str(
            matches[0]
        )

    raise FileNotFoundError(
        f"Could not resolve Chapman waveform for ECG {ecg_key}"
    )


def load_chapman_waveform(
    path,
):
    from scipy.io import loadmat

    obj = loadmat(
        path,
        squeeze_me=False,
        struct_as_record=False,
    )

    if "val" in obj:
        arr = np.asarray(
            obj["val"]
        )

    else:
        candidates = []

        for key, value in obj.items():
            if key.startswith("__"):
                continue

            arr_candidate = np.asarray(
                value
            )

            if (
                np.issubdtype(
                    arr_candidate.dtype,
                    np.number,
                )
                and
                arr_candidate.ndim == 2
                and
                12 in arr_candidate.shape
            ):
                candidates.append(
                    (
                        key,
                        arr_candidate,
                    )
                )

        if len(candidates) != 1:
            raise ValueError(
                "Could not uniquely identify Chapman ECG matrix. "
                f"Candidates={[(k, a.shape) for k, a in candidates]}"
            )

        arr = candidates[0][1]

    return np.asarray(
        arr,
        dtype=np.float32,
    )


class ChapmanDataset(
    Dataset
):

    def __init__(
        self,
        df,
        patient_col,
        ecg_col,
        sex_col,
    ):
        self.df = df.reset_index(
            drop=True
        ).copy()

        self.patient_col = patient_col
        self.ecg_col = ecg_col
        self.sex_col = sex_col

    def __len__(
        self,
    ):
        return len(self.df)

    def __getitem__(
        self,
        idx,
    ):
        row = self.df.iloc[idx]

        patient_key = str(
            row[
                self.patient_col
            ]
        )

        ecg_key = str(
            row[
                self.ecg_col
            ]
        )

        try:
            path = resolve_chapman_path(
                row,
                ecg_key,
            )

            x12 = orient_and_resample(
                load_chapman_waveform(
                    path
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
                    int(
                        row[
                            self.sex_col
                        ]
                    ),

                "patient_key":
                    patient_key,

                "ecg_key":
                    ecg_key,

                "record_path":
                    str(path),

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
                    int(
                        row[
                            self.sex_col
                        ]
                    ),

                "patient_key":
                    patient_key,

                "ecg_key":
                    ecg_key,

                "record_path":
                    "",

                "error":
                    repr(exc),
            }


def load_chapman_cohort():
    if not CHAPMAN_COHORT.exists():
        raise FileNotFoundError(
            CHAPMAN_COHORT
        )

    df = pd.read_csv(
        CHAPMAN_COHORT,
        low_memory=False,
    )

    sex_col = first_existing_column(
        df,
        [
            "sex_binary",
            "sex_true_binary",
            "target",
            "label",
            "y",
        ],
    )

    patient_col = first_existing_column(
        df,
        [
            "patient_id",
            "patient_key",
            "subject_id",
            "patient",
            "record_id",
        ],
        required=False,
    )

    ecg_col = first_existing_column(
        df,
        [
            "ecg_id",
            "ecg_key",
            "record_id",
            "record",
            "ecg",
            "filename",
        ],
        required=False,
    )

    if patient_col is None:
        df[
            "_patient_id"
        ] = np.arange(
            len(df)
        ).astype(str)

        patient_col = "_patient_id"

    if ecg_col is None:
        path_col = first_existing_column(
            df,
            CHAPMAN_PATH_COLUMNS,
            required=False,
        )

        if path_col is None:
            raise RuntimeError(
                "Could not resolve Chapman ECG ID."
            )

        df[
            "_ecg_id"
        ] = (
            df[path_col]
            .astype(str)
            .map(
                lambda x:
                    Path(x).stem
            )
        )

        ecg_col = "_ecg_id"

    df[
        sex_col
    ] = pd.to_numeric(
        df[
            sex_col
        ],
        errors="raise",
    ).astype(int)

    return (
        df.reset_index(
            drop=True
        ),
        patient_col,
        ecg_col,
        sex_col,
    )


# ==========================================================================================
# INFERENCE
# ==========================================================================================

def make_loader(
    dataset,
):
    kwargs = {
        "dataset":
            dataset,

        "batch_size":
            BATCH_SIZE,

        "shuffle":
            False,

        "num_workers":
            NUM_WORKERS,

        "pin_memory":
            DEVICE.type == "cuda",

        "drop_last":
            False,

        "collate_fn":
            external_collate,
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


def build_models():
    vcg_models = {
        seed:
            load_vcg_model(seed)
        for seed in SEEDS
    }

    resnet_models = {
        seed:
            load_resnet_model(seed)
        for seed in SEEDS
    }

    return (
        vcg_models,
        resnet_models,
    )


def run_external(
    dataset_name,
):
    if DEVICE.type != "cuda":
        raise RuntimeError(
            "External inference requires CUDA."
        )

    (
        vcg_threshold,
        resnet_threshold,
        wcmc_summary,
    ) = load_frozen_thresholds()

    if dataset_name == "PTBXL":
        cohort = load_ptb_cohort()

        dataset = PTBXLDataset(
            cohort
        )

        out_dir = PTB_OUT

        safe = "PTBXL"

        display_name = "PTB-XL"

    elif dataset_name == "CHAPMAN":
        (
            cohort,
            patient_col,
            ecg_col,
            sex_col,
        ) = load_chapman_cohort()

        dataset = ChapmanDataset(
            cohort,
            patient_col,
            ecg_col,
            sex_col,
        )

        out_dir = CHAPMAN_OUT

        safe = "ChapmanNingbo"

        display_name = "Chapman/Ningbo"

    else:
        raise ValueError(
            dataset_name
        )

    (
        vcg_models,
        resnet_models,
    ) = build_models()

    loader = make_loader(
        dataset
    )

    rows = []
    failures = []

    print(
        "=" * 120
    )

    print(
        "VCG-GEOMETRY RESNET 5-SEED ZERO-SHOT EXTERNAL CONFIRMATION"
    )

    print(
        "=" * 120
    )

    print(
        "Dataset:",
        display_name,
    )

    print(
        "Cohort N:",
        f"{len(dataset):,}",
    )

    print(
        "Seeds:",
        SEEDS,
    )

    print(
        "Frozen VCG threshold:",
        f"{vcg_threshold:.6f}",
    )

    print(
        "Frozen ResNet threshold:",
        f"{resnet_threshold:.6f}",
    )

    print(
        "External training/recalibration:",
        "NONE",
    )

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    with torch.no_grad():
        for batch_idx, batch in enumerate(
            loader,
            start=1,
        ):
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

                        "record_path":
                            bad[
                                "record_path"
                            ],

                        "error":
                            bad[
                                "error"
                            ],
                    }
                )

            if batch[
                "x_base"
            ] is None:
                continue

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

            vcg_probs = {}
            res_probs = {}

            with autocast(
                enabled=True
            ):
                for seed in SEEDS:
                    vcg_logits = vcg_models[
                        seed
                    ](
                        x_base,
                        x_vcg8,
                    )

                    vcg_probs[
                        seed
                    ] = (
                        torch.sigmoid(
                            vcg_logits
                        )
                        .float()
                        .cpu()
                        .numpy()
                    )

                    res_logits = resnet_models[
                        seed
                    ](
                        x_base
                    )

                    res_probs[
                        seed
                    ] = (
                        torch.sigmoid(
                            res_logits
                        )
                        .float()
                        .cpu()
                        .numpy()
                    )

            n_valid = len(
                batch[
                    "valid"
                ]
            )

            for i in range(
                n_valid
            ):
                meta = batch[
                    "valid"
                ][i]

                row = {
                    "patient_key":
                        str(
                            meta[
                                "patient_key"
                            ]
                        ),

                    "ecg_key":
                        str(
                            meta[
                                "ecg_key"
                            ]
                        ),

                    "sex_true_binary":
                        int(
                            meta[
                                "y"
                            ]
                        ),

                    "record_path":
                        str(
                            meta[
                                "record_path"
                            ]
                        ),
                }

                for seed in SEEDS:
                    row[
                        f"VCG_prob_seed{seed}"
                    ] = float(
                        vcg_probs[
                            seed
                        ][i]
                    )

                    row[
                        f"ResNet_prob_seed{seed}"
                    ] = float(
                        res_probs[
                            seed
                        ][i]
                    )

                row[
                    "VCG_prob_5seed_ensemble"
                ] = float(
                    np.mean(
                        [
                            vcg_probs[
                                seed
                            ][i]
                            for seed in SEEDS
                        ]
                    )
                )

                row[
                    "ResNet_prob_5seed_ensemble"
                ] = float(
                    np.mean(
                        [
                            res_probs[
                                seed
                            ][i]
                            for seed in SEEDS
                        ]
                    )
                )

                rows.append(
                    row
                )

            if batch_idx % 25 == 0:
                print(
                    f"Batch {batch_idx:,}/{len(loader):,} | "
                    f"successful={len(rows):,} | "
                    f"failed={len(failures):,}"
                )

            del x_base
            del x_vcg8

    pred = pd.DataFrame(
        rows
    )

    failures_df = pd.DataFrame(
        failures
    )

    if pred.empty:
        raise RuntimeError(
            "No successful external predictions."
        )

    pred_file = (
        out_dir
        /
        f"{safe}_VCG_vs_ResNet_5Seed_ZeroShot_Predictions.csv"
    )

    failure_file = (
        out_dir
        /
        f"{safe}_VCG_vs_ResNet_5Seed_Failures.csv"
    )

    perf_file = (
        out_dir
        /
        f"{safe}_VCG_vs_ResNet_5Seed_ZeroShot_PointPerformance.csv"
    )

    pred.to_csv(
        pred_file,
        index=False,
    )

    failures_df.to_csv(
        failure_file,
        index=False,
    )

    y = pred[
        "sex_true_binary"
    ].to_numpy(
        dtype=int
    )

    p_vcg = pred[
        "VCG_prob_5seed_ensemble"
    ].to_numpy(
        dtype=float
    )

    p_res = pred[
        "ResNet_prob_5seed_ensemble"
    ].to_numpy(
        dtype=float
    )

    perf_rows = []

    for model_name, p, threshold in [
        (
            "VCG-Geometry ResNet 5-seed ensemble",
            p_vcg,
            vcg_threshold,
        ),
        (
            "ResNet 5-seed ensemble",
            p_res,
            resnet_threshold,
        ),
    ]:
        perf_rows.append(
            {
                "dataset":
                    display_name,

                "model":
                    model_name,

                "N":
                    int(
                        len(y)
                    ),

                "Female_N":
                    int(
                        np.sum(
                            y == 0
                        )
                    ),

                "Male_N":
                    int(
                        np.sum(
                            y == 1
                        )
                    ),

                "threshold":
                    float(
                        threshold
                    ),

                "threshold_source":
                    "Frozen WCMC validation 5-seed ensemble threshold",

                "external_training":
                    False,

                "external_recalibration":
                    False,

                **calc_metrics(
                    y,
                    p,
                    threshold,
                ),
            }
        )

    perf_df = pd.DataFrame(
        perf_rows
    )

    perf_df.to_csv(
        perf_file,
        index=False,
    )

    print(
        "\nPoint performance:"
    )

    print(
        perf_df.to_string(
            index=False
        )
    )

    print(
        "\nSaved predictions:",
        pred_file,
    )

    print(
        "Saved failures:",
        failure_file,
    )

    print(
        "Saved performance:",
        perf_file,
    )

    del vcg_models
    del resnet_models

    gc.collect()

    torch.cuda.empty_cache()


# ==========================================================================================
# FINAL PAIRED BOOTSTRAP
# ==========================================================================================

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


def bootstrap_dataset(
    pred,
    dataset_name,
    vcg_threshold,
    resnet_threshold,
):
    y = pred[
        "sex_true_binary"
    ].to_numpy(
        dtype=int
    )

    p_vcg = pred[
        "VCG_prob_5seed_ensemble"
    ].to_numpy(
        dtype=float
    )

    p_res = pred[
        "ResNet_prob_5seed_ensemble"
    ].to_numpy(
        dtype=float
    )

    point_vcg = calc_metrics(
        y,
        p_vcg,
        vcg_threshold,
    )

    point_res = calc_metrics(
        y,
        p_res,
        resnet_threshold,
    )

    rng = np.random.default_rng(
        42
    )

    rows = []

    n = len(y)

    for b in range(
        N_BOOT
    ):
        idx = rng.integers(
            0,
            n,
            n,
        )

        yy = y[idx]

        if np.unique(
            yy
        ).size < 2:
            continue

        mv = calc_metrics(
            yy,
            p_vcg[idx],
            vcg_threshold,
        )

        mr = calc_metrics(
            yy,
            p_res[idx],
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
            ] = mv[
                metric
            ]

            row[
                f"ResNet__{metric}"
            ] = mr[
                metric
            ]

            row[
                f"Delta_VCG_minus_ResNet__{metric}"
            ] = (
                mv[
                    metric
                ]
                -
                mr[
                    metric
                ]
            )

        rows.append(
            row
        )

        if (
            b + 1
        ) % 100 == 0:
            print(
                f"{dataset_name}: bootstrap {b+1:,}/{N_BOOT:,}"
            )

    boot = pd.DataFrame(
        rows
    )

    perf_rows = []

    for prefix, model_name, point in [
        (
            "VCG",
            "VCG-Geometry ResNet 5-seed ensemble",
            point_vcg,
        ),
        (
            "ResNet",
            "ResNet 5-seed ensemble",
            point_res,
        ),
    ]:
        row = {
            "dataset":
                dataset_name,

            "model":
                model_name,

            "N":
                len(y),

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

            row[
                metric
            ] = point[
                metric
            ]

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

        perf_rows.append(
            row
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
        delta = boot[
            f"Delta_VCG_minus_ResNet__{metric}"
        ].to_numpy(
            dtype=float
        )

        diff_rows.append(
            {
                "dataset":
                    dataset_name,

                "comparison":
                    "VCG-Geometry ResNet minus ResNet",

                "metric":
                    metric,

                "VCG_point":
                    point_vcg[
                        metric
                    ],

                "ResNet_point":
                    point_res[
                        metric
                    ],

                "delta_VCG_minus_ResNet":
                    (
                        point_vcg[
                            metric
                        ]
                        -
                        point_res[
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

    return (
        boot,
        pd.DataFrame(
            perf_rows
        ),
        pd.DataFrame(
            diff_rows
        ),
    )


def run_final():
    FINAL_OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        vcg_threshold,
        resnet_threshold,
        wcmc_summary,
    ) = load_frozen_thresholds()

    datasets = [
        (
            "PTB-XL",
            PTB_OUT
            /
            "PTBXL_VCG_vs_ResNet_5Seed_ZeroShot_Predictions.csv",
        ),
        (
            "Chapman/Ningbo",
            CHAPMAN_OUT
            /
            "ChapmanNingbo_VCG_vs_ResNet_5Seed_ZeroShot_Predictions.csv",
        ),
    ]

    all_perf = []
    all_diff = []

    for dataset_name, path in datasets:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing zero-shot predictions for {dataset_name}:\n{path}"
            )

        pred = pd.read_csv(
            path,
            low_memory=False,
        )

        required = [
            "patient_key",
            "ecg_key",
            "sex_true_binary",
            "VCG_prob_5seed_ensemble",
            "ResNet_prob_5seed_ensemble",
        ]

        missing = [
            c
            for c in required
            if c not in pred.columns
        ]

        if missing:
            raise RuntimeError(
                f"{dataset_name} predictions missing {missing}"
            )

        if pred.duplicated(
            [
                "patient_key",
                "ecg_key",
            ]
        ).any():
            raise RuntimeError(
                f"{dataset_name}: duplicate prediction IDs."
            )

        safe = (
            dataset_name
            .lower()
            .replace(
                "/",
                "_"
            )
            .replace(
                "-",
                ""
            )
            .replace(
                " ",
                "_"
            )
        )

        (
            boot,
            perf,
            diff,
        ) = bootstrap_dataset(
            pred,
            dataset_name,
            vcg_threshold,
            resnet_threshold,
        )

        boot.to_csv(
            FINAL_OUT
            /
            f"{safe}_VCG_vs_ResNet_bootstrap_{N_BOOT}.csv",
            index=False,
        )

        perf.to_csv(
            FINAL_OUT
            /
            f"{safe}_VCG_vs_ResNet_performance_with95CI.csv",
            index=False,
        )

        diff.to_csv(
            FINAL_OUT
            /
            f"{safe}_VCG_minus_ResNet_paired_differences.csv",
            index=False,
        )

        all_perf.append(
            perf
        )

        all_diff.append(
            diff
        )

        print(
            "\n"
            +
            "=" * 120
        )

        print(
            dataset_name,
            "ZERO-SHOT PERFORMANCE",
        )

        print(
            "=" * 120
        )

        print(
            perf[
                [
                    "dataset",
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
            "\nPAIRED DIFFERENCE — VCG MINUS RESNET"
        )

        print(
            diff.to_string(
                index=False
            )
        )

    combined_perf = pd.concat(
        all_perf,
        ignore_index=True,
    )

    combined_diff = pd.concat(
        all_diff,
        ignore_index=True,
    )

    combined_perf.to_csv(
        FINAL_OUT
        /
        "VCGGeometryResNet_ExternalConfirmation_Combined_Performance.csv",
        index=False,
    )

    combined_diff.to_csv(
        FINAL_OUT
        /
        "VCGGeometryResNet_ExternalConfirmation_Combined_PairedDifferences.csv",
        index=False,
    )

    decision_rows = []

    for dataset_name in [
        "PTB-XL",
        "Chapman/Ningbo",
    ]:
        for metric in [
            "AUROC",
            "AUPRC",
        ]:
            row = combined_diff.loc[
                (
                    combined_diff[
                        "dataset"
                    ]
                    ==
                    dataset_name
                )
                &
                (
                    combined_diff[
                        "metric"
                    ]
                    ==
                    metric
                )
            ]

            if len(row) != 1:
                raise RuntimeError(
                    f"Expected exactly one {dataset_name}/{metric} result."
                )

            r = row.iloc[0]

            decision_rows.append(
                {
                    "dataset":
                        dataset_name,

                    "metric":
                        metric,

                    "delta_VCG_minus_ResNet":
                        float(
                            r[
                                "delta_VCG_minus_ResNet"
                            ]
                        ),

                    "CI_low":
                        float(
                            r[
                                "CI_low"
                            ]
                        ),

                    "CI_high":
                        float(
                            r[
                                "CI_high"
                            ]
                        ),

                    "P":
                        float(
                            r[
                                "P"
                            ]
                        ),

                    "significantly_better":
                        bool(
                            r[
                                "CI_low"
                            ]
                            >
                            0
                        ),

                    "significantly_worse":
                        bool(
                            r[
                                "CI_high"
                            ]
                            <
                            0
                        ),
                }
            )

    decision = pd.DataFrame(
        decision_rows
    )

    decision.to_csv(
        FINAL_OUT
        /
        "VCGGeometryResNet_ExternalConfirmation_DecisionTable.csv",
        index=False,
    )

    summary = {
        "architecture":
            "VCG-Geometry ResNet",

        "seeds":
            SEEDS,

        "external_training":
            False,

        "external_recalibration":
            False,

        "VCG_frozen_WCMC_threshold":
            vcg_threshold,

        "ResNet_frozen_WCMC_threshold":
            resnet_threshold,

        "N_bootstrap":
            N_BOOT,

        "decision_table":
            decision.to_dict(
                orient="records"
            ),
    }

    with open(
        FINAL_OUT
        /
        "VCGGeometryResNet_ExternalConfirmation_FinalSummary.json",
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
        "EXTERNAL CONFIRMATION DECISION TABLE"
    )

    print(
        "=" * 120
    )

    print(
        decision.to_string(
            index=False
        )
    )

    print(
        "\nSaved to:",
        FINAL_OUT,
    )


# ==========================================================================================
# MAIN
# ==========================================================================================

if __name__ == "__main__":
    if MODE == "PTBXL":
        run_external(
            "PTBXL"
        )

    elif MODE == "CHAPMAN":
        run_external(
            "CHAPMAN"
        )

    elif MODE == "FINAL":
        run_final()
