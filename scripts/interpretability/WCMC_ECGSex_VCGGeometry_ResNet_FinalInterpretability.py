#!/usr/bin/env python
# ==========================================================================================
# ECG-SEX — FINAL VCG-GEOMETRY RESNET INTERPRETABILITY
# ==========================================================================================
#
# FINAL PRIMARY MODEL:
#   Five-seed VCG-Geometry ResNet ensemble, seeds 11/22/33/42/55
#
# THIS PIPELINE DOES:
#   1. Exact locked-WCMC inference audit against the completed final summary.
#   2. Full locked-test single-lead ablation for all 12 leads.
#   3. VCG mechanistic/component ablation:
#        - VCG branch off
#        - learned inverse-Dower correction removed
#        - direct XYZ channels removed
#        - velocity/dynamic channels removed
#        - magnitude/direction channels removed
#        - trajectory-turning channel removed
#   4. 2,000 paired bootstrap CIs for performance changes vs intact VCG model.
#   5. Learned inverse-Dower matrix / ΔD summary across five seeds.
#   6. VCG gate distribution on locked WCMC test.
#   7. Five-seed ensemble Integrated Gradients on a balanced N=256 WCMC subset:
#        - ResNet 24-channel input attribution
#        - VCG 8-independent-lead input attribution
#        - signed male-logit evidence
#        - true-class evidence
#        - completeness audit
#   8. Saves machine-readable CSV/NPZ outputs + preliminary publication summary figure.
#
# IMPORTANT:
#   - NO training.
#   - NO model selection.
#   - NO threshold tuning.
#   - Uses frozen WCMC validation threshold from the final locked-test analysis.
#   - Lead ablation is performed at the raw 12-lead waveform level BEFORE preprocessing.
#   - Integrated Gradients baseline = zero in each model-input space.
#
# RESTARTABLE:
#   Existing successful condition prediction files are reused unless VCGINTERP_FORCE=1.
#
# MODE:
#   VCGINTERP_MODE=ALL       (default; recommended)
#   VCGINTERP_MODE=ABLATION
#   VCGINTERP_MODE=IG
#   VCGINTERP_MODE=FINAL
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
    "VCGINTERP_MODE",
    "ALL",
).strip().upper()

if MODE not in {
    "ALL",
    "ABLATION",
    "IG",
    "FINAL",
}:
    raise RuntimeError(
        f"Unknown VCGINTERP_MODE={MODE}"
    )

FORCE = (
    os.environ.get(
        "VCGINTERP_FORCE",
        "0",
    )
    ==
    "1"
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
        "VCGINTERP_BATCH_SIZE",
        "96",
    )
)

NUM_WORKERS = int(
    os.environ.get(
        "VCGINTERP_NUM_WORKERS",
        "8",
    )
)

N_BOOT = int(
    os.environ.get(
        "VCGINTERP_N_BOOT",
        "2000",
    )
)

IG_N = int(
    os.environ.get(
        "VCGINTERP_IG_N",
        "256",
    )
)

IG_STEPS = int(
    os.environ.get(
        "VCGINTERP_IG_STEPS",
        "24",
    )
)

IG_BATCH_SIZE = int(
    os.environ.get(
        "VCGINTERP_IG_BATCH_SIZE",
        "8",
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

TEST_CSV = (
    SPLIT_DIR
    /
    "WCMC_ECGSex_test.csv"
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

OUT_ROOT = (
    WCMC_ROOT
    /
    "21_vcg_geometry_resnet_interpretability"
)

PRED_DIR = (
    OUT_ROOT
    /
    "01_condition_predictions"
)

MECH_DIR = (
    OUT_ROOT
    /
    "02_vcg_mechanistic"
)

IG_DIR = (
    OUT_ROOT
    /
    "03_integrated_gradients"
)

FINAL_DIR = (
    OUT_ROOT
    /
    "04_final_summary"
)

FIG_DIR = (
    OUT_ROOT
    /
    "05_figures"
)


# ==========================================================================================
# CONDITIONS
# ==========================================================================================

LEAD_CONDITIONS = [
    {
        "name":
            f"lead_ablation_{lead}",

        "type":
            "lead",

        "lead":
            lead,
    }
    for lead in LEADS
]

MECHANISTIC_CONDITIONS = [
    {
        "name":
            "vcg_branch_off",

        "type":
            "mechanistic",

        "mechanism":
            "branch_off",
    },
    {
        "name":
            "fixed_inverse_dower",

        "type":
            "mechanistic",

        "mechanism":
            "fixed_dower",
    },
    {
        "name":
            "vcg_no_direct_xyz",

        "type":
            "mechanistic",

        "mechanism":
            "no_xyz",
    },
    {
        "name":
            "vcg_no_velocity_dynamics",

        "type":
            "mechanistic",

        "mechanism":
            "no_velocity",
    },
    {
        "name":
            "vcg_no_magnitude_direction",

        "type":
            "mechanistic",

        "mechanism":
            "no_magnitude_direction",
    },
    {
        "name":
            "vcg_no_trajectory_turning",

        "type":
            "mechanistic",

        "mechanism":
            "no_turning",
    },
]

ALL_CONDITIONS = (
    [
        {
            "name":
                "baseline",

            "type":
                "baseline",
        }
    ]
    +
    LEAD_CONDITIONS
    +
    MECHANISTIC_CONDITIONS
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


# ==========================================================================================
# PHYSIOLOGICAL PROJECTION
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

    return x.astype(
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
# DATASET
# ==========================================================================================

class WCMCTestDataset(
    Dataset
):

    def __init__(
        self,
        df,
        ablate_lead=None,
    ):
        self.df = df.reset_index(
            drop=True
        ).copy()

        self.ablate_lead = ablate_lead

        if (
            ablate_lead is not None
            and
            ablate_lead not in LEADS
        ):
            raise ValueError(
                ablate_lead
            )

        self.ablate_idx = (
            LEADS.index(
                ablate_lead
            )
            if ablate_lead is not None
            else
            None
        )

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

            x12 = orient_and_resample(
                arr
            )

            if self.ablate_idx is not None:
                x12 = x12.copy()

                x12[
                    self.ablate_idx,
                    :
                ] = 0.0

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
                "x_base":
                    None,

                "x_vcg8":
                    None,

                "y":
                    int(
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
        x
        for x in batch
        if x[
            "x_base"
        ] is not None
    ]

    invalid = [
        x
        for x in batch
        if x[
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
                    x[
                        "x_base"
                    ]
                    for x in valid
                ]
            ),

        "x_vcg8":
            torch.stack(
                [
                    x[
                        "x_vcg8"
                    ]
                    for x in valid
                ]
            ),

        "valid":
            valid,

        "invalid":
            invalid,
    }


def make_loader(
    df,
    ablate_lead=None,
    batch_size=None,
    shuffle=False,
    num_workers=None,
):
    if batch_size is None:
        batch_size = BATCH_SIZE

    if num_workers is None:
        num_workers = NUM_WORKERS

    kwargs = {
        "dataset":
            WCMCTestDataset(
                df,
                ablate_lead=ablate_lead,
            ),

        "batch_size":
            batch_size,

        "shuffle":
            shuffle,

        "num_workers":
            num_workers,

        "pin_memory":
            DEVICE.type == "cuda",

        "drop_last":
            False,

        "collate_fn":
            collate_fn,
    }

    if num_workers > 0:
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
# MODEL
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

    def project(
        self,
        x8,
        fixed=False,
    ):
        matrix = (
            self.base_matrix
            if fixed
            else
            self.effective_matrix()
        )

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

    def forward(
        self,
        x8,
    ):
        return self.project(
            x8,
            fixed=False,
        )


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
            xyz,                 # 0:3
            vel,                 # 3:6
            magnitude,           # 6
            speed,               # 7
            unit,                # 8:11
            radial_velocity,     # 11
            turning,             # 12
        ],
        dim=1,
    )


def apply_geometry_ablation(
    features,
    mechanism,
):
    if mechanism is None:
        return features

    x = features.clone()

    if mechanism == "no_xyz":
        x[
            :,
            0:3,
            :
        ] = 0.0

    elif mechanism == "no_velocity":
        # dXYZ, speed, radial velocity
        x[
            :,
            3:6,
            :
        ] = 0.0

        x[
            :,
            7:8,
            :
        ] = 0.0

        x[
            :,
            11:12,
            :
        ] = 0.0

    elif mechanism == "no_magnitude_direction":
        # vector magnitude + unit direction
        x[
            :,
            6:7,
            :
        ] = 0.0

        x[
            :,
            8:11,
            :
        ] = 0.0

    elif mechanism == "no_turning":
        x[
            :,
            12:13,
            :
        ] = 0.0

    else:
        raise ValueError(
            f"Unknown geometry ablation: {mechanism}"
        )

    return x


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

        self.gate = nn.Linear(
            512,
            1,
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
        mechanism=None,
        return_components=False,
    ):
        base_logit, z_base = self.base_forward(
            x_base
        )

        if mechanism == "branch_off":
            if return_components:
                return {
                    "logit":
                        base_logit,

                    "base_logit":
                        base_logit,

                    "gate":
                        torch.zeros_like(
                            base_logit
                        ),

                    "delta":
                        torch.zeros_like(
                            base_logit
                        ),
                }

            return base_logit

        xyz = self.vcg_projection.project(
            x_vcg8,
            fixed=(
                mechanism
                ==
                "fixed_dower"
            ),
        )

        features = vcg_geometry_features(
            xyz
        )

        if mechanism in {
            "no_xyz",
            "no_velocity",
            "no_magnitude_direction",
            "no_turning",
        }:
            features = apply_geometry_ablation(
                features,
                mechanism,
            )

        z_vcg = self.vcg_encoder(
            features
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


def checkpoint_path(
    seed,
):
    return (
        VCG_SEED_ROOT
        /
        f"seed_{seed}"
        /
        "best_VCGGeometryResNet_WCMC.pt"
    )


def load_models():
    models = {}

    for seed in SEEDS:
        path = checkpoint_path(
            seed
        )

        if not path.exists():
            raise FileNotFoundError(
                path
            )

        checkpoint = torch.load(
            path,
            map_location=DEVICE,
            weights_only=False,
        )

        model = VCGGeometryResNet().to(
            DEVICE
        )

        model.load_state_dict(
            extract_state_dict(
                checkpoint
            ),
            strict=True,
        )

        model.eval()

        models[
            seed
        ] = model

    return models


# ==========================================================================================
# FINAL SUMMARY / THRESHOLD
# ==========================================================================================

def load_final_summary():
    if not WCMC_FINAL_SUMMARY.exists():
        raise FileNotFoundError(
            WCMC_FINAL_SUMMARY
        )

    with open(
        WCMC_FINAL_SUMMARY,
        "r",
    ) as f:
        return json.load(
            f
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

        "Brier":
            float(
                brier_score_loss(
                    y,
                    p,
                )
            ),
    }


def p_two_sided(
    values,
):
    x = np.asarray(
        values,
        dtype=float,
    )

    p = (
        2.0
        *
        min(
            np.mean(
                x <= 0
            ),
            np.mean(
                x >= 0
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
                x
            )
            +
            1
        ),
    )


# ==========================================================================================
# CONDITION INFERENCE
# ==========================================================================================

@torch.no_grad()
def infer_condition(
    models,
    test_df,
    condition,
):
    name = condition[
        "name"
    ]

    path = (
        PRED_DIR
        /
        f"{name}.csv"
    )

    if (
        path.exists()
        and
        not FORCE
    ):
        print(
            f"[reuse] {name}: {path}"
        )

        return

    ablate_lead = (
        condition.get(
            "lead"
        )
        if condition[
            "type"
        ]
        ==
        "lead"
        else
        None
    )

    mechanism = condition.get(
        "mechanism",
        None,
    )

    loader = make_loader(
        test_df,
        ablate_lead=ablate_lead,
    )

    rows = []
    failures = []

    print(
        "\n"
        +
        "=" * 120
    )

    print(
        "CONDITION:",
        name
    )

    print(
        "=" * 120
    )

    for batch_idx, batch in enumerate(
        loader,
        start=1,
    ):
        failures.extend(
            batch[
                "invalid"
            ]
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

        seed_probs = []
        seed_gates = []

        for seed in SEEDS:
            comp = models[
                seed
            ](
                x_base,
                x_vcg8,
                mechanism=mechanism,
                return_components=True,
            )

            seed_probs.append(
                torch.sigmoid(
                    comp[
                        "logit"
                    ]
                )
                .float()
                .cpu()
                .numpy()
            )

            seed_gates.append(
                comp[
                    "gate"
                ]
                .float()
                .cpu()
                .numpy()
            )

        probs = np.mean(
            np.stack(
                seed_probs,
                axis=0,
            ),
            axis=0,
        )

        gates = np.mean(
            np.stack(
                seed_gates,
                axis=0,
            ),
            axis=0,
        )

        for i, item in enumerate(
            batch[
                "valid"
            ]
        ):
            rows.append(
                {
                    "patient_key":
                        item[
                            "patient_key"
                        ],

                    "ecg_key":
                        item[
                            "ecg_key"
                        ],

                    "sex_true_binary":
                        item[
                            "y"
                        ],

                    "prob_male":
                        float(
                            probs[
                                i
                            ]
                        ),

                    "mean_vcg_gate":
                        float(
                            gates[
                                i
                            ]
                        ),
                }
            )

        if batch_idx % 25 == 0:
            print(
                f"{name}: batch {batch_idx:,}/{len(loader):,} | "
                f"N={len(rows):,} | failed={len(failures):,}"
            )

    pred = pd.DataFrame(
        rows
    )

    if pred.empty:
        raise RuntimeError(
            f"No predictions for {name}"
        )

    pred.to_csv(
        path,
        index=False,
    )

    pd.DataFrame(
        [
            {
                "patient_key":
                    x[
                        "patient_key"
                    ],

                "ecg_key":
                    x[
                        "ecg_key"
                    ],

                "path":
                    x[
                        "path"
                    ],

                "error":
                    x[
                        "error"
                    ],
            }
            for x in failures
        ]
    ).to_csv(
        PRED_DIR
        /
        f"{name}_failures.csv",
        index=False,
    )

    print(
        "Saved:",
        path,
    )


# ==========================================================================================
# DOWER / GATE
# ==========================================================================================

def save_dower_summary(
    models,
):
    rows = []

    matrices = []

    base = inverse_dower_matrix().numpy()

    for seed in SEEDS:
        effective = (
            models[
                seed
            ]
            .vcg_projection
            .effective_matrix()
            .detach()
            .cpu()
            .numpy()
        )

        matrices.append(
            effective
        )

        for i, axis in enumerate(
            [
                "X",
                "Y",
                "Z",
            ]
        ):
            for j, lead in enumerate(
                DOWER_INPUT_LEADS
            ):
                rows.append(
                    {
                        "seed":
                            seed,

                        "VCG_axis":
                            axis,

                        "ECG_lead":
                            lead,

                        "base_inverse_Dower":
                            float(
                                base[
                                    i,
                                    j
                                ]
                            ),

                        "effective":
                            float(
                                effective[
                                    i,
                                    j
                                ]
                            ),

                        "delta":
                            float(
                                effective[
                                    i,
                                    j
                                ]
                                -
                                base[
                                    i,
                                    j
                                ]
                            ),
                    }
                )

    detail = pd.DataFrame(
        rows
    )

    detail.to_csv(
        MECH_DIR
        /
        "InverseDower_PerSeed_Coefficients.csv",
        index=False,
    )

    mean_matrix = np.mean(
        np.stack(
            matrices,
            axis=0,
        ),
        axis=0,
    )

    mean_delta = (
        mean_matrix
        -
        base
    )

    np.savez_compressed(
        MECH_DIR
        /
        "InverseDower_5Seed_Matrices.npz",
        base_matrix=base,
        effective_matrices=np.stack(
            matrices,
            axis=0,
        ),
        mean_effective_matrix=mean_matrix,
        mean_delta=mean_delta,
        seeds=np.asarray(
            SEEDS,
            dtype=int,
        ),
    )


# ==========================================================================================
# INTEGRATED GRADIENTS
# ==========================================================================================

def select_balanced_ig_subset(
    test_df,
):
    if IG_N % 2 != 0:
        raise RuntimeError(
            "IG_N must be even."
        )

    n_each = (
        IG_N
        //
        2
    )

    female = test_df.loc[
        test_df[
            "sex_binary"
        ]
        ==
        0
    ].sample(
        n=n_each,
        random_state=42,
        replace=False,
    )

    male = test_df.loc[
        test_df[
            "sex_binary"
        ]
        ==
        1
    ].sample(
        n=n_each,
        random_state=42,
        replace=False,
    )

    subset = pd.concat(
        [
            female,
            male,
        ],
        ignore_index=True,
    ).sample(
        frac=1.0,
        random_state=42,
    ).reset_index(
        drop=True
    )

    return subset


def integrated_gradients_batch(
    model,
    x_base,
    x_vcg8,
    steps,
):
    """
    Integrated gradients for the male logit.
    Zero baseline in both model-input spaces.
    Trapezoidal rule over [0,1].
    """
    base0 = torch.zeros_like(
        x_base
    )

    vcg0 = torch.zeros_like(
        x_vcg8
    )

    total_grad_base = torch.zeros_like(
        x_base
    )

    total_grad_vcg = torch.zeros_like(
        x_vcg8
    )

    alphas = torch.linspace(
        0.0,
        1.0,
        steps + 1,
        device=x_base.device,
        dtype=x_base.dtype,
    )

    for step_idx, alpha in enumerate(
        alphas
    ):
        xb = (
            base0
            +
            alpha
            *
            (
                x_base
                -
                base0
            )
        ).detach().requires_grad_(
            True
        )

        xv = (
            vcg0
            +
            alpha
            *
            (
                x_vcg8
                -
                vcg0
            )
        ).detach().requires_grad_(
            True
        )

        logits = model(
            xb,
            xv,
        )

        grads = torch.autograd.grad(
            outputs=logits.sum(),
            inputs=[
                xb,
                xv,
            ],
            create_graph=False,
            retain_graph=False,
        )

        weight = (
            0.5
            if (
                step_idx == 0
                or
                step_idx == steps
            )
            else
            1.0
        )

        total_grad_base += (
            weight
            *
            grads[
                0
            ]
        )

        total_grad_vcg += (
            weight
            *
            grads[
                1
            ]
        )

    avg_grad_base = (
        total_grad_base
        /
        steps
    )

    avg_grad_vcg = (
        total_grad_vcg
        /
        steps
    )

    ig_base = (
        x_base
        -
        base0
    ) * avg_grad_base

    ig_vcg = (
        x_vcg8
        -
        vcg0
    ) * avg_grad_vcg

    with torch.no_grad():
        fx = model(
            x_base,
            x_vcg8,
        )

        f0 = model(
            base0,
            vcg0,
        )

    attribution_sum = (
        ig_base.flatten(
            1
        ).sum(
            dim=1
        )
        +
        ig_vcg.flatten(
            1
        ).sum(
            dim=1
        )
    )

    completeness_error = (
        attribution_sum
        -
        (
            fx
            -
            f0
        )
    )

    return (
        ig_base.detach(),
        ig_vcg.detach(),
        completeness_error.detach(),
    )


def run_integrated_gradients(
    models,
    test_df,
):
    manifest_path = (
        IG_DIR
        /
        "IG_balanced_N256_manifest.csv"
    )

    npz_path = (
        IG_DIR
        /
        "VCGGeometryResNet_5Seed_Ensemble_IG_N256.npz"
    )

    summary_path = (
        IG_DIR
        /
        "IG_LeadImportance_Summary.csv"
    )

    if (
        npz_path.exists()
        and
        summary_path.exists()
        and
        not FORCE
    ):
        print(
            "[reuse] Integrated Gradients outputs already exist."
        )

        return

    subset = select_balanced_ig_subset(
        test_df
    )

    subset[
        [
            "patient_key",
            "ecg_key",
            "sex_binary",
            "raw_local_path",
        ]
    ].to_csv(
        manifest_path,
        index=False,
    )

    loader = make_loader(
        subset,
        ablate_lead=None,
        batch_size=IG_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    all_base_ig = []
    all_vcg_ig = []
    all_y = []
    all_patient = []
    all_ecg = []
    all_completeness = []

    print(
        "\n"
        +
        "=" * 120
    )

    print(
        f"5-SEED ENSEMBLE INTEGRATED GRADIENTS | N={len(subset)} | steps={IG_STEPS}"
    )

    print(
        "=" * 120
    )

    for batch_idx, batch in enumerate(
        loader,
        start=1,
    ):
        if batch[
            "x_base"
        ] is None:
            continue

        x_base = batch[
            "x_base"
        ].to(
            DEVICE
        )

        x_vcg8 = batch[
            "x_vcg8"
        ].to(
            DEVICE
        )

        ensemble_base = torch.zeros_like(
            x_base
        )

        ensemble_vcg = torch.zeros_like(
            x_vcg8
        )

        completeness_seed = []

        for seed in SEEDS:
            model = models[
                seed
            ]

            (
                ig_base,
                ig_vcg,
                completeness,
            ) = integrated_gradients_batch(
                model,
                x_base,
                x_vcg8,
                IG_STEPS,
            )

            ensemble_base += (
                ig_base
                /
                len(
                    SEEDS
                )
            )

            ensemble_vcg += (
                ig_vcg
                /
                len(
                    SEEDS
                )
            )

            completeness_seed.append(
                completeness
                .float()
                .cpu()
                .numpy()
            )

        all_base_ig.append(
            ensemble_base
            .float()
            .cpu()
            .numpy()
        )

        all_vcg_ig.append(
            ensemble_vcg
            .float()
            .cpu()
            .numpy()
        )

        completeness_mean = np.mean(
            np.stack(
                completeness_seed,
                axis=0,
            ),
            axis=0,
        )

        all_completeness.append(
            completeness_mean
        )

        for item in batch[
            "valid"
        ]:
            all_y.append(
                int(
                    item[
                        "y"
                    ]
                )
            )

            all_patient.append(
                item[
                    "patient_key"
                ]
            )

            all_ecg.append(
                item[
                    "ecg_key"
                ]
            )

        print(
            f"IG batch {batch_idx:,}/{len(loader):,}"
        )

    base_ig = np.concatenate(
        all_base_ig,
        axis=0,
    ).astype(
        np.float32
    )

    vcg_ig = np.concatenate(
        all_vcg_ig,
        axis=0,
    ).astype(
        np.float32
    )

    y = np.asarray(
        all_y,
        dtype=np.int8,
    )

    completeness = np.concatenate(
        all_completeness,
        axis=0,
    ).astype(
        np.float32
    )

    np.savez_compressed(
        npz_path,
        base_ig_signed_male_logit=base_ig,
        vcg_ig_signed_male_logit=vcg_ig,
        y_true=y,
        patient_key=np.asarray(
            all_patient
        ),
        ecg_key=np.asarray(
            all_ecg
        ),
        lead_names=np.asarray(
            LEADS
        ),
        vcg_input_leads=np.asarray(
            DOWER_INPUT_LEADS
        ),
        completeness_error=completeness,
        ig_steps=np.asarray(
            [
                IG_STEPS
            ],
            dtype=int,
        ),
        seeds=np.asarray(
            SEEDS,
            dtype=int,
        ),
    )

    # ------------------------------------
    # Base-branch lead attribution
    # ------------------------------------
    # 0:12 = z-scored waveform
    # 12:24 = first derivative
    #
    # Aggregate within each lead but do not combine numerically with VCG-input IG,
    # because the two branches use different preprocessing/scales.

    base_abs_by_lead = []

    base_signed_by_lead = []

    for lead_idx, lead in enumerate(
        LEADS
    ):
        raw = base_ig[
            :,
            lead_idx,
            :
        ]

        deriv = base_ig[
            :,
            lead_idx
            +
            12,
            :
        ]

        abs_score = (
            np.mean(
                np.abs(
                    raw
                ),
                axis=1,
            )
            +
            np.mean(
                np.abs(
                    deriv
                ),
                axis=1,
            )
        )

        signed_score = (
            np.sum(
                raw,
                axis=1,
            )
            +
            np.sum(
                deriv,
                axis=1,
            )
        )

        base_abs_by_lead.append(
            abs_score
        )

        base_signed_by_lead.append(
            signed_score
        )

    base_abs_by_lead = np.stack(
        base_abs_by_lead,
        axis=1,
    )

    base_signed_by_lead = np.stack(
        base_signed_by_lead,
        axis=1,
    )

    vcg_abs_by_lead = np.mean(
        np.abs(
            vcg_ig
        ),
        axis=2,
    )

    vcg_signed_by_lead = np.sum(
        vcg_ig,
        axis=2,
    )

    # True-class evidence:
    # male y=1 -> male logit direction +1
    # female y=0 -> female evidence = -male-logit attribution
    direction = np.where(
        y == 1,
        1.0,
        -1.0,
    ).astype(
        np.float32
    )

    base_trueclass = (
        base_signed_by_lead
        *
        direction[
            :,
            None
        ]
    )

    vcg_trueclass = (
        vcg_signed_by_lead
        *
        direction[
            :,
            None
        ]
    )

    rows = []

    base_mean = np.mean(
        base_abs_by_lead,
        axis=0,
    )

    base_norm = (
        base_mean
        /
        np.sum(
            base_mean
        )
    )

    for i, lead in enumerate(
        LEADS
    ):
        rows.append(
            {
                "branch":
                    "ResNet_24ch",

                "lead":
                    lead,

                "mean_abs_IG":
                    float(
                        base_mean[
                            i
                        ]
                    ),

                "normalized_importance":
                    float(
                        base_norm[
                            i
                        ]
                    ),

                "mean_signed_male_logit":
                    float(
                        np.mean(
                            base_signed_by_lead[
                                :,
                                i
                            ]
                        )
                    ),

                "mean_true_class_evidence":
                    float(
                        np.mean(
                            base_trueclass[
                                :,
                                i
                            ]
                        )
                    ),

                "female_trueclass_evidence":
                    float(
                        np.mean(
                            base_trueclass[
                                y == 0,
                                i
                            ]
                        )
                    ),

                "male_trueclass_evidence":
                    float(
                        np.mean(
                            base_trueclass[
                                y == 1,
                                i
                            ]
                        )
                    ),
            }
        )

    vcg_mean = np.mean(
        vcg_abs_by_lead,
        axis=0,
    )

    vcg_norm = (
        vcg_mean
        /
        np.sum(
            vcg_mean
        )
    )

    for i, lead in enumerate(
        DOWER_INPUT_LEADS
    ):
        rows.append(
            {
                "branch":
                    "VCG_8lead",

                "lead":
                    lead,

                "mean_abs_IG":
                    float(
                        vcg_mean[
                            i
                        ]
                    ),

                "normalized_importance":
                    float(
                        vcg_norm[
                            i
                        ]
                    ),

                "mean_signed_male_logit":
                    float(
                        np.mean(
                            vcg_signed_by_lead[
                                :,
                                i
                            ]
                        )
                    ),

                "mean_true_class_evidence":
                    float(
                        np.mean(
                            vcg_trueclass[
                                :,
                                i
                            ]
                        )
                    ),

                "female_trueclass_evidence":
                    float(
                        np.mean(
                            vcg_trueclass[
                                y == 0,
                                i
                            ]
                        )
                    ),

                "male_trueclass_evidence":
                    float(
                        np.mean(
                            vcg_trueclass[
                                y == 1,
                                i
                            ]
                        )
                    ),
            }
        )

    pd.DataFrame(
        rows
    ).to_csv(
        summary_path,
        index=False,
    )

    pd.DataFrame(
        {
            "patient_key":
                all_patient,

            "ecg_key":
                all_ecg,

            "sex_true_binary":
                y,

            "ensemble_mean_completeness_error":
                completeness,

            "abs_completeness_error":
                np.abs(
                    completeness
                ),
        }
    ).to_csv(
        IG_DIR
        /
        "IG_Completeness_Audit.csv",
        index=False,
    )

    print(
        "IG median |completeness error|:",
        float(
            np.median(
                np.abs(
                    completeness
                )
            )
        ),
    )

    print(
        "Saved IG NPZ:",
        npz_path,
    )


# ==========================================================================================
# PAIRED BOOTSTRAP
# ==========================================================================================

def bootstrap_one_condition(
    condition_name,
    baseline_df,
    condition_df,
    threshold,
):
    merged = baseline_df[
        [
            "patient_key",
            "ecg_key",
            "sex_true_binary",
            "prob_male",
        ]
    ].rename(
        columns={
            "prob_male":
                "prob_baseline",
        }
    ).merge(
        condition_df[
            [
                "patient_key",
                "ecg_key",
                "sex_true_binary",
                "prob_male",
            ]
        ].rename(
            columns={
                "sex_true_binary":
                    "sex_condition",

                "prob_male":
                    "prob_condition",
            }
        ),
        on=[
            "patient_key",
            "ecg_key",
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
            "sex_condition"
        ].to_numpy(
            dtype=int
        ),
    ):
        raise RuntimeError(
            f"Label mismatch: {condition_name}"
        )

    if merged[
        "patient_key"
    ].duplicated().any():
        raise RuntimeError(
            "Locked WCMC test is expected to contain one ECG per patient."
        )

    y = merged[
        "sex_true_binary"
    ].to_numpy(
        dtype=int
    )

    pb = merged[
        "prob_baseline"
    ].to_numpy(
        dtype=float
    )

    pc = merged[
        "prob_condition"
    ].to_numpy(
        dtype=float
    )

    base_point = calc_metrics(
        y,
        pb,
        threshold,
    )

    cond_point = calc_metrics(
        y,
        pc,
        threshold,
    )

    rng = np.random.default_rng(
        20260812
        +
        sum(
            ord(
                c
            )
            for c in condition_name
        )
    )

    records = []

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

        mb = calc_metrics(
            yy,
            pb[
                idx
            ],
            threshold,
        )

        mc = calc_metrics(
            yy,
            pc[
                idx
            ],
            threshold,
        )

        records.append(
            {
                "AUROC_drop":
                    mb[
                        "AUROC"
                    ]
                    -
                    mc[
                        "AUROC"
                    ],

                "AUPRC_drop":
                    mb[
                        "AUPRC"
                    ]
                    -
                    mc[
                        "AUPRC"
                    ],

                "F1_drop":
                    mb[
                        "F1"
                    ]
                    -
                    mc[
                        "F1"
                    ],

                "Brier_increase":
                    mc[
                        "Brier"
                    ]
                    -
                    mb[
                        "Brier"
                    ],
            }
        )

    boot = pd.DataFrame(
        records
    )

    out = {
        "condition":
            condition_name,

        "N":
            int(
                n
            ),

        "baseline_AUROC":
            base_point[
                "AUROC"
            ],

        "condition_AUROC":
            cond_point[
                "AUROC"
            ],

        "AUROC_drop":
            base_point[
                "AUROC"
            ]
            -
            cond_point[
                "AUROC"
            ],

        "AUROC_drop_CI_low":
            float(
                np.percentile(
                    boot[
                        "AUROC_drop"
                    ],
                    2.5,
                )
            ),

        "AUROC_drop_CI_high":
            float(
                np.percentile(
                    boot[
                        "AUROC_drop"
                    ],
                    97.5,
                )
            ),

        "AUROC_P":
            p_two_sided(
                boot[
                    "AUROC_drop"
                ]
            ),

        "baseline_AUPRC":
            base_point[
                "AUPRC"
            ],

        "condition_AUPRC":
            cond_point[
                "AUPRC"
            ],

        "AUPRC_drop":
            base_point[
                "AUPRC"
            ]
            -
            cond_point[
                "AUPRC"
            ],

        "AUPRC_drop_CI_low":
            float(
                np.percentile(
                    boot[
                        "AUPRC_drop"
                    ],
                    2.5,
                )
            ),

        "AUPRC_drop_CI_high":
            float(
                np.percentile(
                    boot[
                        "AUPRC_drop"
                    ],
                    97.5,
                )
            ),

        "AUPRC_P":
            p_two_sided(
                boot[
                    "AUPRC_drop"
                ]
            ),

        "baseline_F1":
            base_point[
                "F1"
            ],

        "condition_F1":
            cond_point[
                "F1"
            ],

        "F1_drop":
            base_point[
                "F1"
            ]
            -
            cond_point[
                "F1"
            ],

        "F1_drop_CI_low":
            float(
                np.percentile(
                    boot[
                        "F1_drop"
                    ],
                    2.5,
                )
            ),

        "F1_drop_CI_high":
            float(
                np.percentile(
                    boot[
                        "F1_drop"
                    ],
                    97.5,
                )
            ),

        "F1_P":
            p_two_sided(
                boot[
                    "F1_drop"
                ]
            ),

        "baseline_Brier":
            base_point[
                "Brier"
            ],

        "condition_Brier":
            cond_point[
                "Brier"
            ],

        "Brier_increase":
            cond_point[
                "Brier"
            ]
            -
            base_point[
                "Brier"
            ],

        "Brier_increase_CI_low":
            float(
                np.percentile(
                    boot[
                        "Brier_increase"
                    ],
                    2.5,
                )
            ),

        "Brier_increase_CI_high":
            float(
                np.percentile(
                    boot[
                        "Brier_increase"
                    ],
                    97.5,
                )
            ),

        "Brier_P":
            p_two_sided(
                boot[
                    "Brier_increase"
                ]
            ),
    }

    return (
        out,
        boot,
    )


# ==========================================================================================
# FINAL SUMMARY / FIGURE
# ==========================================================================================

def create_final_summary():
    FINAL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_summary = load_final_summary()

    threshold = float(
        final_summary[
            "VCG_validation_threshold"
        ]
    )

    baseline_path = (
        PRED_DIR
        /
        "baseline.csv"
    )

    if not baseline_path.exists():
        raise FileNotFoundError(
            baseline_path
        )

    baseline_df = pd.read_csv(
        baseline_path,
        low_memory=False,
    )

    y = baseline_df[
        "sex_true_binary"
    ].to_numpy(
        dtype=int
    )

    p = baseline_df[
        "prob_male"
    ].to_numpy(
        dtype=float
    )

    audit = calc_metrics(
        y,
        p,
        threshold,
    )

    expected = final_summary[
        "VCG_locked_test"
    ]

    audit_rows = []

    for metric in [
        "AUROC",
        "AUPRC",
        "F1",
        "Brier",
    ]:
        diff = (
            audit[
                metric
            ]
            -
            float(
                expected[
                    metric
                ]
            )
        )

        audit_rows.append(
            {
                "metric":
                    metric,

                "recomputed":
                    audit[
                        metric
                    ],

                "expected":
                    float(
                        expected[
                            metric
                        ]
                    ),

                "difference":
                    diff,
            }
        )

        if abs(
            diff
        ) > 5e-4:
            raise RuntimeError(
                f"Baseline audit failed for {metric}: "
                f"recomputed={audit[metric]:.6f}, "
                f"expected={float(expected[metric]):.6f}"
            )

    pd.DataFrame(
        audit_rows
    ).to_csv(
        FINAL_DIR
        /
        "Baseline_Inference_Audit.csv",
        index=False,
    )

    # Gate distribution.
    baseline_df[
        [
            "patient_key",
            "ecg_key",
            "sex_true_binary",
            "mean_vcg_gate",
        ]
    ].to_csv(
        MECH_DIR
        /
        "VCG_Gate_Distribution_LockedWCMC.csv",
        index=False,
    )

    gate_summary = (
        baseline_df.groupby(
            "sex_true_binary"
        )[
            "mean_vcg_gate"
        ]
        .agg(
            [
                "count",
                "mean",
                "std",
                "median",
                "min",
                "max",
            ]
        )
        .reset_index()
    )

    gate_summary.to_csv(
        MECH_DIR
        /
        "VCG_Gate_Summary_BySex.csv",
        index=False,
    )

    # Bootstrap all non-baseline conditions in parallel.
    from joblib import Parallel, delayed

    jobs = []

    for condition in ALL_CONDITIONS:
        if condition[
            "name"
        ] == "baseline":
            continue

        path = (
            PRED_DIR
            /
            f"{condition['name']}.csv"
        )

        if not path.exists():
            raise FileNotFoundError(
                path
            )

        jobs.append(
            (
                condition,
                pd.read_csv(
                    path,
                    low_memory=False,
                ),
            )
        )

    print(
        f"Running paired bootstrap for {len(jobs)} ablation conditions "
        f"with {N_BOOT:,} replicates each..."
    )

    results = Parallel(
        n_jobs=min(
            12,
            len(
                jobs
            ),
        ),
        verbose=10,
    )(
        delayed(
            bootstrap_one_condition
        )(
            condition[
                "name"
            ],
            baseline_df,
            condition_df,
            threshold,
        )
        for condition, condition_df in jobs
    )

    summary_rows = []

    for (
        condition,
        _
    ), (
        summary_row,
        boot
    ) in zip(
        jobs,
        results,
    ):
        summary_row[
            "condition_type"
        ] = condition[
            "type"
        ]

        summary_row[
            "lead"
        ] = condition.get(
            "lead",
            ""
        )

        summary_row[
            "mechanism"
        ] = condition.get(
            "mechanism",
            ""
        )

        summary_rows.append(
            summary_row
        )

        boot.to_csv(
            FINAL_DIR
            /
            f"bootstrap_{condition['name']}.csv",
            index=False,
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        FINAL_DIR
        /
        "VCGGeometryResNet_Ablation_PairedBootstrap_Summary.csv",
        index=False,
    )

    lead_df = summary_df.loc[
        summary_df[
            "condition_type"
        ]
        ==
        "lead"
    ].copy()

    lead_df[
        "lead_order"
    ] = lead_df[
        "lead"
    ].map(
        {
            lead:
                i
            for i, lead in enumerate(
                LEADS
            )
        }
    )

    lead_df = lead_df.sort_values(
        "AUROC_drop",
        ascending=False,
    )

    lead_df.to_csv(
        FINAL_DIR
        /
        "LeadAblation_Ranked_By_AUROC_Drop.csv",
        index=False,
    )

    mech_df = summary_df.loc[
        summary_df[
            "condition_type"
        ]
        ==
        "mechanistic"
    ].copy()

    mech_df = mech_df.sort_values(
        "AUROC_drop",
        ascending=False,
    )

    mech_df.to_csv(
        FINAL_DIR
        /
        "VCG_Mechanistic_Ablation_Ranked.csv",
        index=False,
    )

    # ------------------------------------
    # Preliminary publication summary
    # ------------------------------------
    import matplotlib
    matplotlib.use(
        "Agg"
    )

    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(
        "plasma"
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(
            13,
            9,
        ),
        constrained_layout=True,
    )

    # A: lead ablation
    ax = axes[
        0,
        0
    ]

    plot_lead = lead_df.sort_values(
        "AUROC_drop",
        ascending=True,
    )

    colors = cmap(
        np.linspace(
            0.18,
            0.88,
            len(
                plot_lead
            ),
        )
    )

    xerr = np.vstack(
        [
            plot_lead[
                "AUROC_drop"
            ].to_numpy()
            -
            plot_lead[
                "AUROC_drop_CI_low"
            ].to_numpy(),

            plot_lead[
                "AUROC_drop_CI_high"
            ].to_numpy()
            -
            plot_lead[
                "AUROC_drop"
            ].to_numpy(),
        ]
    )

    ax.barh(
        plot_lead[
            "lead"
        ],
        plot_lead[
            "AUROC_drop"
        ],
        xerr=xerr,
        color=colors,
        alpha=0.9,
        capsize=2,
    )

    ax.axvline(
        0,
        linewidth=1,
        color="black",
    )

    ax.set_xlabel(
        "AUROC decrease after lead ablation"
    )

    ax.set_title(
        "A  Lead-level model reliance",
        loc="left",
        fontweight="bold",
    )

    # B: VCG mechanism
    ax = axes[
        0,
        1
    ]

    labels = {
        "vcg_branch_off":
            "VCG branch off",

        "fixed_inverse_dower":
            "Fixed Dower transform",

        "vcg_no_direct_xyz":
            "Remove direct XYZ",

        "vcg_no_velocity_dynamics":
            "Remove velocity dynamics",

        "vcg_no_magnitude_direction":
            "Remove magnitude/direction",

        "vcg_no_trajectory_turning":
            "Remove turning",
    }

    plot_mech = mech_df.copy()

    plot_mech[
        "display"
    ] = plot_mech[
        "condition"
    ].map(
        labels
    )

    plot_mech = plot_mech.sort_values(
        "AUROC_drop",
        ascending=True,
    )

    colors = cmap(
        np.linspace(
            0.25,
            0.85,
            len(
                plot_mech
            ),
        )
    )

    xerr = np.vstack(
        [
            plot_mech[
                "AUROC_drop"
            ].to_numpy()
            -
            plot_mech[
                "AUROC_drop_CI_low"
            ].to_numpy(),

            plot_mech[
                "AUROC_drop_CI_high"
            ].to_numpy()
            -
            plot_mech[
                "AUROC_drop"
            ].to_numpy(),
        ]
    )

    ax.barh(
        plot_mech[
            "display"
        ],
        plot_mech[
            "AUROC_drop"
        ],
        xerr=xerr,
        color=colors,
        alpha=0.9,
        capsize=2,
    )

    ax.axvline(
        0,
        linewidth=1,
        color="black",
    )

    ax.set_xlabel(
        "AUROC decrease after component ablation"
    )

    ax.set_title(
        "B  VCG component contribution",
        loc="left",
        fontweight="bold",
    )

    # C: Dower correction
    ax = axes[
        1,
        0
    ]

    dower = np.load(
        MECH_DIR
        /
        "InverseDower_5Seed_Matrices.npz"
    )

    delta = dower[
        "mean_delta"
    ]

    im = ax.imshow(
        delta,
        aspect="auto",
        cmap="plasma",
    )

    ax.set_xticks(
        np.arange(
            len(
                DOWER_INPUT_LEADS
            )
        )
    )

    ax.set_xticklabels(
        DOWER_INPUT_LEADS
    )

    ax.set_yticks(
        [
            0,
            1,
            2,
        ]
    )

    ax.set_yticklabels(
        [
            "X",
            "Y",
            "Z",
        ]
    )

    ax.set_xlabel(
        "ECG lead"
    )

    ax.set_ylabel(
        "VCG axis"
    )

    ax.set_title(
        "C  Learned correction to inverse-Dower transform",
        loc="left",
        fontweight="bold",
    )

    fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04,
        label="Mean Δ coefficient",
    )

    # D: IG lead importance
    ax = axes[
        1,
        1
    ]

    ig_path = (
        IG_DIR
        /
        "IG_LeadImportance_Summary.csv"
    )

    if ig_path.exists():
        ig = pd.read_csv(
            ig_path
        )

        base_ig = ig.loc[
            ig[
                "branch"
            ]
            ==
            "ResNet_24ch"
        ].copy()

        base_ig[
            "order"
        ] = base_ig[
            "lead"
        ].map(
            {
                lead:
                    i
                for i, lead in enumerate(
                    LEADS
                )
            }
        )

        base_ig = base_ig.sort_values(
            "order"
        )

        ax.bar(
            np.arange(
                len(
                    base_ig
                )
            ),
            100
            *
            base_ig[
                "normalized_importance"
            ],
            color=cmap(
                np.linspace(
                    0.18,
                    0.88,
                    len(
                        base_ig
                    ),
                )
            ),
        )

        ax.set_xticks(
            np.arange(
                len(
                    base_ig
                )
            )
        )

        ax.set_xticklabels(
            base_ig[
                "lead"
            ],
            rotation=45,
            ha="right",
        )

        ax.set_ylabel(
            "Normalized absolute IG (%)"
        )

        ax.set_title(
            "D  Integrated-gradient lead attribution",
            loc="left",
            fontweight="bold",
        )

    else:
        ax.text(
            0.5,
            0.5,
            "IG output not available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

        ax.set_axis_off()

    fig.suptitle(
        "VCG-Geometry ResNet interpretability — locked WCMC test",
        fontsize=15,
        fontweight="bold",
    )

    fig.savefig(
        FIG_DIR
        /
        "VCGGeometryResNet_Interpretability_Summary.png",
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        FIG_DIR
        /
        "VCGGeometryResNet_Interpretability_Summary.pdf",
        bbox_inches="tight",
    )

    plt.show()

    print(
        "\n"
        +
        "=" * 120
    )

    print(
        "BASELINE AUDIT"
    )

    print(
        "=" * 120
    )

    print(
        pd.DataFrame(
            audit_rows
        ).to_string(
            index=False
        )
    )

    print(
        "\nTOP LEADS BY AUROC DROP"
    )

    print(
        lead_df[
            [
                "lead",
                "AUROC_drop",
                "AUROC_drop_CI_low",
                "AUROC_drop_CI_high",
                "AUPRC_drop",
                "F1_drop",
                "Brier_increase",
            ]
        ].head(
            12
        ).to_string(
            index=False
        )
    )

    print(
        "\nVCG MECHANISTIC ABLATION"
    )

    print(
        mech_df[
            [
                "condition",
                "AUROC_drop",
                "AUROC_drop_CI_low",
                "AUROC_drop_CI_high",
                "AUPRC_drop",
                "F1_drop",
                "Brier_increase",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nSaved final outputs to:",
        FINAL_DIR,
    )


# ==========================================================================================
# MAIN
# ==========================================================================================

def main():
    set_seed(
        42
    )

    for path in [
        TEST_CSV,
        WCMC_FINAL_SUMMARY,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    for seed in SEEDS:
        if not checkpoint_path(
            seed
        ).exists():
            raise FileNotFoundError(
                checkpoint_path(
                    seed
                )
            )

    for directory in [
        OUT_ROOT,
        PRED_DIR,
        MECH_DIR,
        IG_DIR,
        FINAL_DIR,
        FIG_DIR,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    test_df = pd.read_csv(
        TEST_CSV,
        low_memory=False,
    )

    print(
        "=" * 120
    )

    print(
        "FINAL VCG-GEOMETRY RESNET INTERPRETABILITY"
    )

    print(
        "=" * 120
    )

    print(
        "Mode:",
        MODE,
    )

    print(
        "Locked WCMC N:",
        f"{len(test_df):,}",
    )

    print(
        "Seeds:",
        SEEDS,
    )

    print(
        "Device:",
        DEVICE,
    )

    if MODE in {
        "ALL",
        "ABLATION",
        "IG",
    }:
        if DEVICE.type != "cuda":
            raise RuntimeError(
                "GPU required for inference / IG."
            )

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            ),
        )

        models = load_models()

    else:
        models = None

    if MODE in {
        "ALL",
        "ABLATION",
    }:
        save_dower_summary(
            models
        )

        for condition in ALL_CONDITIONS:
            infer_condition(
                models,
                test_df,
                condition,
            )

    if MODE in {
        "ALL",
        "IG",
    }:
        run_integrated_gradients(
            models,
            test_df,
        )

    if models is not None:
        del models

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if MODE in {
        "ALL",
        "FINAL",
    }:
        create_final_summary()


if __name__ == "__main__":
    main()
