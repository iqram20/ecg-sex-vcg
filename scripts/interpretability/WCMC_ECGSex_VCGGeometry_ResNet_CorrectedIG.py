#!/usr/bin/env python
# ==========================================================================================
# ECG-SEX — FINAL VCG-GEOMETRY RESNET
# CORRECTED DECOMPOSED INTEGRATED GRADIENTS
# ==========================================================================================
#
# WHY THIS SCRIPT EXISTS
# ----------------------
# The first IG run interpolated the raw VCG input from zero through an RMS-normalized
# ECG→VCG projection. Because normalization removes most of the interpolation scale,
# that path is ill-conditioned near alpha=0 and produced a very large completeness error.
#
# THIS CORRECTED ANALYSIS DOES NOT REPEAT ABLATION.
#
# It decomposes attribution into two stable representation-space paths:
#
#   A. RESNET/WAVEFORM PATH
#      zero 24-channel base representation -> observed 24-channel representation,
#      while the subject's observed VCG geometry representation is HELD FIXED.
#
#   B. VCG-GEOMETRY PATH
#      zero 13-channel VCG geometry representation -> observed VCG geometry representation,
#      while the subject's observed ResNet/base representation is HELD FIXED.
#
# Thus, no integration path passes through the problematic RMS-normalized raw VCG projection.
#
# Numerical integration:
#   Gauss-Legendre quadrature on [0,1] (default 48 nodes).
#
# Attribution target:
#   male logit.
#
# Sex-specific evidence:
#   Male true-class evidence   = signed male-logit attribution
#   Female true-class evidence = - signed male-logit attribution
#
# ENSEMBLE
# --------
# Five frozen final VCG-Geometry ResNet models:
# seeds 11, 22, 33, 42, 55.
# Attributions are calculated separately per seed and probability/logit function, then
# averaged across seeds.
#
# SAMPLE
# ------
# Reuses the exact balanced N=256 manifest from the previous interpretability run when
# available. Otherwise deterministically recreates 128 female + 128 male cases.
#
# STRICT COMPLETENESS AUDIT
# -------------------------
# For each sample:
#
#   waveform attribution sum ≈
#       f(x_base, geometry_observed) - f(0, geometry_observed)
#
#   geometry attribution sum ≈
#       f(x_base_observed, geometry_observed) - f(x_base_observed, 0)
#
# Publication summaries are generated only if completeness passes.
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

warnings.filterwarnings("ignore")


# ==========================================================================================
# CONFIG
# ==========================================================================================

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

GEOMETRY_CHANNELS = [
    "X",
    "Y",
    "Z",
    "dX",
    "dY",
    "dZ",
    "Magnitude",
    "Speed",
    "UnitX",
    "UnitY",
    "UnitZ",
    "RadialVelocity",
    "Turning",
]

GEOMETRY_GROUPS = {
    "Direct XYZ":
        [0, 1, 2],

    "Velocity":
        [3, 4, 5],

    "Magnitude":
        [6],

    "Speed":
        [7],

    "Direction":
        [8, 9, 10],

    "Radial velocity":
        [11],

    "Trajectory turning":
        [12],
}

VCG_DIM = 192
N_TRAJECTORY_SLOTS = 3
MAX_DOWER_DELTA = 0.03

IG_N = int(
    os.environ.get(
        "VCGIG_N",
        "256",
    )
)

IG_BATCH_SIZE = int(
    os.environ.get(
        "VCGIG_BATCH_SIZE",
        "8",
    )
)

QUAD_NODES = int(
    os.environ.get(
        "VCGIG_QUAD_NODES",
        "48",
    )
)

NUM_WORKERS = int(
    os.environ.get(
        "VCGIG_NUM_WORKERS",
        "0",
    )
)

FORCE = (
    os.environ.get(
        "VCGIG_FORCE",
        "0",
    )
    ==
    "1"
)

# Conservative numerical QC gates.
MAX_MEDIAN_ABS_COMPLETENESS_ERROR = float(
    os.environ.get(
        "VCGIG_MAX_MEDIAN_ABS_ERROR",
        "0.05",
    )
)

MAX_P95_ABS_COMPLETENESS_ERROR = float(
    os.environ.get(
        "VCGIG_MAX_P95_ABS_ERROR",
        "0.25",
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

TEST_CSV = (
    WCMC_ROOT
    /
    "04_wcmc_finetune_from_mimic"
    /
    "splits"
    /
    "WCMC_ECGSex_test.csv"
)

VCG_SEED_ROOT = (
    WCMC_ROOT
    /
    "19_vcg_geometry_resnet_5seed_locked_wcmc"
    /
    "seeds"
)

OLD_INTERP_ROOT = (
    WCMC_ROOT
    /
    "21_vcg_geometry_resnet_interpretability"
)

OLD_MANIFEST = (
    OLD_INTERP_ROOT
    /
    "03_integrated_gradients"
    /
    "IG_balanced_N256_manifest.csv"
)

OUT_ROOT = (
    WCMC_ROOT
    /
    "22_vcg_geometry_resnet_corrected_ig"
)

OUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

NPZ_PATH = (
    OUT_ROOT
    /
    "VCGGeometryResNet_Corrected_DecomposedIG_N256.npz"
)

MANIFEST_PATH = (
    OUT_ROOT
    /
    "CorrectedIG_N256_manifest.csv"
)

COMPLETENESS_PATH = (
    OUT_ROOT
    /
    "CorrectedIG_Completeness_Audit.csv"
)

WAVEFORM_LEAD_SUMMARY = (
    OUT_ROOT
    /
    "CorrectedIG_Waveform_LeadImportance.csv"
)

GEOMETRY_CHANNEL_SUMMARY = (
    OUT_ROOT
    /
    "CorrectedIG_VCGGeometry_ChannelImportance.csv"
)

GEOMETRY_GROUP_SUMMARY = (
    OUT_ROOT
    /
    "CorrectedIG_VCGGeometry_GroupImportance.csv"
)

SEX_EVIDENCE_SUMMARY = (
    OUT_ROOT
    /
    "CorrectedIG_SexSpecific_Evidence.csv"
)

QC_JSON = (
    OUT_ROOT
    /
    "CorrectedIG_QC_Summary.json"
)

FIG_PNG = (
    OUT_ROOT
    /
    "VCGGeometryResNet_CorrectedIG_Summary.png"
)

FIG_PDF = (
    OUT_ROOT
    /
    "VCGGeometryResNet_CorrectedIG_Summary.pdf"
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
# INVERSE DOWER
# ==========================================================================================

def inverse_dower_matrix():
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

    arr = np.squeeze(
        arr
    )

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
            f"Cannot infer 12-lead orientation from {arr.shape}"
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

class ECGDataset(
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

        arr = np.load(
            str(
                row[
                    "raw_local_path"
                ]
            ),
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
        }


def collate_fn(
    batch,
):
    return {
        "x_base":
            torch.stack(
                [
                    x[
                        "x_base"
                    ]
                    for x in batch
                ]
            ),

        "x_vcg8":
            torch.stack(
                [
                    x[
                        "x_vcg8"
                    ]
                    for x in batch
                ]
            ),

        "y":
            np.asarray(
                [
                    x[
                        "y"
                    ]
                    for x in batch
                ],
                dtype=np.int8,
            ),

        "patient_key":
            [
                x[
                    "patient_key"
                ]
                for x in batch
            ],

        "ecg_key":
            [
                x[
                    "ecg_key"
                ]
                for x in batch
            ],
    }


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

    out = torch.cat(
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

    if out.shape[1] != 13:
        raise RuntimeError(
            f"Expected 13 geometry channels, got {out.shape}"
        )

    return out


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
        x_base,
    ):
        h = self.stem(
            x_base
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

    def geometry_from_vcg_input(
        self,
        x_vcg8,
    ):
        xyz = self.vcg_projection(
            x_vcg8
        )

        return vcg_geometry_features(
            xyz
        )

    def forward_from_geometry(
        self,
        x_base,
        geometry,
    ):
        base_logit, z_base = self.base_forward(
            x_base
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

    def forward(
        self,
        x_base,
        x_vcg8,
    ):
        geometry = self.geometry_from_vcg_input(
            x_vcg8
        )

        return self.forward_from_geometry(
            x_base,
            geometry,
        )


# ==========================================================================================
# CHECKPOINTS
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

        for p in model.parameters():
            p.requires_grad_(
                False
            )

        models[
            seed
        ] = model

    return models


# ==========================================================================================
# SAMPLE MANIFEST
# ==========================================================================================

def load_or_create_manifest():
    test_df = pd.read_csv(
        TEST_CSV,
        low_memory=False,
    )

    if OLD_MANIFEST.exists():
        old = pd.read_csv(
            OLD_MANIFEST,
            low_memory=False,
        )

        old[
            "patient_key"
        ] = old[
            "patient_key"
        ].astype(
            str
        )

        old[
            "ecg_key"
        ] = old[
            "ecg_key"
        ].astype(
            str
        )

        test_df[
            "patient_key"
        ] = test_df[
            "patient_key"
        ].astype(
            str
        )

        test_df[
            "ecg_key"
        ] = test_df[
            "ecg_key"
        ].astype(
            str
        )

        manifest = old[
            [
                "patient_key",
                "ecg_key",
            ]
        ].merge(
            test_df,
            on=[
                "patient_key",
                "ecg_key",
            ],
            how="left",
            validate="one_to_one",
        )

        if len(
            manifest
        ) != len(
            old
        ):
            raise RuntimeError(
                "Could not recover all prior IG manifest rows."
            )

        print(
            "Reusing exact prior balanced IG manifest:",
            OLD_MANIFEST,
        )

    else:
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

        manifest = pd.concat(
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

        print(
            "Prior manifest unavailable; recreated deterministic balanced manifest."
        )

    if len(
        manifest
    ) != IG_N:
        raise RuntimeError(
            f"Expected IG_N={IG_N}, got {len(manifest)}"
        )

    counts = manifest[
        "sex_binary"
    ].value_counts().to_dict()

    expected_each = (
        IG_N
        //
        2
    )

    if (
        int(
            counts.get(
                0,
                0,
            )
        )
        !=
        expected_each
        or
        int(
            counts.get(
                1,
                0,
            )
        )
        !=
        expected_each
    ):
        raise RuntimeError(
            f"Manifest not balanced: {counts}"
        )

    manifest.to_csv(
        MANIFEST_PATH,
        index=False,
    )

    return manifest.reset_index(
        drop=True
    )


# ==========================================================================================
# GAUSS-LEGENDRE INTEGRATED GRADIENTS
# ==========================================================================================

def gauss_legendre_nodes_weights(
    n,
    device,
    dtype,
):
    nodes, weights = np.polynomial.legendre.leggauss(
        n
    )

    # Map [-1,1] -> [0,1].
    alphas = (
        nodes
        +
        1.0
    ) / 2.0

    weights = weights / 2.0

    return (
        torch.tensor(
            alphas,
            device=device,
            dtype=dtype,
        ),
        torch.tensor(
            weights,
            device=device,
            dtype=dtype,
        ),
    )


def ig_waveform_with_geometry_fixed(
    model,
    x_base,
    geometry_fixed,
    n_nodes,
):
    baseline = torch.zeros_like(
        x_base
    )

    alphas, weights = gauss_legendre_nodes_weights(
        n_nodes,
        x_base.device,
        x_base.dtype,
    )

    integrated_grad = torch.zeros_like(
        x_base
    )

    for alpha, weight in zip(
        alphas,
        weights,
    ):
        x = (
            baseline
            +
            alpha
            *
            (
                x_base
                -
                baseline
            )
        ).detach().requires_grad_(
            True
        )

        output = model.forward_from_geometry(
            x,
            geometry_fixed,
        )

        grad = torch.autograd.grad(
            outputs=output.sum(),
            inputs=x,
            create_graph=False,
            retain_graph=False,
        )[0]

        integrated_grad += (
            weight
            *
            grad
        )

    attribution = (
        x_base
        -
        baseline
    ) * integrated_grad

    with torch.no_grad():
        fx = model.forward_from_geometry(
            x_base,
            geometry_fixed,
        )

        f0 = model.forward_from_geometry(
            baseline,
            geometry_fixed,
        )

    output_difference = (
        fx
        -
        f0
    )

    attribution_sum = attribution.flatten(
        1
    ).sum(
        dim=1
    )

    error = (
        attribution_sum
        -
        output_difference
    )

    return (
        attribution.detach(),
        output_difference.detach(),
        error.detach(),
    )


def ig_geometry_with_base_fixed(
    model,
    x_base_fixed,
    geometry,
    n_nodes,
):
    baseline = torch.zeros_like(
        geometry
    )

    alphas, weights = gauss_legendre_nodes_weights(
        n_nodes,
        geometry.device,
        geometry.dtype,
    )

    integrated_grad = torch.zeros_like(
        geometry
    )

    for alpha, weight in zip(
        alphas,
        weights,
    ):
        g = (
            baseline
            +
            alpha
            *
            (
                geometry
                -
                baseline
            )
        ).detach().requires_grad_(
            True
        )

        output = model.forward_from_geometry(
            x_base_fixed,
            g,
        )

        grad = torch.autograd.grad(
            outputs=output.sum(),
            inputs=g,
            create_graph=False,
            retain_graph=False,
        )[0]

        integrated_grad += (
            weight
            *
            grad
        )

    attribution = (
        geometry
        -
        baseline
    ) * integrated_grad

    with torch.no_grad():
        fx = model.forward_from_geometry(
            x_base_fixed,
            geometry,
        )

        f0 = model.forward_from_geometry(
            x_base_fixed,
            baseline,
        )

    output_difference = (
        fx
        -
        f0
    )

    attribution_sum = attribution.flatten(
        1
    ).sum(
        dim=1
    )

    error = (
        attribution_sum
        -
        output_difference
    )

    return (
        attribution.detach(),
        output_difference.detach(),
        error.detach(),
    )


# ==========================================================================================
# MAIN IG RUN
# ==========================================================================================

def run_corrected_ig():
    if DEVICE.type != "cuda":
        raise RuntimeError(
            "CUDA required."
        )

    if (
        NPZ_PATH.exists()
        and
        not FORCE
    ):
        print(
            "Corrected IG NPZ already exists. Set VCGIG_FORCE=1 to rerun."
        )

        return

    manifest = load_or_create_manifest()

    loader = DataLoader(
        ECGDataset(
            manifest
        ),
        batch_size=IG_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    models = load_models()

    all_wave_ig = []
    all_geom_ig = []

    all_wave_diff = []
    all_geom_diff = []

    all_wave_error = []
    all_geom_error = []

    all_y = []
    all_patient = []
    all_ecg = []

    print(
        "=" * 120
    )

    print(
        "CORRECTED DECOMPOSED IG"
    )

    print(
        "=" * 120
    )

    print(
        "N:",
        len(
            manifest
        ),
    )

    print(
        "Seeds:",
        SEEDS,
    )

    print(
        "Gauss-Legendre nodes:",
        QUAD_NODES,
    )

    print(
        "Batch size:",
        IG_BATCH_SIZE,
    )

    for batch_idx, batch in enumerate(
        loader,
        start=1,
    ):
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

        batch_wave_ig = torch.zeros_like(
            x_base
        )

        # Geometry representation is seed-specific because the final Dower correction
        # is seed-specific. Every seed receives its own observed geometry representation.
        batch_geom_ig = None

        batch_wave_diff = []
        batch_geom_diff = []

        batch_wave_error = []
        batch_geom_error = []

        for seed in SEEDS:
            model = models[
                seed
            ]

            with torch.no_grad():
                geometry = model.geometry_from_vcg_input(
                    x_vcg8
                ).detach()

            (
                wave_ig,
                wave_diff,
                wave_error,
            ) = ig_waveform_with_geometry_fixed(
                model,
                x_base,
                geometry,
                QUAD_NODES,
            )

            (
                geom_ig,
                geom_diff,
                geom_error,
            ) = ig_geometry_with_base_fixed(
                model,
                x_base,
                geometry,
                QUAD_NODES,
            )

            batch_wave_ig += (
                wave_ig
                /
                len(
                    SEEDS
                )
            )

            if batch_geom_ig is None:
                batch_geom_ig = (
                    geom_ig
                    /
                    len(
                        SEEDS
                    )
                )

            else:
                batch_geom_ig += (
                    geom_ig
                    /
                    len(
                        SEEDS
                    )
                )

            batch_wave_diff.append(
                wave_diff
                .float()
                .cpu()
                .numpy()
            )

            batch_geom_diff.append(
                geom_diff
                .float()
                .cpu()
                .numpy()
            )

            batch_wave_error.append(
                wave_error
                .float()
                .cpu()
                .numpy()
            )

            batch_geom_error.append(
                geom_error
                .float()
                .cpu()
                .numpy()
            )

        all_wave_ig.append(
            batch_wave_ig
            .float()
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        all_geom_ig.append(
            batch_geom_ig
            .float()
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        # Ensemble function differences/errors are averages of seed-specific functions.
        all_wave_diff.append(
            np.mean(
                np.stack(
                    batch_wave_diff,
                    axis=0,
                ),
                axis=0,
            ).astype(
                np.float32
            )
        )

        all_geom_diff.append(
            np.mean(
                np.stack(
                    batch_geom_diff,
                    axis=0,
                ),
                axis=0,
            ).astype(
                np.float32
            )
        )

        all_wave_error.append(
            np.mean(
                np.stack(
                    batch_wave_error,
                    axis=0,
                ),
                axis=0,
            ).astype(
                np.float32
            )
        )

        all_geom_error.append(
            np.mean(
                np.stack(
                    batch_geom_error,
                    axis=0,
                ),
                axis=0,
            ).astype(
                np.float32
            )
        )

        all_y.extend(
            batch[
                "y"
            ].tolist()
        )

        all_patient.extend(
            batch[
                "patient_key"
            ]
        )

        all_ecg.extend(
            batch[
                "ecg_key"
            ]
        )

        print(
            f"Batch {batch_idx:02d}/{len(loader):02d} complete"
        )

        del x_base
        del x_vcg8

        torch.cuda.empty_cache()

    wave_ig = np.concatenate(
        all_wave_ig,
        axis=0,
    )

    geom_ig = np.concatenate(
        all_geom_ig,
        axis=0,
    )

    wave_diff = np.concatenate(
        all_wave_diff,
        axis=0,
    )

    geom_diff = np.concatenate(
        all_geom_diff,
        axis=0,
    )

    wave_error = np.concatenate(
        all_wave_error,
        axis=0,
    )

    geom_error = np.concatenate(
        all_geom_error,
        axis=0,
    )

    y = np.asarray(
        all_y,
        dtype=np.int8,
    )

    np.savez_compressed(
        NPZ_PATH,
        waveform_ig_signed_male_logit=wave_ig,
        geometry_ig_signed_male_logit=geom_ig,
        waveform_output_difference=wave_diff,
        geometry_output_difference=geom_diff,
        waveform_completeness_error=wave_error,
        geometry_completeness_error=geom_error,
        y_true=y,
        patient_key=np.asarray(
            all_patient
        ),
        ecg_key=np.asarray(
            all_ecg
        ),
        waveform_channel_names=np.asarray(
            LEADS
            +
            [
                f"d{lead}"
                for lead in LEADS
            ]
        ),
        geometry_channel_names=np.asarray(
            GEOMETRY_CHANNELS
        ),
        seeds=np.asarray(
            SEEDS,
            dtype=int,
        ),
        gauss_legendre_nodes=np.asarray(
            [
                QUAD_NODES
            ],
            dtype=int,
        ),
    )

    del models

    gc.collect()

    torch.cuda.empty_cache()

    return (
        wave_ig,
        geom_ig,
        wave_diff,
        geom_diff,
        wave_error,
        geom_error,
        y,
        all_patient,
        all_ecg,
    )


# ==========================================================================================
# SUMMARIES / QC
# ==========================================================================================

def load_results():
    data = np.load(
        NPZ_PATH,
        allow_pickle=True,
    )

    return (
        data[
            "waveform_ig_signed_male_logit"
        ],
        data[
            "geometry_ig_signed_male_logit"
        ],
        data[
            "waveform_output_difference"
        ],
        data[
            "geometry_output_difference"
        ],
        data[
            "waveform_completeness_error"
        ],
        data[
            "geometry_completeness_error"
        ],
        data[
            "y_true"
        ].astype(
            np.int8
        ),
        data[
            "patient_key"
        ].tolist(),
        data[
            "ecg_key"
        ].tolist(),
    )


def completeness_qc(
    wave_diff,
    geom_diff,
    wave_error,
    geom_error,
    y,
    patients,
    ecgs,
):
    wave_abs = np.abs(
        wave_error
    )

    geom_abs = np.abs(
        geom_error
    )

    # Scale-aware relative completeness errors, denominator floor prevents
    # unstable ratios for near-zero output differences.
    wave_rel = (
        wave_abs
        /
        np.maximum(
            np.abs(
                wave_diff
            ),
            0.25,
        )
    )

    geom_rel = (
        geom_abs
        /
        np.maximum(
            np.abs(
                geom_diff
            ),
            0.25,
        )
    )

    audit = pd.DataFrame(
        {
            "patient_key":
                patients,

            "ecg_key":
                ecgs,

            "sex_true_binary":
                y,

            "waveform_output_difference":
                wave_diff,

            "waveform_completeness_error":
                wave_error,

            "waveform_abs_error":
                wave_abs,

            "waveform_relative_error":
                wave_rel,

            "geometry_output_difference":
                geom_diff,

            "geometry_completeness_error":
                geom_error,

            "geometry_abs_error":
                geom_abs,

            "geometry_relative_error":
                geom_rel,
        }
    )

    audit.to_csv(
        COMPLETENESS_PATH,
        index=False,
    )

    qc = {
        "N":
            int(
                len(
                    y
                )
            ),

        "Gauss_Legendre_nodes":
            int(
                QUAD_NODES
            ),

        "waveform_median_abs_error":
            float(
                np.median(
                    wave_abs
                )
            ),

        "waveform_p95_abs_error":
            float(
                np.percentile(
                    wave_abs,
                    95,
                )
            ),

        "waveform_median_relative_error":
            float(
                np.median(
                    wave_rel
                )
            ),

        "waveform_p95_relative_error":
            float(
                np.percentile(
                    wave_rel,
                    95,
                )
            ),

        "geometry_median_abs_error":
            float(
                np.median(
                    geom_abs
                )
            ),

        "geometry_p95_abs_error":
            float(
                np.percentile(
                    geom_abs,
                    95,
                )
            ),

        "geometry_median_relative_error":
            float(
                np.median(
                    geom_rel
                )
            ),

        "geometry_p95_relative_error":
            float(
                np.percentile(
                    geom_rel,
                    95,
                )
            ),
    }

    qc[
        "waveform_pass"
    ] = bool(
        qc[
            "waveform_median_abs_error"
        ]
        <=
        MAX_MEDIAN_ABS_COMPLETENESS_ERROR
        and
        qc[
            "waveform_p95_abs_error"
        ]
        <=
        MAX_P95_ABS_COMPLETENESS_ERROR
    )

    qc[
        "geometry_pass"
    ] = bool(
        qc[
            "geometry_median_abs_error"
        ]
        <=
        MAX_MEDIAN_ABS_COMPLETENESS_ERROR
        and
        qc[
            "geometry_p95_abs_error"
        ]
        <=
        MAX_P95_ABS_COMPLETENESS_ERROR
    )

    qc[
        "overall_pass"
    ] = bool(
        qc[
            "waveform_pass"
        ]
        and
        qc[
            "geometry_pass"
        ]
    )

    with open(
        QC_JSON,
        "w",
    ) as f:
        json.dump(
            qc,
            f,
            indent=2,
        )

    print(
        "\n"
        +
        "=" * 120
    )

    print(
        "COMPLETENESS QC"
    )

    print(
        "=" * 120
    )

    print(
        json.dumps(
            qc,
            indent=2,
        )
    )

    if not qc[
        "overall_pass"
    ]:
        raise RuntimeError(
            "Corrected IG failed completeness QC. "
            "Do not use attribution outputs for publication."
        )

    return qc


def create_summaries_and_figure(
    wave_ig,
    geom_ig,
    y,
):
    direction = np.where(
        y == 1,
        1.0,
        -1.0,
    ).astype(
        np.float32
    )

    # ----------------------------------------------------------------------
    # WAVEFORM / RESNET LEAD IMPORTANCE
    # 0:12 waveform; 12:24 first derivative.
    # ----------------------------------------------------------------------

    wave_abs_by_lead = []
    wave_signed_by_lead = []

    for lead_idx in range(
        12
    ):
        raw_attr = wave_ig[
            :,
            lead_idx,
            :
        ]

        deriv_attr = wave_ig[
            :,
            lead_idx
            +
            12,
            :
        ]

        wave_abs_by_lead.append(
            (
                np.mean(
                    np.abs(
                        raw_attr
                    ),
                    axis=1,
                )
                +
                np.mean(
                    np.abs(
                        deriv_attr
                    ),
                    axis=1,
                )
            )
        )

        wave_signed_by_lead.append(
            (
                np.sum(
                    raw_attr,
                    axis=1,
                )
                +
                np.sum(
                    deriv_attr,
                    axis=1,
                )
            )
        )

    wave_abs_by_lead = np.stack(
        wave_abs_by_lead,
        axis=1,
    )

    wave_signed_by_lead = np.stack(
        wave_signed_by_lead,
        axis=1,
    )

    wave_trueclass = (
        wave_signed_by_lead
        *
        direction[
            :,
            None
        ]
    )

    mean_wave_abs = np.mean(
        wave_abs_by_lead,
        axis=0,
    )

    norm_wave = (
        mean_wave_abs
        /
        np.sum(
            mean_wave_abs
        )
    )

    wave_rows = []

    for i, lead in enumerate(
        LEADS
    ):
        wave_rows.append(
            {
                "lead":
                    lead,

                "mean_abs_IG":
                    float(
                        mean_wave_abs[
                            i
                        ]
                    ),

                "normalized_importance":
                    float(
                        norm_wave[
                            i
                        ]
                    ),

                "mean_signed_male_logit":
                    float(
                        np.mean(
                            wave_signed_by_lead[
                                :,
                                i
                            ]
                        )
                    ),

                "mean_trueclass_evidence":
                    float(
                        np.mean(
                            wave_trueclass[
                                :,
                                i
                            ]
                        )
                    ),

                "female_trueclass_evidence":
                    float(
                        np.mean(
                            wave_trueclass[
                                y == 0,
                                i
                            ]
                        )
                    ),

                "male_trueclass_evidence":
                    float(
                        np.mean(
                            wave_trueclass[
                                y == 1,
                                i
                            ]
                        )
                    ),
            }
        )

    wave_df = pd.DataFrame(
        wave_rows
    )

    wave_df[
        "rank"
    ] = wave_df[
        "mean_abs_IG"
    ].rank(
        method="min",
        ascending=False,
    ).astype(
        int
    )

    wave_df = wave_df.sort_values(
        "rank"
    )

    wave_df.to_csv(
        WAVEFORM_LEAD_SUMMARY,
        index=False,
    )

    # ----------------------------------------------------------------------
    # GEOMETRY CHANNEL IMPORTANCE
    # ----------------------------------------------------------------------

    geom_abs = np.mean(
        np.abs(
            geom_ig
        ),
        axis=2,
    )

    geom_signed = np.sum(
        geom_ig,
        axis=2,
    )

    geom_trueclass = (
        geom_signed
        *
        direction[
            :,
            None
        ]
    )

    mean_geom_abs = np.mean(
        geom_abs,
        axis=0,
    )

    norm_geom = (
        mean_geom_abs
        /
        np.sum(
            mean_geom_abs
        )
    )

    geom_rows = []

    for i, channel in enumerate(
        GEOMETRY_CHANNELS
    ):
        geom_rows.append(
            {
                "channel":
                    channel,

                "mean_abs_IG":
                    float(
                        mean_geom_abs[
                            i
                        ]
                    ),

                "normalized_importance":
                    float(
                        norm_geom[
                            i
                        ]
                    ),

                "mean_signed_male_logit":
                    float(
                        np.mean(
                            geom_signed[
                                :,
                                i
                            ]
                        )
                    ),

                "mean_trueclass_evidence":
                    float(
                        np.mean(
                            geom_trueclass[
                                :,
                                i
                            ]
                        )
                    ),

                "female_trueclass_evidence":
                    float(
                        np.mean(
                            geom_trueclass[
                                y == 0,
                                i
                            ]
                        )
                    ),

                "male_trueclass_evidence":
                    float(
                        np.mean(
                            geom_trueclass[
                                y == 1,
                                i
                            ]
                        )
                    ),
            }
        )

    geom_df = pd.DataFrame(
        geom_rows
    )

    geom_df[
        "rank"
    ] = geom_df[
        "mean_abs_IG"
    ].rank(
        method="min",
        ascending=False,
    ).astype(
        int
    )

    geom_df = geom_df.sort_values(
        "rank"
    )

    geom_df.to_csv(
        GEOMETRY_CHANNEL_SUMMARY,
        index=False,
    )

    # ----------------------------------------------------------------------
    # GEOMETRY GROUP IMPORTANCE
    # ----------------------------------------------------------------------

    group_rows = []

    group_abs_values = {}

    for group, indices in GEOMETRY_GROUPS.items():
        # Sum mean absolute attribution over all channels/time within group.
        sample_group_abs = np.sum(
            geom_abs[
                :,
                indices
            ],
            axis=1,
        )

        sample_group_signed = np.sum(
            geom_signed[
                :,
                indices
            ],
            axis=1,
        )

        sample_group_trueclass = (
            sample_group_signed
            *
            direction
        )

        group_abs_values[
            group
        ] = float(
            np.mean(
                sample_group_abs
            )
        )

        group_rows.append(
            {
                "group":
                    group,

                "channels":
                    ", ".join(
                        [
                            GEOMETRY_CHANNELS[
                                i
                            ]
                            for i in indices
                        ]
                    ),

                "mean_abs_IG":
                    group_abs_values[
                        group
                    ],

                "mean_signed_male_logit":
                    float(
                        np.mean(
                            sample_group_signed
                        )
                    ),

                "mean_trueclass_evidence":
                    float(
                        np.mean(
                            sample_group_trueclass
                        )
                    ),

                "female_trueclass_evidence":
                    float(
                        np.mean(
                            sample_group_trueclass[
                                y == 0
                            ]
                        )
                    ),

                "male_trueclass_evidence":
                    float(
                        np.mean(
                            sample_group_trueclass[
                                y == 1
                            ]
                        )
                    ),
            }
        )

    total_group_abs = sum(
        group_abs_values.values()
    )

    for row in group_rows:
        row[
            "normalized_importance"
        ] = (
            row[
                "mean_abs_IG"
            ]
            /
            total_group_abs
        )

    group_df = pd.DataFrame(
        group_rows
    )

    group_df[
        "rank"
    ] = group_df[
        "mean_abs_IG"
    ].rank(
        method="min",
        ascending=False,
    ).astype(
        int
    )

    group_df = group_df.sort_values(
        "rank"
    )

    group_df.to_csv(
        GEOMETRY_GROUP_SUMMARY,
        index=False,
    )

    # ----------------------------------------------------------------------
    # COMPACT SEX EVIDENCE TABLE
    # ----------------------------------------------------------------------

    sex_rows = []

    for _, row in wave_df.iterrows():
        sex_rows.append(
            {
                "representation":
                    "Waveform lead",

                "feature":
                    row[
                        "lead"
                    ],

                "female_trueclass_evidence":
                    row[
                        "female_trueclass_evidence"
                    ],

                "male_trueclass_evidence":
                    row[
                        "male_trueclass_evidence"
                    ],
            }
        )

    for _, row in group_df.iterrows():
        sex_rows.append(
            {
                "representation":
                    "VCG geometry group",

                "feature":
                    row[
                        "group"
                    ],

                "female_trueclass_evidence":
                    row[
                        "female_trueclass_evidence"
                    ],

                "male_trueclass_evidence":
                    row[
                        "male_trueclass_evidence"
                    ],
            }
        )

    pd.DataFrame(
        sex_rows
    ).to_csv(
        SEX_EVIDENCE_SUMMARY,
        index=False,
    )

    # ----------------------------------------------------------------------
    # FIGURE
    # ----------------------------------------------------------------------

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

    # A waveform lead importance
    ax = axes[
        0,
        0
    ]

    plot_wave = wave_df.sort_values(
        "normalized_importance",
        ascending=True,
    )

    ax.barh(
        plot_wave[
            "lead"
        ],
        100
        *
        plot_wave[
            "normalized_importance"
        ],
        color=cmap(
            np.linspace(
                0.18,
                0.88,
                len(
                    plot_wave
                ),
            )
        ),
    )

    ax.set_xlabel(
        "Normalized absolute IG (%)"
    )

    ax.set_title(
        "A  Waveform lead attribution",
        loc="left",
        fontweight="bold",
    )

    # B geometry groups
    ax = axes[
        0,
        1
    ]

    plot_group = group_df.sort_values(
        "normalized_importance",
        ascending=True,
    )

    ax.barh(
        plot_group[
            "group"
        ],
        100
        *
        plot_group[
            "normalized_importance"
        ],
        color=cmap(
            np.linspace(
                0.20,
                0.85,
                len(
                    plot_group
                ),
            )
        ),
    )

    ax.set_xlabel(
        "Normalized absolute IG (%)"
    )

    ax.set_title(
        "B  VCG geometry attribution",
        loc="left",
        fontweight="bold",
    )

    # C sex-specific waveform evidence
    ax = axes[
        1,
        0
    ]

    order = [
        lead
        for lead in LEADS
    ]

    plot_sex_wave = wave_df.set_index(
        "lead"
    ).loc[
        order
    ].reset_index()

    x = np.arange(
        len(
            plot_sex_wave
        )
    )

    width = 0.38

    ax.bar(
        x
        -
        width
        /
        2,
        plot_sex_wave[
            "female_trueclass_evidence"
        ],
        width=width,
        label="Female",
    )

    ax.bar(
        x
        +
        width
        /
        2,
        plot_sex_wave[
            "male_trueclass_evidence"
        ],
        width=width,
        label="Male",
    )

    ax.axhline(
        0,
        linewidth=1,
        color="black",
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        plot_sex_wave[
            "lead"
        ],
        rotation=45,
        ha="right",
    )

    ax.set_ylabel(
        "Mean true-class evidence"
    )

    ax.legend(
        frameon=False
    )

    ax.set_title(
        "C  Sex-specific waveform evidence",
        loc="left",
        fontweight="bold",
    )

    # D completeness
    ax = axes[
        1,
        1
    ]

    comp = pd.read_csv(
        COMPLETENESS_PATH
    )

    values = [
        comp[
            "waveform_abs_error"
        ].to_numpy(),
        comp[
            "geometry_abs_error"
        ].to_numpy(),
    ]

    ax.boxplot(
        values,
        tick_labels=[
            "Waveform",
            "VCG geometry",
        ],
        showfliers=False,
    )

    ax.set_ylabel(
        "|Completeness error|"
    )

    ax.set_title(
        "D  Attribution completeness",
        loc="left",
        fontweight="bold",
    )

    fig.suptitle(
        "Corrected decomposed Integrated Gradients — final VCG-Geometry ResNet",
        fontsize=15,
        fontweight="bold",
    )

    fig.savefig(
        FIG_PNG,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        FIG_PDF,
        bbox_inches="tight",
    )

    plt.show()

    print(
        "\n"
        +
        "=" * 120
    )

    print(
        "WAVEFORM LEAD IMPORTANCE"
    )

    print(
        "=" * 120
    )

    print(
        wave_df[
            [
                "rank",
                "lead",
                "normalized_importance",
                "female_trueclass_evidence",
                "male_trueclass_evidence",
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
        "VCG GEOMETRY GROUP IMPORTANCE"
    )

    print(
        "=" * 120
    )

    print(
        group_df[
            [
                "rank",
                "group",
                "normalized_importance",
                "female_trueclass_evidence",
                "male_trueclass_evidence",
            ]
        ].to_string(
            index=False
        )
    )


# ==========================================================================================
# MAIN
# ==========================================================================================

def main():
    set_seed(
        42
    )

    if DEVICE.type != "cuda":
        raise RuntimeError(
            "CUDA required."
        )

    print(
        "=" * 120
    )

    print(
        "FINAL VCG-GEOMETRY RESNET — CORRECTED DECOMPOSED IG"
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
        "Seeds:",
        SEEDS,
    )

    print(
        "IG N:",
        IG_N,
    )

    print(
        "Gauss-Legendre nodes:",
        QUAD_NODES,
    )

    if (
        not NPZ_PATH.exists()
        or
        FORCE
    ):
        result = run_corrected_ig()

        if result is None:
            (
                wave_ig,
                geom_ig,
                wave_diff,
                geom_diff,
                wave_error,
                geom_error,
                y,
                patients,
                ecgs,
            ) = load_results()

        else:
            (
                wave_ig,
                geom_ig,
                wave_diff,
                geom_diff,
                wave_error,
                geom_error,
                y,
                patients,
                ecgs,
            ) = result

    else:
        print(
            "Reusing existing corrected IG NPZ:",
            NPZ_PATH,
        )

        (
            wave_ig,
            geom_ig,
            wave_diff,
            geom_diff,
            wave_error,
            geom_error,
            y,
            patients,
            ecgs,
        ) = load_results()

    completeness_qc(
        wave_diff,
        geom_diff,
        wave_error,
        geom_error,
        y,
        patients,
        ecgs,
    )

    create_summaries_and_figure(
        wave_ig,
        geom_ig,
        y,
    )

    print(
        "\n"
        +
        "=" * 120
    )

    print(
        "CORRECTED IG COMPLETE"
    )

    print(
        "=" * 120
    )

    print(
        "Output:",
        OUT_ROOT,
    )


if __name__ == "__main__":
    main()
