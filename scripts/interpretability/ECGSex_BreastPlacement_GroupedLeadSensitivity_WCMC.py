#!/usr/bin/env python3
# ==========================================================================================
# ECG-SEX — BREAST / PRECORDIAL-PLACEMENT SENSITIVITY ANALYSIS
# GROUPED LEAD ABLATION | FINAL 5-SEED VCG-GEOMETRY RESNET | LOCKED WCMC/NYP TEST
# ==========================================================================================
#
# PURPOSE
# -------
# Evaluate whether the final ECG-sex signal depends disproportionately on precordial leads
# potentially affected by breast anatomy / chest-electrode positioning.
#
# This script:
#   1) Reuses the EXACT final VCG-Geometry ResNet model definition/checkpoints from:
#        WCMC_ECGSex_VCGGeometry_ResNet_FinalInterpretability.py
#   2) Performs grouped raw-signal ablations BEFORE preprocessing:
#        A. V3-V6 removed          (primary breast-placement sensitivity condition)
#        B. V1-V2 removed          (parasternal comparison)
#        C. V1-V6 removed          (all precordial leads removed)
#        D. all limb leads removed (I, II, III, aVR, aVL, aVF)
#   3) Evaluates the frozen five-seed ensemble on the same locked WCMC/NYP test cohort.
#   4) Uses the existing intact-model baseline predictions from the final interpretability run.
#   5) Performs 2,000 paired patient-level bootstrap comparisons.
#   6) Repeats grouped-ablation AUROC analyses within BMI strata:
#        <25, 25-29.9, >=30 kg/m^2
#   7) Tests whether ablation-induced TRUE-CLASS MARGIN LOSS differs by sex/BMI:
#
#        margin_loss ~ sex + BMI + sex*BMI + age
#
#      using HC3 robust standard errors.
#
# IMPORTANT INTERPRETATION
# ------------------------
# - This is a sensitivity analysis, not a direct measurement of electrode position.
# - BMI is an indirect proxy for body habitus and does not measure breast anatomy.
# - Grouped test-time masking introduces an out-of-distribution perturbation.
# - Therefore this analysis assesses model reliance, not causal physiology.
#
# Female=0, Male=1
# ==========================================================================================

from __future__ import annotations

import os
import gc
import math
import random
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

import torch
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

import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==========================================================================================
# CONFIG
# ==========================================================================================

SEED = 20260925

N_BOOT = int(
    os.environ.get(
        "BREASTABL_N_BOOT",
        "2000",
    )
)

BATCH_SIZE = int(
    os.environ.get(
        "BREASTABL_BATCH_SIZE",
        "96",
    )
)

NUM_WORKERS = int(
    os.environ.get(
        "BREASTABL_NUM_WORKERS",
        "8",
    )
)

FORCE = (
    os.environ.get(
        "BREASTABL_FORCE",
        "0",
    )
    ==
    "1"
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else
    "cpu"
)

if DEVICE.type != "cuda":
    raise RuntimeError(
        "CUDA/GPU is required for grouped-ablation inference."
    )


# ==========================================================================================
# PATHS
# ==========================================================================================

CODE_DIR = Path(
    "/athena/madelab/scratch/iqh4001/ECG/ECG-gender/Code"
)

ORIGINAL_INTERP_CANDIDATES = [
    CODE_DIR
    /
    "WCMC_ECGSex_VCGGeometry_ResNet_FinalInterpretability.py",

    Path(
        "/athena/madelab/scratch/iqh4001/ECG/GitHub/ecg-sex-vcg/"
        "scripts/interpretability/"
        "WCMC_ECGSex_VCGGeometry_ResNet_FinalInterpretability.py"
    ),
]

ORIGINAL_INTERP = next(
    (
        p
        for p in ORIGINAL_INTERP_CANDIDATES
        if p.exists()
    ),
    None,
)

if ORIGINAL_INTERP is None:
    raise FileNotFoundError(
        "Could not locate the final interpretability pipeline.\nTried:\n"
        +
        "\n".join(
            str(p)
            for p in ORIGINAL_INTERP_CANDIDATES
        )
    )

MASTER_PARQUET = Path(
    "/athena/madelab/scratch/iqh4001/Azure_Cradle_ECG/"
    "Data/ECG-Echo_MetaData/ecg_echo_master_cohort_v1.parquet"
)

OUT_DIR = Path(
    "/athena/madelab/scratch/iqh4001/ECG/ECG-gender/Results/"
    "BreastPlacement_GroupedLeadSensitivity_WCMC"
)

PRED_DIR = (
    OUT_DIR
    /
    "predictions"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PRED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OVERALL_CSV = (
    OUT_DIR
    /
    "GroupedLeadAblation_Overall_2000Bootstrap.csv"
)

BMI_CSV = (
    OUT_DIR
    /
    "GroupedLeadAblation_BMIstratified_2000Bootstrap.csv"
)

INTERACTION_CSV = (
    OUT_DIR
    /
    "GroupedLeadAblation_SexBMI_MarginInteraction_HC3.csv"
)

COHORT_AUDIT_CSV = (
    OUT_DIR
    /
    "GroupedLeadAblation_CohortAudit.csv"
)

FIG_PNG = (
    OUT_DIR
    /
    "GroupedLeadAblation_BreastPlacementSensitivity.png"
)

FIG_PDF = (
    OUT_DIR
    /
    "GroupedLeadAblation_BreastPlacementSensitivity.pdf"
)


# ==========================================================================================
# LOAD EXACT FINAL PIPELINE AS A MODULE
# ==========================================================================================

spec = importlib.util.spec_from_file_location(
    "final_vcg_interp",
    ORIGINAL_INTERP,
)

if spec is None or spec.loader is None:
    raise RuntimeError(
        f"Could not import: {ORIGINAL_INTERP}"
    )

final = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    final
)

print(
    "Imported exact final VCG pipeline:"
)

print(
    ORIGINAL_INTERP
)


# ==========================================================================================
# GROUPED ABLATION CONDITIONS
# ==========================================================================================

GROUP_CONDITIONS = [
    {
        "name":
            "remove_V3_V6",

        "display":
            "Remove V3-V6",

        "leads":
            [
                "V3",
                "V4",
                "V5",
                "V6",
            ],

        "purpose":
            "Primary breast-placement sensitivity",
    },
    {
        "name":
            "remove_V1_V2",

        "display":
            "Remove V1-V2",

        "leads":
            [
                "V1",
                "V2",
            ],

        "purpose":
            "Parasternal precordial comparison",
    },
    {
        "name":
            "remove_all_precordial",

        "display":
            "Remove V1-V6",

        "leads":
            [
                "V1",
                "V2",
                "V3",
                "V4",
                "V5",
                "V6",
            ],

        "purpose":
            "All precordial leads removed",
    },
    {
        "name":
            "remove_all_limb",

        "display":
            "Remove all limb leads",

        "leads":
            [
                "I",
                "II",
                "III",
                "aVR",
                "aVL",
                "aVF",
            ],

        "purpose":
            "Precordial-dominant comparison",
    },
]


# ==========================================================================================
# REPRODUCIBILITY
# ==========================================================================================

random.seed(
    SEED
)

np.random.seed(
    SEED
)

torch.manual_seed(
    SEED
)

torch.cuda.manual_seed_all(
    SEED
)

torch.backends.cudnn.benchmark = True


# ==========================================================================================
# DATASET — GROUP MASKING BEFORE PREPROCESSING
# ==========================================================================================

class GroupMaskDataset(
    Dataset
):

    def __init__(
        self,
        df,
        leads_to_mask,
    ):
        self.df = df.reset_index(
            drop=True
        ).copy()

        self.leads_to_mask = list(
            leads_to_mask
        )

        bad = [
            lead
            for lead in self.leads_to_mask
            if lead not in final.LEADS
        ]

        if bad:
            raise ValueError(
                f"Unknown lead(s): {bad}"
            )

        self.mask_indices = [
            final.LEADS.index(
                lead
            )
            for lead in self.leads_to_mask
        ]

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

            x12 = final.orient_and_resample(
                arr
            )

            x12 = x12.copy()

            x12[
                self.mask_indices,
                :
            ] = 0.0

            return {
                "x_base":
                    final.make_base_input(
                        x12
                    ),

                "x_vcg8":
                    final.make_vcg_input(
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


def make_group_loader(
    test_df,
    leads_to_mask,
):
    kwargs = {
        "dataset":
            GroupMaskDataset(
                test_df,
                leads_to_mask,
            ),

        "batch_size":
            BATCH_SIZE,

        "shuffle":
            False,

        "num_workers":
            NUM_WORKERS,

        "pin_memory":
            True,

        "drop_last":
            False,

        "collate_fn":
            final.collate_fn,
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
# GROUPED-ABLATION INFERENCE
# ==========================================================================================

@torch.no_grad()
def infer_group_condition(
    models,
    test_df,
    condition,
):
    path = (
        PRED_DIR
        /
        f"{condition['name']}.csv"
    )

    if (
        path.exists()
        and
        not FORCE
    ):
        print(
            f"[reuse] {path}"
        )

        return pd.read_csv(
            path,
            low_memory=False,
        )

    print(
        "\n"
        +
        "=" * 120
    )

    print(
        "GROUPED CONDITION:",
        condition[
            "display"
        ]
    )

    print(
        "Mask leads:",
        condition[
            "leads"
        ]
    )

    print(
        "=" * 120
    )

    loader = make_group_loader(
        test_df,
        condition[
            "leads"
        ],
    )

    rows = []
    failures = []

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

        for model_seed in final.SEEDS:
            comp = models[
                model_seed
            ](
                x_base,
                x_vcg8,
                mechanism=None,
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
                f"{condition['name']}: "
                f"batch {batch_idx:,}/{len(loader):,} | "
                f"N={len(rows):,} | "
                f"failed={len(failures):,}"
            )

    pred = pd.DataFrame(
        rows
    )

    if pred.empty:
        raise RuntimeError(
            f"No predictions produced for {condition['name']}"
        )

    pred.to_csv(
        path,
        index=False,
    )

    failure_path = (
        PRED_DIR
        /
        f"{condition['name']}_failures.csv"
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
        failure_path,
        index=False,
    )

    print(
        "Saved:",
        path
    )

    return pred


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


def two_sided_bootstrap_p(
    values,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    n = len(
        values
    )

    p = (
        2.0
        *
        min(
            np.mean(
                values <= 0
            ),
            np.mean(
                values >= 0
            ),
        )
    )

    return max(
        min(
            float(
                p
            ),
            1.0,
        ),
        1.0
        /
        (
            n
            +
            1
        ),
    )


def paired_bootstrap_summary(
    baseline_df,
    condition_df,
    threshold,
    n_boot,
    seed,
    label,
):
    merged = baseline_df.merge(
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
        how="inner",
        validate="one_to_one",
    )

    if len(
        merged