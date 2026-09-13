#!/usr/bin/env python
# ==========================================================================================
# WCMC/NYP ECG-SEX — AGE-MATCHED FEMALE vs MALE RAW ECG MORPHOLOGY
# NATURE-STYLE MEDIAN + IQR
# CURRENT LOCKED WCMC TEST COHORT
# ==========================================================================================
#
# GOAL
# ----
# Compare population-level beat-aligned 12-lead ECG morphology between females and males
# after one-to-one age matching in the CURRENT WCMC/NYP locked test cohort.
#
# SEX CODING
# ----------
# Female = 0
# Male   = 1
#
# MATCHING
# --------
# 1. Exact matching within integer year of age, without replacement.
# 2. At each age, randomly sample min(N_female, N_male) from each sex using a fixed seed.
# 3. Extract one representative median beat per patient.
# 4. After waveform/beat QC, re-match valid female and male subjects within integer age
#    so the FINAL plotted cohort remains exactly age-balanced.
#
# MORPHOLOGY
# ----------
# - Raw WCMC .npy ECGs.
# - FS = 250 Hz; target length = 2500 samples (10 s).
# - Per-recording, per-lead z-normalization.
# - Polarity-agnostic R-peak detection using lead II.
# - Fine QRS alignment by largest absolute lead-II deflection near each detected peak.
# - One median representative beat per subject: each patient receives equal weight.
# - Group line = median across subjects.
# - Shading = interquartile range across subjects.
# - Female and male traces within a lead use ONE shared scaling factor.
#
# NO MODEL INFERENCE OR TRAINING.
# ==========================================================================================

from pathlib import Path
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import gaussian_filter1d
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")


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

RAW_ECG_ROOT = Path(
    "/athena/madelab/scratch/iqh4001/Azure_Cradle_ECG/"
    "Data/cradle-ecg/raw_ecg"
)

OUT_DIR = (
    RUN_DIR
    /
    "publication_figures"
    /
    "WCMC_AgeMatched_Female_vs_Male_Morphology"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

STEM = (
    OUT_DIR
    /
    "WCMC_LockedTest_AgeMatched_Female_vs_Male_12Lead_Morphology_MedianIQR"
)

PNG_OUT = Path(str(STEM) + ".png")
PDF_OUT = Path(str(STEM) + ".pdf")
SVG_OUT = Path(str(STEM) + ".svg")

MATCHED_PRE_QC_CSV = (
    OUT_DIR
    /
    "WCMC_AgeMatched_Female_vs_Male_PreQC_Cohort.csv"
)

SUBJECT_QC_CSV = (
    OUT_DIR
    /
    "WCMC_AgeMatched_Female_vs_Male_SubjectQC.csv"
)

MATCHED_FINAL_CSV = (
    OUT_DIR
    /
    "WCMC_AgeMatched_Female_vs_Male_FinalMatched_Cohort.csv"
)

AGE_AUDIT_CSV = (
    OUT_DIR
    /
    "WCMC_AgeMatched_Female_vs_Male_AgeAudit.csv"
)

TEMPLATE_NPZ = (
    OUT_DIR
    /
    "WCMC_AgeMatched_Female_vs_Male_Morphology_Templates.npz"
)


# ==========================================================================================
# CONFIG
# ==========================================================================================

FS = 250
TARGET_LEN = 2500

WINDOW_LEFT = int(
    0.40
    *
    FS
)

WINDOW_RIGHT = int(
    0.60
    *
    FS
)

BEAT_LEN = (
    WINDOW_LEFT
    +
    WINDOW_RIGHT
)

MIN_BEATS_PER_ECG = 2

RANDOM_SEED = 42

N_JOBS = int(
    os.environ.get(
        "WCMC_MORPH_N_JOBS",
        "8",
    )
)

# If None, use every possible matched subject.
# If an integer is set, cap each SEX after exact age matching while preserving age balance.
MAX_PER_SEX = None

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

LEAD_II_INDEX = 1

GROUPS = [
    "Female",
    "Male",
]

SEX_NAME = {
    0:
        "Female",

    1:
        "Male",
}

CMAP = plt.get_cmap(
    "plasma"
)

COLORS = {
    "Female":
        CMAP(
            0.22
        ),

    "Male":
        CMAP(
            0.82
        ),
}

SG_WINDOW = 21
SG_POLY = 3


# ==========================================================================================
# AUDIT INPUT
# ==========================================================================================

if not TEST_CSV.exists():
    raise FileNotFoundError(
        f"Missing WCMC locked test split:\n{TEST_CSV}"
    )

df = pd.read_csv(
    TEST_CSV,
    low_memory=False,
)

required_cols = [
    "patient_key",
    "ecg_key",
    "age_at_ecg",
    "sex_binary",
]

missing = [
    c
    for c in required_cols
    if c not in df.columns
]

if missing:
    raise RuntimeError(
        f"Locked test split missing columns: {missing}\n"
        f"Observed: {df.columns.tolist()}"
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
    "age_at_ecg"
] = pd.to_numeric(
    df[
        "age_at_ecg"
    ],
    errors="coerce",
)

df[
    "sex_binary"
] = pd.to_numeric(
    df[
        "sex_binary"
    ],
    errors="coerce",
)

df = df.dropna(
    subset=[
        "age_at_ecg",
        "sex_binary",
    ]
).copy()

df[
    "sex_binary"
] = df[
    "sex_binary"
].astype(
    int
)

df = df.loc[
    df[
        "sex_binary"
    ].isin(
        [
            0,
            1,
        ]
    )
].copy()

if df[
    "patient_key"
].duplicated().any():
    raise RuntimeError(
        "Locked WCMC test split contains duplicate patients."
    )

if df[
    "ecg_key"
].duplicated().any():
    raise RuntimeError(
        "Locked WCMC test split contains duplicate ECG keys."
    )

if "raw_local_path" not in df.columns:
    df[
        "raw_local_path"
    ] = df[
        "ecg_key"
    ].map(
        lambda x:
            str(
                RAW_ECG_ROOT
                /
                f"{x}.npy"
            )
    )

else:
    missing_path = (
        df[
            "raw_local_path"
        ].isna()
        |
        (
            df[
                "raw_local_path"
            ].astype(
                str
            ).str.strip()
            ==
            ""
        )
    )

    df.loc[
        missing_path,
        "raw_local_path",
    ] = df.loc[
        missing_path,
        "ecg_key",
    ].map(
        lambda x:
            str(
                RAW_ECG_ROOT
                /
                f"{x}.npy"
            )
    )


# ==========================================================================================
# AGE VARIABLE FOR MATCHING
# ==========================================================================================
#
# age_at_ecg in this locked cohort is integer-valued in the current run, but rounding makes
# the script robust if a future copy stores fractional age.
# ==========================================================================================

df[
    "age_match_year"
] = np.rint(
    df[
        "age_at_ecg"
    ]
).astype(
    int
)

df[
    "sex"
] = df[
    "sex_binary"
].map(
    SEX_NAME
)


# ==========================================================================================
# BEFORE-MATCHING SUMMARY
# ==========================================================================================

print(
    "=" * 120
)

print(
    "WCMC/NYP — AGE-MATCHED FEMALE vs MALE ECG MORPHOLOGY"
)

print(
    "=" * 120
)

print(
    f"\nLocked test N = {len(df):,}"
)

print(
    "\nSex counts before matching:"
)

print(
    df[
        "sex"
    ].value_counts().to_string()
)

print(
    "\nAge before matching:"
)

print(
    df.groupby(
        "sex"
    )[
        "age_at_ecg"
    ].agg(
        [
            "count",
            "mean",
            "std",
            "median",
            "min",
            "max",
        ]
    ).to_string()
)


# ==========================================================================================
# EXACT ONE-TO-ONE AGE MATCHING WITHOUT REPLACEMENT
# ==========================================================================================

def exact_age_match(
    source_df,
    random_seed,
):
    rng = np.random.default_rng(
        random_seed
    )

    female = source_df.loc[
        source_df[
            "sex_binary"
        ]
        ==
        0
    ].copy()

    male = source_df.loc[
        source_df[
            "sex_binary"
        ]
        ==
        1
    ].copy()

    ages = sorted(
        set(
            female[
                "age_match_year"
            ].unique()
        ).intersection(
            set(
                male[
                    "age_match_year"
                ].unique()
            )
        )
    )

    selected_f = []
    selected_m = []

    pair_id = 0

    for age in ages:
        f_age = female.loc[
            female[
                "age_match_year"
            ]
            ==
            age
        ].copy()

        m_age = male.loc[
            male[
                "age_match_year"
            ]
            ==
            age
        ].copy()

        n = min(
            len(
                f_age
            ),
            len(
                m_age
            ),
        )

        if n == 0:
            continue

        f_idx = rng.choice(
            f_age.index.to_numpy(),
            size=n,
            replace=False,
        )

        m_idx = rng.choice(
            m_age.index.to_numpy(),
            size=n,
            replace=False,
        )

        # Pair ordering within an age year is arbitrary because age is the only matching variable.
        f_sel = source_df.loc[
            f_idx
        ].copy().reset_index(
            drop=True
        )

        m_sel = source_df.loc[
            m_idx
        ].copy().reset_index(
            drop=True
        )

        f_sel[
            "match_pair_id"
        ] = np.arange(
            pair_id,
            pair_id
            +
            n,
        )

        m_sel[
            "match_pair_id"
        ] = np.arange(
            pair_id,
            pair_id
            +
            n,
        )

        pair_id += n

        selected_f.append(
            f_sel
        )

        selected_m.append(
            m_sel
        )

    if not selected_f:
        raise RuntimeError(
            "No exact-age female/male matches were found."
        )

    f = pd.concat(
        selected_f,
        ignore_index=True,
    )

    m = pd.concat(
        selected_m,
        ignore_index=True,
    )

    if MAX_PER_SEX is not None:
        # Preserve one-to-one pairing: sample pair IDs, then keep both subjects.
        available_pairs = np.intersect1d(
            f[
                "match_pair_id"
            ].unique(),
            m[
                "match_pair_id"
            ].unique(),
        )

        if len(
            available_pairs
        ) > MAX_PER_SEX:
            selected_pairs = rng.choice(
                available_pairs,
                size=MAX_PER_SEX,
                replace=False,
            )

            f = f.loc[
                f[
                    "match_pair_id"
                ].isin(
                    selected_pairs
                )
            ].copy()

            m = m.loc[
                m[
                    "match_pair_id"
                ].isin(
                    selected_pairs
                )
            ].copy()

    matched = pd.concat(
        [
            f,
            m,
        ],
        ignore_index=True,
    )

    matched = matched.sort_values(
        [
            "match_pair_id",
            "sex_binary",
        ]
    ).reset_index(
        drop=True
    )

    return matched


matched_pre = exact_age_match(
    df,
    RANDOM_SEED,
)

matched_pre.to_csv(
    MATCHED_PRE_QC_CSV,
    index=False,
)

n_f_pre = int(
    np.sum(
        matched_pre[
            "sex_binary"
        ]
        ==
        0
    )
)

n_m_pre = int(
    np.sum(
        matched_pre[
            "sex_binary"
        ]
        ==
        1
    )
)

if n_f_pre != n_m_pre:
    raise RuntimeError(
        f"Pre-QC matching failed balance: Female={n_f_pre}, Male={n_m_pre}"
    )

print(
    "\nExact-age matched before waveform QC:"
)

print(
    f"Female N = {n_f_pre:,}"
)

print(
    f"Male N   = {n_m_pre:,}"
)

print(
    "\nMatched age summary before waveform QC:"
)

print(
    matched_pre.groupby(
        "sex"
    )[
        "age_at_ecg"
    ].agg(
        [
            "count",
            "mean",
            "std",
            "median",
            "min",
            "max",
        ]
    ).to_string()
)


# ==========================================================================================
# ECG LOADING
# ==========================================================================================

def orient_and_resample(
    arr,
):
    x = np.asarray(
        arr,
        dtype=np.float32,
    )

    x = np.squeeze(
        x
    )

    if x.ndim != 2:
        raise ValueError(
            f"Expected 2D ECG; got {x.shape}"
        )

    if x.shape[
        0
    ] == 12:
        pass

    elif x.shape[
        1
    ] == 12:
        x = x.T

    else:
        raise ValueError(
            f"Could not infer 12-lead orientation: {x.shape}"
        )

    x = np.nan_to_num(
        x,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if x.shape[
        1
    ] != TARGET_LEN:
        old_t = np.linspace(
            0.0,
            1.0,
            x.shape[
                1
            ],
        )

        new_t = np.linspace(
            0.0,
            1.0,
            TARGET_LEN,
        )

        out = np.zeros(
            (
                12,
                TARGET_LEN,
            ),
            dtype=np.float32,
        )

        for lead_idx in range(
            12
        ):
            out[
                lead_idx
            ] = np.interp(
                new_t,
                old_t,
                x[
                    lead_idx
                ],
            )

        x = out

    return x.astype(
        np.float32,
        copy=False,
    )


def normalize_full_recording(
    x,
):
    mean = np.mean(
        x,
        axis=1,
        keepdims=True,
    )

    std = np.std(
        x,
        axis=1,
        keepdims=True,
    )

    std[
        std
        <
        1e-5
    ] = 1.0

    return (
        (
            x
            -
            mean
        )
        /
        std
    ).astype(
        np.float32
    )


def load_ecg(
    path,
):
    p = Path(
        str(
            path
        )
    )

    if not p.exists():
        raise FileNotFoundError(
            p
        )

    arr = np.load(
        p,
        allow_pickle=False,
    )

    x = orient_and_resample(
        arr
    )

    return normalize_full_recording(
        x
    )


# ==========================================================================================
# R-PEAK DETECTION
# ==========================================================================================

def robust_standardize(
    x,
):
    x = np.asarray(
        x,
        dtype=float,
    )

    med = np.median(
        x
    )

    mad = np.median(
        np.abs(
            x
            -
            med
        )
    )

    scale = (
        1.4826
        *
        mad
    )

    if (
        not np.isfinite(
            scale
        )
        or
        scale < 1e-8
    ):
        scale = np.std(
            x
        )

    if (
        not np.isfinite(
            scale
        )
        or
        scale < 1e-8
    ):
        scale = 1.0

    return (
        x
        -
        med
    ) / scale


def detect_r_peaks(
    lead_ii,
):
    z = robust_standardize(
        lead_ii
    )

    z = gaussian_filter1d(
        z,
        sigma=1.5,
    )

    distance = int(
        0.35
        *
        FS
    )

    p_pos, _ = find_peaks(
        z,
        distance=distance,
        prominence=0.45,
    )

    p_neg, _ = find_peaks(
        -z,
        distance=distance,
        prominence=0.45,
    )

    candidates = []

    for polarity, peaks in [
        (
            "positive",
            p_pos,
        ),
        (
            "negative",
            p_neg,
        ),
    ]:
        peaks = np.asarray(
            peaks,
            dtype=int,
        )

        peaks = peaks[
            (
                peaks
                >=
                WINDOW_LEFT
            )
            &
            (
                peaks
                +
                WINDOW_RIGHT
                <=
                TARGET_LEN
            )
        ]

        if len(
            peaks
        ) < MIN_BEATS_PER_ECG:
            continue

        rr = np.diff(
            peaks
        ) / FS

        rr_median = (
            float(
                np.median(
                    rr
                )
            )
            if len(
                rr
            )
            else
            np.nan
        )

        rr_cv = (
            float(
                np.std(
                    rr
                )
                /
                max(
                    np.mean(
                        rr
                    ),
                    1e-8,
                )
            )
            if len(
                rr
            )
            else
            np.inf
        )

        count_score = -abs(
            len(
                peaks
            )
            -
            10
        )

        score = (
            count_score
            -
            3.0
            *
            (
                rr_cv
                if np.isfinite(
                    rr_cv
                )
                else
                5.0
            )
        )

        if (
            np.isfinite(
                rr_median
            )
            and
            (
                rr_median < 0.35
                or
                rr_median > 1.8
            )
        ):
            score -= 2.0

        candidates.append(
            {
                "peaks":
                    peaks,

                "polarity":
                    polarity,

                "score":
                    score,

                "rr_median_s":
                    rr_median,

                "rr_cv":
                    rr_cv,
            }
        )

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda x:
            x[
                "score"
            ],
    )


# ==========================================================================================
# ONE REPRESENTATIVE BEAT PER SUBJECT
# ==========================================================================================

def extract_representative_beat(
    sig,
):
    peak_info = detect_r_peaks(
        sig[
            LEAD_II_INDEX
        ]
    )

    if peak_info is None:
        return None

    beats = []

    for p in peak_info[
        "peaks"
    ]:
        # Fine alignment around detected QRS by maximum absolute lead-II deflection.
        local_radius = int(
            0.08
            *
            FS
        )

        ls = max(
            0,
            int(
                p
                -
                local_radius
            ),
        )

        le = min(
            TARGET_LEN,
            int(
                p
                +
                local_radius
                +
                1
            ),
        )

        segment = sig[
            LEAD_II_INDEX,
            ls:le
        ]

        if segment.size == 0:
            continue

        center = (
            ls
            +
            int(
                np.argmax(
                    np.abs(
                        segment
                    )
                )
            )
        )

        s = (
            center
            -
            WINDOW_LEFT
        )

        e = (
            center
            +
            WINDOW_RIGHT
        )

        if (
            s < 0
            or
            e > TARGET_LEN
        ):
            continue

        beat = sig[
            :,
            s:e
        ].copy()

        if beat.shape != (
            12,
            BEAT_LEN,
        ):
            continue

        # Baseline correction only; DO NOT invert polarity.
        baseline_n = max(
            5,
            int(
                0.08
                *
                FS
            ),
        )

        baseline = np.median(
            beat[
                :,
                :baseline_n
            ],
            axis=1,
            keepdims=True,
        )

        beat = (
            beat
            -
            baseline
        )

        beats.append(
            beat.astype(
                np.float32
            )
        )

    if len(
        beats
    ) < MIN_BEATS_PER_ECG:
        return None

    beats = np.stack(
        beats,
        axis=0,
    )

    # Robust subject-level template: one subject = one observation.
    subject_beat = np.median(
        beats,
        axis=0,
    ).astype(
        np.float32
    )

    return {
        "beat":
            subject_beat,

        "n_beats":
            int(
                len(
                    beats
                )
            ),

        "qrs_polarity":
            peak_info[
                "polarity"
            ],

        "rr_median_s":
            peak_info[
                "rr_median_s"
            ],

        "rr_cv":
            peak_info[
                "rr_cv"
            ],
    }


# ==========================================================================================
# PARALLEL SUBJECT PROCESSING
# ==========================================================================================

def process_subject(
    row_dict,
):
    out = {
        "patient_key":
            row_dict[
                "patient_key"
            ],

        "ecg_key":
            row_dict[
                "ecg_key"
            ],

        "sex_binary":
            int(
                row_dict[
                    "sex_binary"
                ]
            ),

        "sex":
            row_dict[
                "sex"
            ],

        "age_at_ecg":
            float(
                row_dict[
                    "age_at_ecg"
                ]
            ),

        "age_match_year":
            int(
                row_dict[
                    "age_match_year"
                ]
            ),

        "match_pair_id":
            int(
                row_dict[
                    "match_pair_id"
                ]
            ),

        "raw_local_path":
            str(
                row_dict[
                    "raw_local_path"
                ]
            ),

        "status":
            "failed",

        "n_beats":
            0,

        "qrs_polarity":
            None,

        "rr_median_s":
            np.nan,

        "rr_cv":
            np.nan,

        "error":
            None,

        "beat":
            None,
    }

    try:
        sig = load_ecg(
            row_dict[
                "raw_local_path"
            ]
        )

        result = extract_representative_beat(
            sig
        )

        if result is None:
            raise RuntimeError(
                "No valid representative beat."
            )

        out[
            "status"
        ] = "ok"

        out[
            "n_beats"
        ] = result[
            "n_beats"
        ]

        out[
            "qrs_polarity"
        ] = result[
            "qrs_polarity"
        ]

        out[
            "rr_median_s"
        ] = result[
            "rr_median_s"
        ]

        out[
            "rr_cv"
        ] = result[
            "rr_cv"
        ]

        out[
            "beat"
        ] = result[
            "beat"
        ]

    except Exception as exc:
        out[
            "error"
        ] = repr(
            exc
        )

    return out


records = matched_pre.to_dict(
    orient="records"
)

print(
    f"\nProcessing {len(records):,} age-matched ECGs with N_JOBS={N_JOBS}..."
)

results = Parallel(
    n_jobs=N_JOBS,
    backend="loky",
    verbose=5,
)(
    delayed(
        process_subject
    )(
        row
    )
    for row in records
)


# ==========================================================================================
# QC TABLE + BEAT LOOKUP
# ==========================================================================================

qc_rows = []
beat_lookup = {}

for result in results:
    beat = result.pop(
        "beat"
    )

    qc_rows.append(
        result.copy()
    )

    if (
        result[
            "status"
        ]
        ==
        "ok"
        and
        beat is not None
    ):
        beat_lookup[
            result[
                "patient_key"
            ]
        ] = beat


qc_df = pd.DataFrame(
    qc_rows
)

qc_df.to_csv(
    SUBJECT_QC_CSV,
    index=False,
)

print(
    "\nWaveform QC summary:"
)

print(
    pd.crosstab(
        qc_df[
            "sex"
        ],
        qc_df[
            "status"
        ]
    ).to_string()
)


# ==========================================================================================
# POST-QC REMATCH — ENSURES FINAL COHORT REMAINS EXACTLY AGE MATCHED
# ==========================================================================================

valid = qc_df.loc[
    qc_df[
        "status"
    ]
    ==
    "ok"
].copy()

final_matched = exact_age_match(
    valid,
    RANDOM_SEED
    +
    1001,
)

final_matched.to_csv(
    MATCHED_FINAL_CSV,
    index=False,
)

n_f = int(
    np.sum(
        final_matched[
            "sex_binary"
        ]
        ==
        0
    )
)

n_m = int(
    np.sum(
        final_matched[
            "sex_binary"
        ]
        ==
        1
    )
)

if n_f != n_m:
    raise RuntimeError(
        f"Final matching imbalance: Female={n_f}, Male={n_m}"
    )

# Strong exact age-frequency audit.
age_tab = pd.crosstab(
    final_matched[
        "age_match_year"
    ],
    final_matched[
        "sex"
    ],
)

if not np.array_equal(
    age_tab[
        "Female"
    ].to_numpy(),
    age_tab[
        "Male"
    ].to_numpy(),
):
    raise RuntimeError(
        "Final female/male integer-age distributions are not identical."
    )


# ==========================================================================================
# FINAL AGE AUDIT
# ==========================================================================================

age_summary = (
    final_matched
    .groupby(
        "sex"
    )[
        "age_at_ecg"
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

year_frequency = (
    final_matched
    .groupby(
        [
            "age_match_year",
            "sex",
        ]
    )
    .size()
    .unstack(
        fill_value=0
    )
    .reset_index()
)

year_frequency.to_csv(
    AGE_AUDIT_CSV,
    index=False,
)

print(
    "\nFinal exact-age matched cohort after waveform QC:"
)

print(
    f"Female N = {n_f:,}"
)

print(
    f"Male N   = {n_m:,}"
)

print(
    "\nFinal age summary:"
)

print(
    age_summary.to_string(
        index=False
    )
)

print(
    "\nAge-frequency identity check: PASS"
)


# ==========================================================================================
# BUILD FINAL SUBJECT-LEVEL BEAT MATRICES
# ==========================================================================================

group_beats = {}

for group in GROUPS:
    sub = final_matched.loc[
        final_matched[
            "sex"
        ]
        ==
        group
    ].copy()

    beats = [
        beat_lookup[
            patient_key
        ]
        for patient_key in sub[
            "patient_key"
        ]
    ]

    group_beats[
        group
    ] = np.stack(
        beats,
        axis=0,
    ).astype(
        np.float32
    )


# ==========================================================================================
# GROUP MEDIAN + IQR
# ==========================================================================================

group_median = {}
group_q25 = {}
group_q75 = {}

for group in GROUPS:
    beats = group_beats[
        group
    ]

    group_median[
        group
    ] = np.median(
        beats,
        axis=0,
    )

    group_q25[
        group
    ] = np.percentile(
        beats,
        25,
        axis=0,
    )

    group_q75[
        group
    ] = np.percentile(
        beats,
        75,
        axis=0,
    )


# ==========================================================================================
# SHARED WITHIN-LEAD SCALING
# ==========================================================================================
#
# Female and male are scaled with the SAME factor in each lead.
# This preserves between-sex morphology differences within the panel.
# ==========================================================================================

lead_scale = np.ones(
    12,
    dtype=np.float32,
)

for lead_idx in range(
    12
):
    values = []

    for group in GROUPS:
        values.extend(
            [
                np.abs(
                    group_median[
                        group
                    ][
                        lead_idx
                    ]
                ),
                np.abs(
                    group_q25[
                        group
                    ][
                        lead_idx
                    ]
                ),
                np.abs(
                    group_q75[
                        group
                    ][
                        lead_idx
                    ]
                ),
            ]
        )

    values = np.concatenate(
        values
    )

    scale = np.percentile(
        values,
        99.5,
    )

    if (
        not np.isfinite(
            scale
        )
        or
        scale < 1e-8
    ):
        scale = 1.0

    lead_scale[
        lead_idx
    ] = scale


# ==========================================================================================
# TIME
# ==========================================================================================

time_ms = (
    np.arange(
        BEAT_LEN
    )
    -
    WINDOW_LEFT
) / FS * 1000.0


# ==========================================================================================
# SMOOTHING
# ==========================================================================================

def smooth(
    x,
):
    x = np.asarray(
        x,
        dtype=float,
    )

    if len(
        x
    ) >= SG_WINDOW:
        return savgol_filter(
            x,
            SG_WINDOW,
            SG_POLY,
        )

    return x


# ==========================================================================================
# STYLE
# ==========================================================================================

plt.style.use(
    "default"
)

plt.rcParams.update(
    {
        "figure.facecolor":
            "white",

        "axes.facecolor":
            "white",

        "axes.edgecolor":
            "black",

        "axes.linewidth":
            0.9,

        "font.family":
            "DejaVu Serif",

        "font.size":
            10.5,

        "axes.titlesize":
            11.2,

        "axes.labelsize":
            10.3,

        "xtick.labelsize":
            8.8,

        "ytick.labelsize":
            8.8,

        "legend.fontsize":
            9.0,

        "axes.spines.top":
            False,

        "axes.spines.right":
            False,

        "figure.dpi":
            120,

        "savefig.dpi":
            600,
    }
)


# ==========================================================================================
# FIGURE — 4 × 3
# ==========================================================================================

fig, axes = plt.subplots(
    4,
    3,
    figsize=(
        14.0,
        11.8,
    ),
    sharex=True,
)

axes = axes.ravel()

for lead_idx, ax in enumerate(
    axes
):
    scale = float(
        lead_scale[
            lead_idx
        ]
    )

    for group in GROUPS:
        median = smooth(
            group_median[
                group
            ][
                lead_idx
            ]
            /
            scale
        )

        q25 = smooth(
            group_q25[
                group
            ][
                lead_idx
            ]
            /
            scale
        )

        q75 = smooth(
            group_q75[
                group
            ][
                lead_idx
            ]
            /
            scale
        )

        ax.plot(
            time_ms,
            median,
            color=COLORS[
                group
            ],
            linewidth=2.2,
            label=(
                group
                if lead_idx
                ==
                0
                else
                None
            ),
            zorder=3,
        )

        ax.fill_between(
            time_ms,
            q25,
            q75,
            color=COLORS[
                group
            ],
            alpha=0.10,
            linewidth=0,
            zorder=2,
        )

    ax.axvline(
        0,
        color="0.50",
        linestyle="--",
        linewidth=0.65,
        alpha=0.70,
        zorder=1,
    )

    ax.axhline(
        0,
        color="0.55",
        linestyle="--",
        linewidth=0.45,
        alpha=0.35,
        zorder=1,
    )

    ax.grid(
        axis="x",
        alpha=0.12,
    )

    ax.set_title(
        LEADS[
            lead_idx
        ],
        fontweight="bold",
    )

    ax.set_xlim(
        time_ms[
            0
        ],
        time_ms[
            -1
        ],
    )

    ax.set_ylim(
        -1.45,
        1.45,
    )

    if lead_idx >= 9:
        ax.set_xlabel(
            "Time from R peak (ms)"
        )

    if lead_idx % 3 == 0:
        ax.set_ylabel(
            "Normalized amplitude"
        )


axes[
    0
].legend(
    frameon=False,
    loc="upper right",
)

fig.suptitle(
    "Age-matched female and male ECG morphology in the WCMC/NYP locked test cohort",
    fontsize=15.5,
    fontweight="bold",
    y=0.985,
)

female_age_mean = float(
    final_matched.loc[
        final_matched[
            "sex_binary"
        ]
        ==
        0,
        "age_at_ecg",
    ].mean()
)

male_age_mean = float(
    final_matched.loc[
        final_matched[
            "sex_binary"
        ]
        ==
        1,
        "age_at_ecg",
    ].mean()
)

fig.text(
    0.5,
    0.958,
    (
        f"Exact one-to-one matching within integer age; "
        f"Female N={n_f:,}, Male N={n_m:,}; "
        f"mean age {female_age_mean:.1f} vs {male_age_mean:.1f} years. "
        "Lines show subject-level median morphology; shading shows IQR."
    ),
    ha="center",
    va="top",
    fontsize=9.2,
)

fig.subplots_adjust(
    left=0.075,
    right=0.985,
    bottom=0.07,
    top=0.925,
    wspace=0.19,
    hspace=0.25,
)


# ==========================================================================================
# SAVE FIGURE
# ==========================================================================================

fig.savefig(
    PNG_OUT,
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
)

fig.savefig(
    PDF_OUT,
    bbox_inches="tight",
    facecolor="white",
)

fig.savefig(
    SVG_OUT,
    bbox_inches="tight",
    facecolor="white",
)

plt.show()


# ==========================================================================================
# SAVE NUMERIC TEMPLATES
# ==========================================================================================

np.savez_compressed(
    TEMPLATE_NPZ,

    time_ms=time_ms.astype(
        np.float32
    ),

    lead_names=np.asarray(
        LEADS
    ),

    lead_scale=lead_scale,

    female_median=group_median[
        "Female"
    ].astype(
        np.float32
    ),

    female_q25=group_q25[
        "Female"
    ].astype(
        np.float32
    ),

    female_q75=group_q75[
        "Female"
    ].astype(
        np.float32
    ),

    male_median=group_median[
        "Male"
    ].astype(
        np.float32
    ),

    male_q25=group_q25[
        "Male"
    ].astype(
        np.float32
    ),

    male_q75=group_q75[
        "Male"
    ].astype(
        np.float32
    ),
)


# ==========================================================================================
# FINAL SUMMARY
# ==========================================================================================

print(
    "\n"
    +
    "=" * 120
)

print(
    "FINAL AGE-MATCHED MORPHOLOGY COHORT"
)

print(
    "=" * 120
)

print(
    age_summary.to_string(
        index=False
    )
)

print(
    "\nSex counts:"
)

print(
    final_matched[
        "sex"
    ].value_counts().to_string()
)

print(
    "\nWaveform QC among pre-QC matched cohort:"
)

print(
    pd.crosstab(
        qc_df[
            "sex"
        ],
        qc_df[
            "status"
        ]
    ).to_string()
)

print(
    "\nSaved:"
)

print(
    PNG_OUT
)

print(
    PDF_OUT
)

print(
    SVG_OUT
)

print(
    MATCHED_PRE_QC_CSV
)

print(
    SUBJECT_QC_CSV
)

print(
    MATCHED_FINAL_CSV
)

print(
    AGE_AUDIT_CSV
)

print(
    TEMPLATE_NPZ
)

print(
    "\nDONE"
)
