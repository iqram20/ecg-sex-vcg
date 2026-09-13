#!/usr/bin/env python
# ==========================================================================================
# ECG-SEX — FINAL CURRENT-RUN SEX-SPECIFIC ECG ATTRIBUTION MAP
# VCG-GEOMETRY RESNET | CORRECTED DECOMPOSED IG
# ==========================================================================================
#
# PURPOSE
# -------
# Publication-quality beat-aligned male vs female ECG attribution signature for the
# FINAL five-seed VCG-Geometry ResNet ensemble.
#
# CURRENT-RUN INPUTS
# ------------------
# 22_vcg_geometry_resnet_corrected_ig/
#   VCGGeometryResNet_Corrected_DecomposedIG_N256.npz
#   CorrectedIG_N256_manifest.csv
#   CorrectedIG_QC_Summary.json
#   CorrectedIG_Waveform_LeadImportance.csv
#
# IMPORTANT DIFFERENCES FROM THE LEGACY SCRIPT
# --------------------------------------------
# 1. DOES NOT use legacy attribution_outputs/integrated_gradients/... files.
# 2. DOES NOT use the invalid initial VCG raw-input IG from run 21.
# 3. Uses ONLY corrected representation-level waveform IG that passed completeness QC.
# 4. Uses the exact balanced corrected-IG N=256 sample (128 female, 128 male expected).
# 5. Does NOT restrict to correctly classified subjects; avoids post-hoc selection bias.
# 6. Each ECG contributes ONE representative median beat, so subjects with more detected
#    beats do not receive more weight.
# 7. For each lead/time point, the raw-waveform and first-derivative representation
#    attributions are SUMMED:
#
#        lead_time_IG = raw_channel_IG + derivative_channel_IG
#
#    This preserves the 24-channel waveform-path attribution sum when collapsed to 12 leads.
# 8. Heatmap scaling is GLOBAL across leads/groups for cross-lead comparability.
#
# ATTRIBUTION SIGN
# ----------------
# The corrected IG target is the MALE LOGIT:
#     positive = contribution toward higher male logit
#     negative = contribution toward lower male logit / female direction
#
# Panels
# ------
# A. Normalized ECG morphology: male vs female
# B. Male-group signed male-logit attribution
# C. Female-group signed male-logit attribution
# D. Male − female signed attribution difference
#
# Secondary figure:
#   3×4 difference-only lead map
#
# NO MODEL INFERENCE OR TRAINING.
# ==========================================================================================

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings("ignore")


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

# R-peak detection tries these leads in order and chooses the best-quality candidate.
RPEAK_CANDIDATE_LEADS = [
    "II",
    "V2",
    "V3",
    "I",
]

SMOOTH_ECG = True
SMOOTH_ATTR_SIGMA = 1.5

# Robust global heat scaling percentile.
ATTR_COLOR_PERCENTILE = 99.0

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

LEAD_TO_INDEX = {
    lead:
        i
    for i, lead in enumerate(
        LEADS
    )
}

CMAP = "RdBu_r"


# ==========================================================================================
# PATHS — FINAL CURRENT RUN
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

CORRECTED_IG_DIR = (
    WCMC_ROOT
    /
    "22_vcg_geometry_resnet_corrected_ig"
)

NPZ_PATH = (
    CORRECTED_IG_DIR
    /
    "VCGGeometryResNet_Corrected_DecomposedIG_N256.npz"
)

MANIFEST_PATH = (
    CORRECTED_IG_DIR
    /
    "CorrectedIG_N256_manifest.csv"
)

QC_JSON = (
    CORRECTED_IG_DIR
    /
    "CorrectedIG_QC_Summary.json"
)

OFFICIAL_LEAD_CSV = (
    CORRECTED_IG_DIR
    /
    "CorrectedIG_Waveform_LeadImportance.csv"
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

OUT_DIR = (
    CORRECTED_IG_DIR
    /
    "sex_specific_beat_aligned_signature"
)

FIG_DIR = (
    OUT_DIR
    /
    "figures"
)

ARR_DIR = (
    OUT_DIR
    /
    "arrays"
)

TAB_DIR = (
    OUT_DIR
    /
    "tables"
)

for d in [
    OUT_DIR,
    FIG_DIR,
    ARR_DIR,
    TAB_DIR,
]:
    d.mkdir(
        parents=True,
        exist_ok=True,
    )


# ==========================================================================================
# OUTPUTS
# ==========================================================================================

MAIN_PNG = (
    FIG_DIR
    /
    "Figure_FinalVCG_SexSpecific_SignedAttribution_12Lead.png"
)

MAIN_PDF = (
    FIG_DIR
    /
    "Figure_FinalVCG_SexSpecific_SignedAttribution_12Lead.pdf"
)

MAIN_SVG = (
    FIG_DIR
    /
    "Figure_FinalVCG_SexSpecific_SignedAttribution_12Lead.svg"
)

DIFF_PNG = (
    FIG_DIR
    /
    "Figure_FinalVCG_SexSpecific_DifferenceOnly_3x4.png"
)

DIFF_PDF = (
    FIG_DIR
    /
    "Figure_FinalVCG_SexSpecific_DifferenceOnly_3x4.pdf"
)

SUMMARY_CSV = (
    TAB_DIR
    /
    "Figure_FinalVCG_SexSpecific_LeadSummary.csv"
)

SUBJECT_AUDIT_CSV = (
    TAB_DIR
    /
    "Figure_FinalVCG_SexSpecific_SubjectBeatAudit.csv"
)

ARRAYS_NPZ = (
    ARR_DIR
    /
    "Figure_FinalVCG_SexSpecific_BeatAlignedArrays.npz"
)


# ==========================================================================================
# REQUIRED FILE AUDIT
# ==========================================================================================

required_files = [
    NPZ_PATH,
    MANIFEST_PATH,
    QC_JSON,
    OFFICIAL_LEAD_CSV,
    TEST_CSV,
]

missing = [
    p
    for p in required_files
    if not p.exists()
]

if missing:
    raise FileNotFoundError(
        "Missing required FINAL current-run file(s):\n"
        +
        "\n".join(
            str(p)
            for p in missing
        )
    )


# ==========================================================================================
# COMPLETENESS QC — HARD GATE
# ==========================================================================================

with open(
    QC_JSON,
    "r",
) as f:
    qc = json.load(
        f
    )

if not bool(
    qc.get(
        "overall_pass",
        False,
    )
):
    raise RuntimeError(
        "Corrected IG completeness QC did not pass. "
        "Do not generate publication attribution figures."
    )

print(
    "=" * 120
)

print(
    "FINAL VCG-GEOMETRY RESNET — SEX-SPECIFIC BEAT-ALIGNED ATTRIBUTION"
)

print(
    "=" * 120
)

print(
    "\nCorrected IG QC: PASS"
)

for key in [
    "waveform_median_abs_error",
    "waveform_p95_abs_error",
    "geometry_median_abs_error",
    "geometry_p95_abs_error",
]:
    if key in qc:
        print(
            f"{key}: {qc[key]}"
        )


# ==========================================================================================
# LOAD CORRECTED IG
# ==========================================================================================

pkg = np.load(
    NPZ_PATH,
    allow_pickle=True,
)

required_npz_keys = [
    "waveform_ig_signed_male_logit",
    "y_true",
    "patient_key",
    "ecg_key",
    "waveform_channel_names",
]

missing_npz = [
    key
    for key in required_npz_keys
    if key not in pkg.files
]

if missing_npz:
    raise RuntimeError(
        f"Corrected IG NPZ missing keys: {missing_npz}\n"
        f"Observed keys: {pkg.files}"
    )

wave_ig_24 = np.asarray(
    pkg[
        "waveform_ig_signed_male_logit"
    ],
    dtype=np.float32,
)

y_true = np.asarray(
    pkg[
        "y_true"
    ],
    dtype=np.int8,
)

patient_keys = np.asarray(
    pkg[
        "patient_key"
    ]
).astype(
    str
)

ecg_keys = np.asarray(
    pkg[
        "ecg_key"
    ]
).astype(
    str
)

wave_channel_names = [
    str(
        x
    )
    for x in pkg[
        "waveform_channel_names"
    ].tolist()
]

if wave_ig_24.ndim != 3:
    raise RuntimeError(
        f"Expected waveform IG shape [N,24,T]; found {wave_ig_24.shape}"
    )

if wave_ig_24.shape[
    1
] != 24:
    raise RuntimeError(
        f"Expected 24 waveform representation channels; found {wave_ig_24.shape}"
    )

if wave_ig_24.shape[
    2
] != TARGET_LEN:
    raise RuntimeError(
        f"Expected T={TARGET_LEN}; found {wave_ig_24.shape}"
    )

n_ig = wave_ig_24.shape[
    0
]

if not (
    len(
        y_true
    )
    ==
    len(
        patient_keys
    )
    ==
    len(
        ecg_keys
    )
    ==
    n_ig
):
    raise RuntimeError(
        "Corrected IG metadata lengths do not align."
    )

expected_channels = (
    LEADS
    +
    [
        f"d{lead}"
        for lead in LEADS
    ]
)

if wave_channel_names != expected_channels:
    raise RuntimeError(
        "Unexpected corrected waveform channel order.\n"
        f"Expected: {expected_channels}\n"
        f"Observed: {wave_channel_names}"
    )

print(
    "\nCorrected waveform IG:"
)

print(
    wave_ig_24.shape
)

print(
    "Female:",
    int(
        np.sum(
            y_true
            ==
            0
        )
    ),
)

print(
    "Male:",
    int(
        np.sum(
            y_true
            ==
            1
        )
    ),
)


# ==========================================================================================
# COLLAPSE 24 REPRESENTATION CHANNELS -> 12 LEAD-TIME ATTRIBUTION MAPS
# ==========================================================================================
#
# For each lead:
#   0:12   = z-scored waveform attribution
#   12:24  = first-derivative attribution
#
# Summation at matching time indices preserves the total signed waveform-path attribution
# when collapsed over the 24 input channels.
# ==========================================================================================

lead_time_ig = (
    wave_ig_24[
        :,
        :12,
        :
    ]
    +
    wave_ig_24[
        :,
        12:24,
        :
    ]
).astype(
    np.float32
)

# Numerical identity audit.
original_sum = np.sum(
    wave_ig_24,
    axis=(
        1,
        2,
    ),
)

collapsed_sum = np.sum(
    lead_time_ig,
    axis=(
        1,
        2,
    ),
)

collapse_error = np.max(
    np.abs(
        original_sum
        -
        collapsed_sum
    )
)

print(
    f"\n24ch→12lead attribution-collapse max absolute sum error: {collapse_error:.3e}"
)

if collapse_error > 1e-4:
    raise RuntimeError(
        "Unexpected attribution-collapse numerical mismatch."
    )


# ==========================================================================================
# ALIGN CORRECTED IG ROWS TO RAW ECG PATHS
# ==========================================================================================

manifest = pd.read_csv(
    MANIFEST_PATH,
    low_memory=False,
)

test_df = pd.read_csv(
    TEST_CSV,
    low_memory=False,
)

for df, label in [
    (
        manifest,
        "Corrected manifest",
    ),
    (
        test_df,
        "Locked test split",
    ),
]:
    for col in [
        "patient_key",
        "ecg_key",
    ]:
        if col not in df.columns:
            raise RuntimeError(
                f"{label} missing {col}"
            )

        df[
            col
        ] = df[
            col
        ].astype(
            str
        )


# Manifest should normally already contain raw_local_path.
if (
    "raw_local_path"
    not in manifest.columns
    or
    manifest[
        "raw_local_path"
    ].isna().any()
):
    if "raw_local_path" not in test_df.columns:
        raise RuntimeError(
            "Neither corrected manifest nor locked test split contains raw_local_path."
        )

    manifest = (
        manifest.drop(
            columns=[
                "raw_local_path"
            ],
            errors="ignore",
        )
        .merge(
            test_df[
                [
                    "patient_key",
                    "ecg_key",
                    "raw_local_path",
                ]
            ],
            on=[
                "patient_key",
                "ecg_key",
            ],
            how="left",
            validate="one_to_one",
        )
    )


ig_meta = pd.DataFrame(
    {
        "ig_index":
            np.arange(
                n_ig,
                dtype=int,
            ),

        "patient_key":
            patient_keys,

        "ecg_key":
            ecg_keys,

        "sex_true_binary":
            y_true.astype(
                int
            ),
    }
)

aligned = ig_meta.merge(
    manifest[
        [
            "patient_key",
            "ecg_key",
            "raw_local_path",
        ]
    ],
    on=[
        "patient_key",
        "ecg_key",
    ],
    how="left",
    validate="one_to_one",
)

aligned = aligned.sort_values(
    "ig_index"
).reset_index(
    drop=True
)

if len(
    aligned
) != n_ig:
    raise RuntimeError(
        f"IG/manifest alignment mismatch: {len(aligned)} vs {n_ig}"
    )

if aligned[
    "raw_local_path"
].isna().any():
    bad = aligned.loc[
        aligned[
            "raw_local_path"
        ].isna(),
        [
            "patient_key",
            "ecg_key",
        ],
    ]

    raise RuntimeError(
        "Missing raw ECG paths for corrected IG cases:\n"
        +
        bad.to_string(
            index=False
        )
    )


# ==========================================================================================
# RAW ECG ORIENTATION / RESAMPLING
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
            f"Cannot infer 12-lead orientation from {arr.shape}"
        )

    if x.shape[
        1
    ] != TARGET_LEN:
        old_grid = np.linspace(
            0.0,
            1.0,
            x.shape[
                1
            ],
        )

        new_grid = np.linspace(
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
                new_grid,
                old_grid,
                x[
                    lead_idx
                ],
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


# ==========================================================================================
# R-PEAK DETECTION
# ==========================================================================================

def robust_standardize_1d(
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


def candidate_r_peaks(
    signal_1d,
):
    z = robust_standardize_1d(
        signal_1d
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

    # Examine both polarities because QRS polarity depends on lead/patient.
    p_pos, props_pos = find_peaks(
        z,
        distance=distance,
        prominence=0.45,
    )

    p_neg, props_neg = find_peaks(
        -z,
        distance=distance,
        prominence=0.45,
    )

    candidates = []

    for polarity, peaks, props in [
        (
            "positive",
            p_pos,
            props_pos,
        ),
        (
            "negative",
            p_neg,
            props_neg,
        ),
    ]:
        peaks = np.asarray(
            peaks,
            dtype=int,
        )

        # Only peaks that support a complete beat window.
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

        if len(
            rr
        ):
            rr_med = float(
                np.median(
                    rr
                )
            )

            rr_cv = float(
                np.std(
                    rr
                )
                /
                max(
                    np.mean(
                        rr
                    ),
                    1e-6,
                )
            )

        else:
            rr_med = np.nan
            rr_cv = np.inf

        # Physiologically plausible count/regularity preference.
        count = len(
            peaks
        )

        count_score = -abs(
            count
            -
            10
        )

        rr_penalty = (
            rr_cv
            if np.isfinite(
                rr_cv
            )
            else
            5.0
        )

        if (
            np.isfinite(
                rr_med
            )
            and
            (
                rr_med < 0.35
                or
                rr_med > 1.8
            )
        ):
            rr_penalty += 1.5

        score = (
            count_score
            -
            3.0
            *
            rr_penalty
        )

        candidates.append(
            {
                "polarity":
                    polarity,

                "peaks":
                    peaks,

                "score":
                    score,

                "rr_median_s":
                    rr_med,

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


def detect_best_r_peaks(
    ecg12,
):
    best = None

    for lead_name in RPEAK_CANDIDATE_LEADS:
        lead_idx = LEAD_TO_INDEX[
            lead_name
        ]

        candidate = candidate_r_peaks(
            ecg12[
                lead_idx
            ]
        )

        if candidate is None:
            continue

        candidate = dict(
            candidate
        )

        candidate[
            "reference_lead"
        ] = lead_name

        if (
            best is None
            or
            candidate[
                "score"
            ]
            >
            best[
                "score"
            ]
        ):
            best = candidate

    return best


# ==========================================================================================
# BEAT NORMALIZATION / EXTRACTION
# ==========================================================================================

def normalize_ecg_beat_for_display(
    beat,
):
    """
    Normalize each lead within an individual beat for morphology visualization.
    This deliberately emphasizes shape rather than absolute acquisition amplitude.
    """
    beat = np.asarray(
        beat,
        dtype=np.float32,
    ).copy()

    out = np.zeros_like(
        beat,
        dtype=np.float32,
    )

    for lead_idx in range(
        12
    ):
        x = beat[
            lead_idx
        ].astype(
            float
        )

        center = np.median(
            x
        )

        scale = np.percentile(
            np.abs(
                x
                -
                center
            ),
            99,
        )

        if (
            not np.isfinite(
                scale
            )
            or
            scale < 1e-8
        ):
            scale = 1.0

        out[
            lead_idx
        ] = (
            x
            -
            center
        ) / scale

    return out


def extract_ecg_representative_beat(
    raw_ecg,
    lead_attr,
):
    """
    Returns one median ECG beat + one median attribution beat per ECG.
    Thus the statistical/display unit is the ECG/subject, not the number of beats.
    """
    peak_info = detect_best_r_peaks(
        raw_ecg
    )

    if peak_info is None:
        return None

    peaks = peak_info[
        "peaks"
    ]

    ecg_beats = []
    attr_beats = []

    for p in peaks:
        s = int(
            p
            -
            WINDOW_LEFT
        )

        e = int(
            p
            +
            WINDOW_RIGHT
        )

        if (
            s < 0
            or
            e > TARGET_LEN
        ):
            continue

        ecg_beat = raw_ecg[
            :,
            s:e
        ]

        attr_beat = lead_attr[
            :,
            s:e
        ]

        if (
            ecg_beat.shape[
                1
            ]
            !=
            BEAT_LEN
            or
            attr_beat.shape[
                1
            ]
            !=
            BEAT_LEN
        ):
            continue

        ecg_beats.append(
            normalize_ecg_beat_for_display(
                ecg_beat
            )
        )

        attr_beats.append(
            attr_beat.astype(
                np.float32
            )
        )

    if len(
        ecg_beats
    ) < MIN_BEATS_PER_ECG:
        return None

    ecg_beats = np.stack(
        ecg_beats,
        axis=0,
    )

    attr_beats = np.stack(
        attr_beats,
        axis=0,
    )

    return {
        "ecg":
            np.median(
                ecg_beats,
                axis=0,
            ).astype(
                np.float32
            ),

        "attr":
            np.median(
                attr_beats,
                axis=0,
            ).astype(
                np.float32
            ),

        "n_beats":
            int(
                len(
                    ecg_beats
                )
            ),

        "reference_lead":
            peak_info[
                "reference_lead"
            ],

        "polarity":
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
# PROCESS ALL CORRECTED-IG SUBJECTS
# ==========================================================================================

subject_ecg_beats = []
subject_attr_beats = []
audit_rows = []

for row_idx, row in aligned.iterrows():
    ig_idx = int(
        row[
            "ig_index"
        ]
    )

    raw_path = Path(
        str(
            row[
                "raw_local_path"
            ]
        )
    )

    audit = {
        "ig_index":
            ig_idx,

        "patient_key":
            row[
                "patient_key"
            ],

        "ecg_key":
            row[
                "ecg_key"
            ],

        "sex_true_binary":
            int(
                row[
                    "sex_true_binary"
                ]
            ),

        "raw_local_path":
            str(
                raw_path
            ),

        "status":
            "failed",

        "n_beats":
            0,

        "reference_lead":
            None,

        "qrs_polarity":
            None,

        "rr_median_s":
            np.nan,

        "rr_cv":
            np.nan,

        "error":
            None,
    }

    try:
        if not raw_path.exists():
            raise FileNotFoundError(
                raw_path
            )

        raw = np.load(
            raw_path,
            allow_pickle=False,
        )

        ecg12 = orient_and_resample(
            raw
        )

        result = extract_ecg_representative_beat(
            ecg12,
            lead_time_ig[
                ig_idx
            ],
        )

        if result is None:
            raise RuntimeError(
                "Insufficient valid beat alignment."
            )

        subject_ecg_beats.append(
            result[
                "ecg"
            ]
        )

        subject_attr_beats.append(
            result[
                "attr"
            ]
        )

        audit[
            "status"
        ] = "ok"

        audit[
            "n_beats"
        ] = result[
            "n_beats"
        ]

        audit[
            "reference_lead"
        ] = result[
            "reference_lead"
        ]

        audit[
            "qrs_polarity"
        ] = result[
            "polarity"
        ]

        audit[
            "rr_median_s"
        ] = result[
            "rr_median_s"
        ]

        audit[
            "rr_cv"
        ] = result[
            "rr_cv"
        ]

    except Exception as exc:
        audit[
            "error"
        ] = repr(
            exc
        )

    audit_rows.append(
        audit
    )

    if (
        (
            row_idx
            +
            1
        )
        %
        32
        ==
        0
        or
        (
            row_idx
            +
            1
        )
        ==
        n_ig
    ):
        print(
            f"Beat alignment: {row_idx + 1}/{n_ig}"
        )


audit_df = pd.DataFrame(
    audit_rows
)

audit_df.to_csv(
    SUBJECT_AUDIT_CSV,
    index=False,
)

ok_audit = audit_df.loc[
    audit_df[
        "status"
    ]
    ==
    "ok"
].copy()

if len(
    ok_audit
) != len(
    subject_ecg_beats
):
    raise RuntimeError(
        "Internal beat-audit alignment mismatch."
    )

if len(
    ok_audit
) < 0.90 * n_ig:
    raise RuntimeError(
        f"Too many beat-alignment failures: "
        f"{len(ok_audit)}/{n_ig} successful."
    )

subject_ecg_beats = np.stack(
    subject_ecg_beats,
    axis=0,
).astype(
    np.float32
)

subject_attr_beats = np.stack(
    subject_attr_beats,
    axis=0,
).astype(
    np.float32
)

subject_y = ok_audit[
    "sex_true_binary"
].to_numpy(
    dtype=np.int8,
)

female_mask = (
    subject_y
    ==
    0
)

male_mask = (
    subject_y
    ==
    1
)

n_female = int(
    np.sum(
        female_mask
    )
)

n_male = int(
    np.sum(
        male_mask
    )
)

if (
    n_female < 100
    or
    n_male < 100
):
    raise RuntimeError(
        f"Unexpected sex counts after beat alignment: "
        f"Female={n_female}, Male={n_male}"
    )


print(
    "\nBeat alignment successful:"
)

print(
    f"N={len(ok_audit)} | Female={n_female} | Male={n_male}"
)

print(
    "\nR-peak reference lead usage:"
)

print(
    ok_audit[
        "reference_lead"
    ].value_counts().to_string()
)


# ==========================================================================================
# GROUP REPRESENTATIVE MAPS
# ==========================================================================================

# Subject-level median, then group median: equal weight per subject.
male_ecg = np.median(
    subject_ecg_beats[
        male_mask
    ],
    axis=0,
)

female_ecg = np.median(
    subject_ecg_beats[
        female_mask
    ],
    axis=0,
)

male_attr = np.median(
    subject_attr_beats[
        male_mask
    ],
    axis=0,
)

female_attr = np.median(
    subject_attr_beats[
        female_mask
    ],
    axis=0,
)

diff_ecg = (
    male_ecg
    -
    female_ecg
)

diff_attr = (
    male_attr
    -
    female_attr
)


# ==========================================================================================
# GLOBAL ATTRIBUTION SCALE
# ==========================================================================================

global_attr_values = np.concatenate(
    [
        np.abs(
            male_attr
        ).ravel(),

        np.abs(
            female_attr
        ).ravel(),

        np.abs(
            diff_attr
        ).ravel(),
    ]
)

ATTR_VMAX = float(
    np.percentile(
        global_attr_values,
        ATTR_COLOR_PERCENTILE,
    )
)

if (
    not np.isfinite(
        ATTR_VMAX
    )
    or
    ATTR_VMAX < 1e-12
):
    ATTR_VMAX = 1.0

print(
    f"\nGlobal signed-attribution color scale ±{ATTR_VMAX:.3e}"
)


# ==========================================================================================
# OFFICIAL CORRECTED LEAD IMPORTANCE
# ==========================================================================================

official = pd.read_csv(
    OFFICIAL_LEAD_CSV,
    low_memory=False,
)

required_official = [
    "lead",
    "mean_abs_IG",
    "normalized_importance",
    "rank",
]

missing_official = [
    c
    for c in required_official
    if c not in official.columns
]

if missing_official:
    raise RuntimeError(
        f"Official corrected lead table missing columns: {missing_official}"
    )


# ==========================================================================================
# LEAD SUMMARY TABLE
# ==========================================================================================

summary_rows = []

time_ms = (
    np.arange(
        BEAT_LEN
    )
    -
    WINDOW_LEFT
) / FS * 1000.0

for lead_idx, lead_name in enumerate(
    LEADS
):
    m = male_attr[
        lead_idx
    ]

    f = female_attr[
        lead_idx
    ]

    d = diff_attr[
        lead_idx
    ]

    peak_idx = int(
        np.argmax(
            np.abs(
                d
            )
        )
    )

    summary_rows.append(
        {
            "lead":
                lead_name,

            "male_group_mean_signed_male_logit_attr":
                float(
                    np.mean(
                        m
                    )
                ),

            "female_group_mean_signed_male_logit_attr":
                float(
                    np.mean(
                        f
                    )
                ),

            "male_minus_female_mean_signed_attr":
                float(
                    np.mean(
                        d
                    )
                ),

            "male_group_mean_abs_attr":
                float(
                    np.mean(
                        np.abs(
                            m
                        )
                    )
                ),

            "female_group_mean_abs_attr":
                float(
                    np.mean(
                        np.abs(
                            f
                        )
                    )
                ),

            "male_minus_female_mean_abs_attr":
                float(
                    np.mean(
                        np.abs(
                            d
                        )
                    )
                ),

            "male_minus_female_peak_abs_attr":
                float(
                    np.max(
                        np.abs(
                            d
                        )
                    )
                ),

            "peak_difference_time_ms":
                float(
                    time_ms[
                        peak_idx
                    ]
                ),
        }
    )

summary_df = pd.DataFrame(
    summary_rows
)

summary_df = summary_df.merge(
    official[
        [
            "lead",
            "mean_abs_IG",
            "normalized_importance",
            "rank",
        ]
    ].rename(
        columns={
            "mean_abs_IG":
                "official_corrected_mean_abs_IG",

            "normalized_importance":
                "official_corrected_normalized_importance",

            "rank":
                "official_corrected_rank",
        }
    ),
    on="lead",
    how="left",
    validate="one_to_one",
)

summary_df = summary_df.sort_values(
    "official_corrected_rank"
).reset_index(
    drop=True
)

summary_df.to_csv(
    SUMMARY_CSV,
    index=False,
)


# ==========================================================================================
# SAVE BEAT-ALIGNED ARRAYS
# ==========================================================================================

np.savez_compressed(
    ARRAYS_NPZ,

    beat_time_ms=time_ms.astype(
        np.float32
    ),

    male_ecg_median=male_ecg.astype(
        np.float32
    ),

    female_ecg_median=female_ecg.astype(
        np.float32
    ),

    male_minus_female_ecg=diff_ecg.astype(
        np.float32
    ),

    male_signed_male_logit_attr_median=male_attr.astype(
        np.float32
    ),

    female_signed_male_logit_attr_median=female_attr.astype(
        np.float32
    ),

    male_minus_female_signed_attr=diff_attr.astype(
        np.float32
    ),

    subject_ecg_representative_beats=subject_ecg_beats,

    subject_signed_attr_representative_beats=subject_attr_beats,

    subject_y_true=subject_y,

    patient_key=ok_audit[
        "patient_key"
    ].astype(
        str
    ).to_numpy(),

    ecg_key=ok_audit[
        "ecg_key"
    ].astype(
        str
    ).to_numpy(),

    lead_names=np.asarray(
        LEADS
    ),

    attr_global_vmax=np.asarray(
        [
            ATTR_VMAX
        ],
        dtype=np.float32,
    ),
)


# ==========================================================================================
# PLOTTING HELPERS
# ==========================================================================================

def smooth_wave(
    y,
):
    y = np.asarray(
        y,
        dtype=float,
    )

    if (
        SMOOTH_ECG
        and
        len(
            y
        )
        >=
        21
    ):
        return savgol_filter(
            y,
            21,
            3,
        )

    return y


def smoothed_attr(
    a,
):
    return gaussian_filter1d(
        np.asarray(
            a,
            dtype=float,
        ),
        sigma=SMOOTH_ATTR_SIGMA,
    )


def plot_morphology(
    ax,
    t,
    male_y,
    female_y,
    title=None,
):
    male_y = smooth_wave(
        male_y
    )

    female_y = smooth_wave(
        female_y
    )

    ymax = max(
        np.max(
            np.abs(
                male_y
            )
        ),
        np.max(
            np.abs(
                female_y
            )
        ),
        1e-6,
    ) * 1.25

    ax.plot(
        t,
        male_y,
        linewidth=1.15,
        label="Male",
    )

    ax.plot(
        t,
        female_y,
        linewidth=1.15,
        label="Female",
    )

    ax.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=0.7,
        alpha=0.65,
    )

    ax.axhline(
        0,
        color="0.5",
        linewidth=0.35,
        alpha=0.5,
    )

    ax.set_xlim(
        t[
            0
        ],
        t[
            -1
        ],
    )

    ax.set_ylim(
        -ymax,
        ymax,
    )

    ax.set_yticks(
        []
    )

    if title is not None:
        ax.set_title(
            title,
            fontsize=9.5,
            fontweight="bold",
            loc="left",
        )


def plot_wave_attr(
    ax,
    t,
    waveform,
    attr,
    vmax,
    title=None,
):
    waveform = smooth_wave(
        waveform
    )

    attr = smoothed_attr(
        attr
    )

    ymax = max(
        np.max(
            np.abs(
                waveform
            )
        )
        *
        1.25,
        1e-6,
    )

    heat = np.tile(
        attr,
        (
            70,
            1,
        ),
    )

    ax.imshow(
        heat,
        aspect="auto",
        cmap=CMAP,
        extent=[
            t[
                0
            ],
            t[
                -1
            ],
            -ymax,
            ymax,
        ],
        origin="lower",
        vmin=-vmax,
        vmax=vmax,
        alpha=0.90,
        interpolation="bilinear",
    )

    ax.plot(
        t,
        waveform,
        color="black",
        linewidth=1.0,
    )

    ax.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=0.7,
        alpha=0.70,
    )

    ax.axhline(
        0,
        color="black",
        linewidth=0.35,
        alpha=0.35,
    )

    ax.set_xlim(
        t[
            0
        ],
        t[
            -1
        ],
    )

    ax.set_yticks(
        []
    )

    if title is not None:
        ax.set_title(
            title,
            fontsize=9.5,
            fontweight="bold",
            loc="left",
        )


# ==========================================================================================
# STYLE
# ==========================================================================================

plt.rcParams.update(
    {
        "font.family":
            "DejaVu Sans",

        "font.size":
            9.2,

        "axes.titlesize":
            9.5,

        "axes.labelsize":
            9.4,

        "xtick.labelsize":
            8.0,

        "ytick.labelsize":
            8.0,

        "legend.fontsize":
            8.0,

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
# MAIN 12 × 4 FIGURE
# ==========================================================================================

fig = plt.figure(
    figsize=(
        17.5,
        21.5,
    )
)

# One dedicated title row + 12 lead rows.
gs = fig.add_gridspec(
    13,
    4,
    height_ratios=[
        0.55
    ]
    +
    [
        1.0
    ]
    *
    12,
    width_ratios=[
        0.95,
        1.05,
        1.05,
        1.05,
    ],
    hspace=0.19,
    wspace=0.07,
)

column_titles = [
    "Normalized ECG morphology\nMale vs female",
    "Male group\nsigned male-logit attribution",
    "Female group\nsigned male-logit attribution",
    "Sex difference\nMale − female attribution",
]

for col, title in enumerate(
    column_titles
):
    ax_title = fig.add_subplot(
        gs[
            0,
            col
        ]
    )

    ax_title.axis(
        "off"
    )

    ax_title.text(
        0.5,
        0.45,
        title,
        ha="center",
        va="center",
        fontsize=11.8,
        fontweight="bold",
        transform=ax_title.transAxes,
    )


for lead_idx, lead_name in enumerate(
    LEADS
):
    row = (
        lead_idx
        +
        1
    )

    ax0 = fig.add_subplot(
        gs[
            row,
            0
        ]
    )

    ax1 = fig.add_subplot(
        gs[
            row,
            1
        ]
    )

    ax2 = fig.add_subplot(
        gs[
            row,
            2
        ]
    )

    ax3 = fig.add_subplot(
        gs[
            row,
            3
        ]
    )

    plot_morphology(
        ax0,
        time_ms,
        male_ecg[
            lead_idx
        ],
        female_ecg[
            lead_idx
        ],
        title=lead_name,
    )

    plot_wave_attr(
        ax1,
        time_ms,
        male_ecg[
            lead_idx
        ],
        male_attr[
            lead_idx
        ],
        ATTR_VMAX,
        title=lead_name,
    )

    plot_wave_attr(
        ax2,
        time_ms,
        female_ecg[
            lead_idx
        ],
        female_attr[
            lead_idx
        ],
        ATTR_VMAX,
        title=lead_name,
    )

    plot_wave_attr(
        ax3,
        time_ms,
        diff_ecg[
            lead_idx
        ],
        diff_attr[
            lead_idx
        ],
        ATTR_VMAX,
        title=lead_name,
    )

    if lead_idx == 0:
        ax0.legend(
            frameon=False,
            loc="upper right",
        )

    for ax in [
        ax0,
        ax1,
        ax2,
        ax3,
    ]:
        if lead_idx < 11:
            ax.set_xticklabels(
                []
            )
        else:
            ax.set_xlabel(
                "Time from R peak (ms)"
            )


fig.suptitle(
    "Sex-specific ECG attribution signature — final VCG-Geometry ResNet",
    fontsize=18,
    fontweight="bold",
    y=0.995,
)

fig.text(
    0.5,
    0.974,
    (
        f"Corrected decomposed IG; balanced WCMC sample; "
        f"equal subject weighting | Female N={n_female}, Male N={n_male}"
    ),
    ha="center",
    va="top",
    fontsize=10.5,
)

# Column labels.
for x, label in zip(
    [
        0.035,
        0.275,
        0.515,
        0.755,
    ],
    [
        "A",
        "B",
        "C",
        "D",
    ],
):
    fig.text(
        x,
        0.952,
        label,
        fontsize=16,
        fontweight="bold",
    )


# Shared colorbar for B-D only.
cax = fig.add_axes(
    [
        0.925,
        0.16,
        0.014,
        0.67,
    ]
)

sm = plt.cm.ScalarMappable(
    cmap=CMAP,
    norm=plt.Normalize(
        vmin=-ATTR_VMAX,
        vmax=ATTR_VMAX,
    ),
)

sm.set_array(
    []
)

cb = fig.colorbar(
    sm,
    cax=cax,
)

cb.set_label(
    "Signed waveform-path IG\n"
    "(+ higher male logit; − lower male logit)",
    fontsize=9.5,
)

fig.subplots_adjust(
    left=0.055,
    right=0.905,
    bottom=0.045,
    top=0.955,
)

fig.savefig(
    MAIN_PNG,
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
)

fig.savefig(
    MAIN_PDF,
    bbox_inches="tight",
    facecolor="white",
)

fig.savefig(
    MAIN_SVG,
    bbox_inches="tight",
    facecolor="white",
)

plt.show()


# ==========================================================================================
# COMPACT 3 × 4 DIFFERENCE-ONLY FIGURE
# ==========================================================================================

diff_vmax = float(
    np.percentile(
        np.abs(
            diff_attr
        ),
        ATTR_COLOR_PERCENTILE,
    )
)

if (
    not np.isfinite(
        diff_vmax
    )
    or
    diff_vmax < 1e-12
):
    diff_vmax = ATTR_VMAX

fig2, axes = plt.subplots(
    4,
    3,
    figsize=(
        13.5,
        10.2,
    ),
    sharex=True,
)

for lead_idx, ax in enumerate(
    axes.ravel()
):
    plot_wave_attr(
        ax,
        time_ms,
        diff_ecg[
            lead_idx
        ],
        diff_attr[
            lead_idx
        ],
        diff_vmax,
        title=LEADS[
            lead_idx
        ],
    )

    if lead_idx >= 9:
        ax.set_xlabel(
            "Time from R peak (ms)"
        )


fig2.suptitle(
    "Sex-specific signed attribution difference — final VCG-Geometry ResNet",
    fontsize=16,
    fontweight="bold",
    y=0.995,
)

fig2.text(
    0.5,
    0.958,
    "Waveform = normalized male − female morphology; "
    "color = male − female signed male-logit attribution",
    ha="center",
    fontsize=9.5,
)

cax2 = fig2.add_axes(
    [
        0.92,
        0.18,
        0.017,
        0.62,
    ]
)

sm2 = plt.cm.ScalarMappable(
    cmap=CMAP,
    norm=plt.Normalize(
        vmin=-diff_vmax,
        vmax=diff_vmax,
    ),
)

sm2.set_array(
    []
)

cb2 = fig2.colorbar(
    sm2,
    cax=cax2,
)

cb2.set_label(
    "Male − female\nsigned IG",
    fontsize=9.5,
)

fig2.subplots_adjust(
    left=0.07,
    right=0.89,
    bottom=0.08,
    top=0.92,
    wspace=0.12,
    hspace=0.24,
)

fig2.savefig(
    DIFF_PNG,
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
)

fig2.savefig(
    DIFF_PDF,
    bbox_inches="tight",
    facecolor="white",
)

plt.show()


# ==========================================================================================
# CONSOLE SUMMARY
# ==========================================================================================

print(
    "\n"
    +
    "=" * 120
)

print(
    "FINAL CURRENT-RUN SEX-SPECIFIC ATTRIBUTION SUMMARY"
)

print(
    "=" * 120
)

print(
    "\nSample:"
)

print(
    f"Corrected IG N              : {n_ig}"
)

print(
    f"Successful beat alignment  : {len(ok_audit)}"
)

print(
    f"Female subjects used       : {n_female}"
)

print(
    f"Male subjects used         : {n_male}"
)

print(
    "\nOfficial corrected waveform-path lead ranking:"
)

display_cols = [
    "official_corrected_rank",
    "lead",
    "official_corrected_normalized_importance",
    "male_group_mean_signed_male_logit_attr",
    "female_group_mean_signed_male_logit_attr",
    "male_minus_female_mean_abs_attr",
    "peak_difference_time_ms",
]

print(
    summary_df[
        display_cols
    ].to_string(
        index=False
    )
)

print(
    "\nSaved:"
)

for path in [
    MAIN_PNG,
    MAIN_PDF,
    MAIN_SVG,
    DIFF_PNG,
    DIFF_PDF,
    SUMMARY_CSV,
    SUBJECT_AUDIT_CSV,
    ARRAYS_NPZ,
]:
    print(
        path
    )

print(
    "\nInterpretation note:"
)

print(
    "Colors represent corrected waveform-path representation-level IG for the male logit. "
    "They are model attribution/evidence, not causal electrophysiological effects."
)

print(
    "\nDONE"
)
