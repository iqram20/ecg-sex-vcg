# ============================================================
# WCMC/NYP ECG-SEX — FROM-SCRATCH PIPELINE
# STEP 01: DISCOVER METADATA, BUILD PURE ECG-SEX COHORT,
#          LOCK ONE ECG PER PATIENT, AUDIT WAVEFORMS
#
# PERFORMANCE NOTE:
#   Optimized to lock one ECG per patient BEFORE filesystem path checks.
#   Recommended for Slurm CPU execution.
#
# PURPOSE
#   - DO NOT use the prior WCMC ECG->ECHO multitask outputs.
#   - Find the WCMC ECG metadata containing:
#         patient_key, ecg_key, ecg_sex,
#         raw/processed ECG paths, age/date if available.
#   - Restrict to known Female/Male labels only.
#   - Build a patient-independent validation cohort:
#         one ECG per patient, chosen deterministically.
#   - Resolve local raw and processed waveform paths.
#   - Audit waveform shape/orientation/quality.
#   - Save a locked cohort for ECG-sex model evaluation.
#
# IMPORTANT
#   This script DOES NOT train or evaluate any model.
#   It only constructs the correct WCMC ECG-sex cohort.
# ============================================================

import os
import json
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

SEED = 20260809
N_WAVEFORM_AUDIT = 50

AZURE_ROOT = Path(
    "/athena/madelab/scratch/iqh4001/Azure_Cradle_ECG"
)

DATA_ROOT = AZURE_ROOT / "Data"

RAW_DIR = (
    DATA_ROOT /
    "cradle-ecg/raw_ecg"
)

PROCESSED_DIR = (
    DATA_ROOT /
    "cradle-ecg/processed_ecg"
)

ORIGINAL_ECGSEX_RUN = Path(
    "/athena/madelab/scratch/iqh4001/ECG/ECG_Echo/Runs/"
    "Master_ECG_Sex_MIMIC_EchoNext/"
    "20260609_020154_MASTER_ECG_SEX_MIMIC_ECHONEXT"
)

OUT_ROOT = (
    ORIGINAL_ECGSEX_RUN /
    "wcmc_ecgsex_from_scratch"
)

OUT_DIR = OUT_ROOT / "01_cohort"

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# If you know the exact WCMC metadata file later, put it here.
# Leaving this as None causes automatic metadata discovery.
WCMC_METADATA_OVERRIDE = None

# Directories searched for WCMC metadata tables.
SEARCH_ROOTS = [
    DATA_ROOT / "ECG-Echo_MetaData",
]

# Do not recursively descend into huge waveform/image folders.
PRUNE_DIR_NAMES = {
    "raw_ecg",
    "processed_ecg",
    "processed_ecg_old",
    "raw_echo",
    "processed_echo",
    "waveforms",
    "waveform",
    "images",
    "figures",
    "plots",
    "runs",
    ".git",
}

# ============================================================
# COLUMN CANDIDATES
# ============================================================

ROLE_CANDIDATES = {
    "patient": [
        "patient_key",
        "patient_id",
        "subject_id",
        "person_id",
    ],
    "ecg_key": [
        "ecg_key",
        "ecg_id",
        "study_id",
    ],
    "sex": [
        "ecg_sex",
        "sex",
        "gender",
        "patient_sex",
    ],
    "age": [
        "ecg_age_at_ecg",
        "age_at_ecg",
        "age",
        "patient_age",
    ],
    "datetime": [
        "acquisition_datetime",
        "study_datetime",
        "ecg_datetime",
        "ecg_time",
        "acquisition_time",
    ],
    "raw_path": [
        "raw_ecg_blob_path",
        "raw_ecg_path",
        "raw_path",
    ],
    "processed_path": [
        "processed_ecg_blob_path",
        "processed_ecg_path",
        "processed_path",
    ],
    "generic_path": [
        "waveform_path",
        "ecg_path",
        "path",
        "file_path",
        "filename",
        "file_name",
    ],
}

# ============================================================
# UTILITIES
# ============================================================

def normalize_column_name(x):
    return str(x).strip().lower()


def normalize_sex(x):
    if pd.isna(x):
        return np.nan

    s = str(x).strip().lower()

    female_values = {
        "f",
        "female",
        "0",
        "0.0",
        "woman",
    }

    male_values = {
        "m",
        "male",
        "1",
        "1.0",
        "man",
    }

    if s in female_values:
        return 0

    if s in male_values:
        return 1

    return np.nan


def find_role_column(columns, role, required=False):
    lookup = {
        normalize_column_name(c): c
        for c in columns
    }

    for candidate in ROLE_CANDIDATES[role]:
        if candidate in lookup:
            return lookup[candidate]

    if required:
        raise RuntimeError(
            f"Could not resolve required '{role}' column.\n"
            f"Candidates: {ROLE_CANDIDATES[role]}\n"
            f"Available: {list(columns)}"
        )

    return None


def inspect_table_columns(path):
    """
    Read only schema/header whenever possible.
    """
    path = Path(path)

    try:
        if path.suffix.lower() == ".csv":
            tmp = pd.read_csv(
                path,
                nrows=5,
                low_memory=False,
            )

            return list(tmp.columns)

        if path.suffix.lower() in {
            ".parquet",
            ".pq",
        }:
            try:
                import pyarrow.parquet as pq

                schema = pq.read_schema(path)
                return list(schema.names)

            except Exception:
                tmp = pd.read_parquet(path)
                return list(tmp.columns)

    except Exception:
        return []

    return []


def discover_metadata_candidates():
    """
    Search metadata directories for CSV/Parquet tables and score them
    by whether they contain the WCMC ECG-sex fields.
    """
    rows = []
    seen = set()

    for root in SEARCH_ROOTS:
        root = Path(root)

        if not root.exists():
            continue

        for dirpath, dirnames, filenames in os.walk(root):

            # prune large folders
            dirnames[:] = [
                d for d in dirnames
                if d.lower() not in PRUNE_DIR_NAMES
            ]

            for fname in filenames:
                suffix = Path(fname).suffix.lower()

                if suffix not in {
                    ".csv",
                    ".parquet",
                    ".pq",
                }:
                    continue

                path = Path(dirpath) / fname

                try:
                    key = str(
                        path.resolve()
                    )
                except Exception:
                    key = str(path)

                if key in seen:
                    continue

                seen.add(key)

                cols = inspect_table_columns(
                    path
                )

                if not cols:
                    continue

                col_norm = {
                    normalize_column_name(c)
                    for c in cols
                }

                hits = {}

                for role, candidates in ROLE_CANDIDATES.items():
                    hits[role] = [
                        c for c in candidates
                        if c in col_norm
                    ]

                # ECG-sex metadata must have patient, ECG ID, sex.
                score = 0

                if hits["patient"]:
                    score += 5

                if hits["ecg_key"]:
                    score += 5

                if hits["sex"]:
                    score += 7

                if hits["raw_path"]:
                    score += 4

                if hits["processed_path"]:
                    score += 4

                if hits["generic_path"]:
                    score += 2

                if hits["age"]:
                    score += 2

                if hits["datetime"]:
                    score += 2

                # Filename hints.
                lname = path.name.lower()

                if "wcmc" in lname:
                    score += 2

                if "ecg" in lname:
                    score += 1

                if "echo" in lname:
                    score -= 0.25

                if score <= 0:
                    continue

                try:
                    size_mb = (
                        path.stat().st_size /
                        1024**2
                    )
                except Exception:
                    size_mb = np.nan

                rows.append(
                    {
                        "path": str(path),
                        "score": score,
                        "size_MB": size_mb,
                        "n_columns": len(cols),
                        "patient_hits": "|".join(hits["patient"]),
                        "ecg_key_hits": "|".join(hits["ecg_key"]),
                        "sex_hits": "|".join(hits["sex"]),
                        "age_hits": "|".join(hits["age"]),
                        "datetime_hits": "|".join(hits["datetime"]),
                        "raw_path_hits": "|".join(hits["raw_path"]),
                        "processed_path_hits": "|".join(hits["processed_path"]),
                        "generic_path_hits": "|".join(hits["generic_path"]),
                    }
                )

    out = pd.DataFrame(rows)

    if len(out):
        out = (
            out
            .sort_values(
                [
                    "score",
                    "size_MB",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
            .reset_index(drop=True)
        )

    return out


def load_table(path):
    path = Path(path)

    print("\nLoading metadata:")
    print(path)

    if path.suffix.lower() == ".csv":
        return pd.read_csv(
            path,
            low_memory=False,
        )

    if path.suffix.lower() in {
        ".parquet",
        ".pq",
    }:
        return pd.read_parquet(
            path
        )

    raise ValueError(
        f"Unsupported metadata extension: {path}"
    )


def select_metadata_file(candidate_df):
    if WCMC_METADATA_OVERRIDE is not None:
        path = Path(
            WCMC_METADATA_OVERRIDE
        )

        if not path.exists():
            raise FileNotFoundError(
                f"WCMC_METADATA_OVERRIDE does not exist:\n{path}"
            )

        return path

    if candidate_df.empty:
        raise RuntimeError(
            "No WCMC metadata candidate was found automatically."
        )

    # Require the core roles if possible.
    eligible = candidate_df.loc[
        candidate_df[
            "patient_hits"
        ].astype(str).str.len().gt(0)
        &
        candidate_df[
            "ecg_key_hits"
        ].astype(str).str.len().gt(0)
        &
        candidate_df[
            "sex_hits"
        ].astype(str).str.len().gt(0)
    ].copy()

    if eligible.empty:
        raise RuntimeError(
            "Candidates were found, but none contained "
            "patient + ECG key + sex fields."
        )

    selected = Path(
        eligible.iloc[0][
            "path"
        ]
    )

    return selected


def basename_from_metadata_value(x):
    if pd.isna(x):
        return None

    s = str(x).strip()

    if not s:
        return None

    # Normalize cloud/Windows separators.
    s = s.replace(
        "\\",
        "/"
    )

    return Path(s).name


def resolve_waveform_path(
    metadata_value,
    local_dir,
    ecg_key=None,
):
    """
    Resolve a WCMC waveform locally.

    Priority:
      1) metadata path if already absolute and exists
      2) local_dir / basename(metadata path)
      3) local_dir / <ecg_key>.npy
    """
    if pd.notna(metadata_value):
        s = str(
            metadata_value
        ).strip()

        if s:
            direct = Path(s)

            if direct.exists():
                return str(
                    direct
                )

            base = basename_from_metadata_value(
                s
            )

            if base:
                p = Path(
                    local_dir
                ) / base

                if p.exists():
                    return str(p)

    if ecg_key is not None and pd.notna(ecg_key):
        key = str(
            ecg_key
        ).strip()

        if key:
            p = (
                Path(local_dir) /
                f"{key}.npy"
            )

            if p.exists():
                return str(p)

    return None


def orientation_from_shape(shape):
    if len(shape) != 2:
        return "not_2d"

    a, b = shape

    if a == 12 and b != 12:
        return "lead_x_time"

    if b == 12 and a != 12:
        return "time_x_lead"

    if a == 12 and b == 12:
        return "ambiguous_12x12"

    return "no_12lead_axis"


def inspect_waveform(path):
    row = {
        "path": path,
        "exists": False,
        "load_ok": False,
        "shape": None,
        "orientation": None,
        "dtype": None,
        "finite_pct": np.nan,
        "zero_pct": np.nan,
        "min": np.nan,
        "max": np.nan,
        "mean": np.nan,
        "sd": np.nan,
    }

    if path is None:
        return row

    p = Path(path)

    row["exists"] = p.exists()

    if not p.exists():
        return row

    try:
        arr = np.load(
            p,
            allow_pickle=False
        )

        arr = np.asarray(
            arr
        )

        row["load_ok"] = True
        row["shape"] = str(
            tuple(arr.shape)
        )
        row["orientation"] = orientation_from_shape(
            arr.shape
        )
        row["dtype"] = str(
            arr.dtype
        )

        if arr.size:
            finite = np.isfinite(
                arr
            )

            row["finite_pct"] = float(
                100.0 *
                finite.mean()
            )

            row["zero_pct"] = float(
                100.0 *
                (arr == 0).mean()
            )

            vals = arr[
                finite
            ]

            if vals.size:
                row["min"] = float(
                    np.min(vals)
                )
                row["max"] = float(
                    np.max(vals)
                )
                row["mean"] = float(
                    np.mean(vals)
                )
                row["sd"] = float(
                    np.std(vals)
                )

    except Exception as exc:
        row["error"] = repr(
            exc
        )

    return row


# ============================================================
# START
# ============================================================

print("=" * 110)
print("WCMC/NYP ECG-SEX — FROM-SCRATCH COHORT BUILD")
print("=" * 110)

print("\nRaw ECG directory:")
print(RAW_DIR)

print("\nProcessed ECG directory:")
print(PROCESSED_DIR)

if not RAW_DIR.exists():
    print(
        "WARNING: raw ECG directory does not exist."
    )

if not PROCESSED_DIR.exists():
    print(
        "WARNING: processed ECG directory does not exist."
    )

# ============================================================
# 1. DISCOVER METADATA
# ============================================================

print("\n" + "=" * 110)
print("1. DISCOVER WCMC ECG METADATA")
print("=" * 110)

candidates = discover_metadata_candidates()

candidate_file = (
    OUT_DIR /
    "WCMC_ECGSex_Metadata_Candidates.csv"
)

candidates.to_csv(
    candidate_file,
    index=False
)

print(
    "\nTop candidates:"
)

if len(candidates):
    print(
        candidates.head(
            20
        ).to_string(
            index=False
        )
    )
else:
    print(
        "No candidates found."
    )

metadata_path = select_metadata_file(
    candidates
)

print(
    "\nSELECTED WCMC METADATA:"
)

print(
    metadata_path
)

# ============================================================
# 2. LOAD + RESOLVE COLUMNS
# ============================================================

print("\n" + "=" * 110)
print("2. LOAD METADATA AND RESOLVE ECG-SEX COLUMNS")
print("=" * 110)

df = load_table(
    metadata_path
)

print(
    "Metadata shape:",
    df.shape
)

patient_col = find_role_column(
    df.columns,
    "patient",
    required=True
)

ecg_col = find_role_column(
    df.columns,
    "ecg_key",
    required=True
)

sex_col = find_role_column(
    df.columns,
    "sex",
    required=True
)

age_col = find_role_column(
    df.columns,
    "age",
    required=False
)

datetime_col = find_role_column(
    df.columns,
    "datetime",
    required=False
)

raw_path_col = find_role_column(
    df.columns,
    "raw_path",
    required=False
)

processed_path_col = find_role_column(
    df.columns,
    "processed_path",
    required=False
)

generic_path_col = find_role_column(
    df.columns,
    "generic_path",
    required=False
)

resolved_columns = {
    "patient_col":
        patient_col,
    "ecg_key_col":
        ecg_col,
    "sex_col":
        sex_col,
    "age_col":
        age_col,
    "datetime_col":
        datetime_col,
    "raw_path_col":
        raw_path_col,
    "processed_path_col":
        processed_path_col,
    "generic_path_col":
        generic_path_col,
}

print(
    json.dumps(
        resolved_columns,
        indent=2
    )
)

with open(
    OUT_DIR /
    "WCMC_ECGSex_Resolved_Columns.json",
    "w",
) as f:
    json.dump(
        {
            "metadata_file":
                str(metadata_path),
            **resolved_columns,
        },
        f,
        indent=2,
    )

# ============================================================
# 3. PURE ECG-SEX CLEANING
# ============================================================

print("\n" + "=" * 110)
print("3. CLEAN ECG-SEX LABELS")
print("=" * 110)

work = pd.DataFrame(
    {
        "patient_key":
            df[
                patient_col
            ].astype(str),
        "ecg_key":
            df[
                ecg_col
            ].astype(str),
        "sex_original":
            df[
                sex_col
            ],
    }
)

work[
    "sex_binary"
] = (
    work[
        "sex_original"
    ]
    .apply(
        normalize_sex
    )
)

work[
    "sex"
] = work[
    "sex_binary"
].map(
    {
        0.0: "Female",
        1.0: "Male",
    }
)

if age_col is not None:
    work[
        "age_at_ecg"
    ] = pd.to_numeric(
        df[
            age_col
        ],
        errors="coerce",
    )
else:
    work[
        "age_at_ecg"
    ] = np.nan

if datetime_col is not None:
    work[
        "ecg_datetime"
    ] = pd.to_datetime(
        df[
            datetime_col
        ],
        errors="coerce",
    )
else:
    work[
        "ecg_datetime"
    ] = pd.NaT

if raw_path_col is not None:
    work[
        "raw_path_metadata"
    ] = df[
        raw_path_col
    ]
else:
    work[
        "raw_path_metadata"
    ] = None

if processed_path_col is not None:
    work[
        "processed_path_metadata"
    ] = df[
        processed_path_col
    ]
else:
    work[
        "processed_path_metadata"
    ] = None

if generic_path_col is not None:
    work[
        "generic_path_metadata"
    ] = df[
        generic_path_col
    ]
else:
    work[
        "generic_path_metadata"
    ] = None

print(
    "\nOriginal sex values:"
)

print(
    work[
        "sex_original"
    ]
    .astype(str)
    .value_counts(
        dropna=False
    )
    .head(20)
)

usable = work.dropna(
    subset=[
        "sex_binary",
        "patient_key",
        "ecg_key",
    ]
).copy()

usable[
    "sex_binary"
] = usable[
    "sex_binary"
].astype(int)

# Remove string representations of missing IDs.
for c in [
    "patient_key",
    "ecg_key",
]:
    usable = usable.loc[
        ~usable[c]
        .str.lower()
        .isin(
            [
                "",
                "nan",
                "none",
                "<na>",
            ]
        )
    ].copy()

print(
    f"\nRows with known sex: "
    f"{len(usable):,}"
)

print(
    f"Unique patients: "
    f"{usable['patient_key'].nunique():,}"
)

print(
    f"Unique ECGs: "
    f"{usable['ecg_key'].nunique():,}"
)

print(
    "\nSex counts — ECG rows:"
)

print(
    usable[
        "sex"
    ].value_counts()
)

# ============================================================
# 4. LOCK ONE ECG PER PATIENT FIRST
#
# IMPORTANT PERFORMANCE OPTIMIZATION:
#   Do NOT resolve waveform paths for all ~933k ECG rows.
#   First reduce to one ECG per patient (~110k), then check paths.
#
# Deterministic rule:
#   - earliest dated ECG if date exists
#   - otherwise lexical ECG key
# ============================================================

print("\n" + "=" * 110)
print("4. LOCK ONE ECG PER PATIENT BEFORE PATH RESOLUTION")
print("=" * 110)

usable[
    "_date_missing"
] = usable[
    "ecg_datetime"
].isna().astype(int)

usable = usable.sort_values(
    [
        "patient_key",
        "_date_missing",
        "ecg_datetime",
        "ecg_key",
    ],
    ascending=[
        True,
        True,
        True,
        True,
    ],
)

locked_pre = (
    usable
    .groupby(
        "patient_key",
        as_index=False,
        sort=False,
    )
    .first()
)

locked_pre = locked_pre.drop(
    columns=[
        "_date_missing",
    ],
    errors="ignore",
)

print(
    f"Rows before locking: {len(usable):,}"
)

print(
    f"Locked ECGs/patients before path check: {len(locked_pre):,}"
)

print(
    "\nSex counts before waveform availability filtering:"
)

print(
    locked_pre[
        "sex"
    ].value_counts()
)

# ============================================================
# 5. RESOLVE LOCAL WAVEFORM PATHS ONLY FOR LOCKED ECGs
# ============================================================

print("\n" + "=" * 110)
print("5. RESOLVE WAVEFORM PATHS FOR LOCKED ECGs ONLY")
print("=" * 110)

raw_meta = locked_pre[
    "raw_path_metadata"
].copy()

proc_meta = locked_pre[
    "processed_path_metadata"
].copy()

if generic_path_col is not None:
    raw_meta = raw_meta.where(
        raw_meta.notna(),
        locked_pre[
            "generic_path_metadata"
        ],
    )

    proc_meta = proc_meta.where(
        proc_meta.notna(),
        locked_pre[
            "generic_path_metadata"
        ],
    )

raw_local = []
processed_local = []

n_locked_pre = len(locked_pre)

for j, (i, row) in enumerate(
    locked_pre.iterrows(),
    start=1,
):
    raw_local.append(
        resolve_waveform_path(
            raw_meta.loc[i],
            RAW_DIR,
            row[
                "ecg_key"
            ],
        )
    )

    processed_local.append(
        resolve_waveform_path(
            proc_meta.loc[i],
            PROCESSED_DIR,
            row[
                "ecg_key"
            ],
        )
    )

    if (
        j % 10000 == 0
        or j == n_locked_pre
    ):
        print(
            f"Resolved {j:,}/{n_locked_pre:,} locked ECGs"
        )

locked_pre[
    "raw_local_path"
] = raw_local

locked_pre[
    "processed_local_path"
] = processed_local

locked_pre[
    "raw_available"
] = locked_pre[
    "raw_local_path"
].notna()

locked_pre[
    "processed_available"
] = locked_pre[
    "processed_local_path"
].notna()

locked_pre[
    "any_waveform_available"
] = (
    locked_pre[
        "raw_available"
    ]
    |
    locked_pre[
        "processed_available"
    ]
)

print(
    "Raw available:",
    f"{locked_pre['raw_available'].sum():,}",
    f"({100 * locked_pre['raw_available'].mean():.2f}%)",
)

print(
    "Processed available:",
    f"{locked_pre['processed_available'].sum():,}",
    f"({100 * locked_pre['processed_available'].mean():.2f}%)",
)

print(
    "Any waveform available:",
    f"{locked_pre['any_waveform_available'].sum():,}",
    f"({100 * locked_pre['any_waveform_available'].mean():.2f}%)",
)

locked = locked_pre.loc[
    locked_pre[
        "any_waveform_available"
    ]
].copy()

locked[
    "cohort"
] = (
    "WCMC_NYP_ECGSex_External"
)

print(
    f"\nFinal locked ECGs/patients with waveform: "
    f"{len(locked):,}"
)

print(
    "\nSex counts:"
)

print(
    locked[
        "sex"
    ].value_counts()
)

print(
    "\nSex percentages:"
)

print(
    (
        100 *
        locked[
            "sex"
        ].value_counts(
            normalize=True
        )
    ).round(2)
)

if locked[
    "age_at_ecg"
].notna().any():
    print(
        "\nAge summary:"
    )

    print(
        locked[
            "age_at_ecg"
        ].describe()
    )

if locked[
    "patient_key"
].duplicated().any():
    raise RuntimeError(
        "Patient duplication remains in locked cohort."
    )

# Keep only the patient-level cohort for downstream model evaluation.
usable = locked.copy()

# ============================================================
# 6. SAVE FULL USABLE + LOCKED COHORT
# ============================================================

all_usable_file = (
    OUT_DIR /
    "WCMC_ECGSex_All_KnownSex_WithWaveform.csv"
)

locked_file = (
    OUT_DIR /
    "WCMC_ECGSex_Locked_OneECGPerPatient.csv"
)

usable.to_csv(
    all_usable_file,
    index=False,
)

locked.to_csv(
    locked_file,
    index=False,
)

print(
    "\nSaved all usable ECGs:"
)

print(
    all_usable_file
)

print(
    "\nSaved locked patient-level cohort:"
)

print(
    locked_file
)

# ============================================================
# 7. WAVEFORM AUDIT
# ============================================================

print("\n" + "=" * 110)
print("7. WAVEFORM AUDIT")
print("=" * 110)

rng = np.random.default_rng(
    SEED
)

n_sample = min(
    N_WAVEFORM_AUDIT,
    len(locked),
)

sample_positions = rng.choice(
    len(locked),
    size=n_sample,
    replace=False,
)

audit_rows = []

for j, pos in enumerate(
    sample_positions,
    start=1,
):
    row = locked.iloc[
        int(pos)
    ]

    for source_name, path_col in [
        (
            "raw",
            "raw_local_path",
        ),
        (
            "processed",
            "processed_local_path",
        ),
    ]:
        wf = inspect_waveform(
            row[
                path_col
            ]
        )

        audit_rows.append(
            {
                "patient_key":
                    row[
                        "patient_key"
                    ],
                "ecg_key":
                    row[
                        "ecg_key"
                    ],
                "sex":
                    row[
                        "sex"
                    ],
                "sex_binary":
                    row[
                        "sex_binary"
                    ],
                "age_at_ecg":
                    row[
                        "age_at_ecg"
                    ],
                "waveform_source":
                    source_name,
                **wf,
            }
        )

    if (
        j % 50
        == 0
        or j == n_sample
    ):
        print(
            f"Audited "
            f"{j:,}/{n_sample:,} "
            "locked ECGs"
        )

audit_df = pd.DataFrame(
    audit_rows
)

audit_file = (
    OUT_DIR /
    "WCMC_ECGSex_Waveform_Audit.csv"
)

audit_df.to_csv(
    audit_file,
    index=False,
)

print(
    "\nWaveform orientation counts:"
)

if "orientation" in audit_df.columns:
    print(
        audit_df.groupby(
            "waveform_source"
        )[
            "orientation"
        ].value_counts(
            dropna=False
        )
    )

print(
    "\nLoad success:"
)

print(
    audit_df.groupby(
        "waveform_source"
    )[
        "load_ok"
    ].agg(
        [
            "sum",
            "count",
            "mean",
        ]
    )
)

print(
    "\nWaveform SD summary:"
)

print(
    audit_df.groupby(
        "waveform_source"
    )[
        "sd"
    ].describe()
)

# ============================================================
# 8. COHORT SUMMARY
# ============================================================

summary = {
    "metadata_file":
        str(metadata_path),
    "source_metadata_rows":
        int(len(df)),
    "known_sex_rows":
        int(
            len(
                work.dropna(
                    subset=[
                        "sex_binary"
                    ]
                )
            )
        ),
    "known_sex_unique_patients":
        int(
            usable[
                "patient_key"
            ].nunique()
        ),
    "locked_N":
        int(
            len(locked)
        ),
    "locked_female_N":
        int(
            (
                locked[
                    "sex_binary"
                ] == 0
            ).sum()
        ),
    "locked_male_N":
        int(
            (
                locked[
                    "sex_binary"
                ] == 1
            ).sum()
        ),
    "raw_available_locked_N":
        int(
            locked[
                "raw_available"
            ].sum()
        ),
    "processed_available_locked_N":
        int(
            locked[
                "processed_available"
            ].sum()
        ),
    "selection_rule":
        (
            "one ECG per patient; earliest available ECG "
            "by ECG datetime, otherwise lexical ECG key"
        ),
    "raw_ecg_dir":
        str(
            RAW_DIR
        ),
    "processed_ecg_dir":
        str(
            PROCESSED_DIR
        ),
    "locked_cohort_file":
        str(
            locked_file
        ),
    "waveform_audit_file":
        str(
            audit_file
        ),
}

if locked[
    "age_at_ecg"
].notna().any():
    summary[
        "mean_age"
    ] = float(
        locked[
            "age_at_ecg"
        ].mean()
    )

    summary[
        "median_age"
    ] = float(
        locked[
            "age_at_ecg"
        ].median()
    )

with open(
    OUT_DIR /
    "WCMC_ECGSex_Cohort_Summary.json",
    "w",
) as f:
    json.dump(
        summary,
        f,
        indent=2,
    )

summary_df = pd.DataFrame(
    [
        summary
    ]
)

summary_df.to_csv(
    OUT_DIR /
    "WCMC_ECGSex_Cohort_Summary.csv",
    index=False,
)

print("\n" + "=" * 110)
print("WCMC ECG-SEX COHORT BUILD COMPLETE")
print("=" * 110)

print(
    json.dumps(
        summary,
        indent=2,
    )
)

print("\nNEXT INPUT FOR MODEL EVALUATION:")
print(locked_file)
print("=" * 110)
