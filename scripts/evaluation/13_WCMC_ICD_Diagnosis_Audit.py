#!/usr/bin/env python3
from pathlib import Path
import re
import duckdb
import pandas as pd
from datetime import datetime

DB = Path("/athena/madelab/scratch/iqh4001/Azure_Cradle_ECG/Data/ECG-Echo_MetaData/WCMC_MPOG_TRUE_MASTER_2018_2023.duckdb")
OUT = Path("/athena/madelab/scratch/iqh4001/ECG/ECG-gender/Results/WCMC_ICD_Diagnosis_Audit")
OUT.mkdir(parents=True, exist_ok=True)

def qi(x):
    return '"' + str(x).replace('"', '""') + '"'

def fq(schema_name, table_name):
    return qi(schema_name) + "." + qi(table_name)

def step(n, total, title):
    print()
    print("=" * 110)
    print(f"[STEP {n}/{total}] {title}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 110)

TOTAL_STEPS = 7

print("=" * 110)
print("WCMC DUCKDB ICD / DIAGNOSIS AUDIT")
print("=" * 110)
print("DB :", DB)
print("OUT:", OUT)

if not DB.exists():
    raise FileNotFoundError(DB)

step(1, TOTAL_STEPS, "Open DuckDB in read-only mode")
con = duckdb.connect(str(DB), read_only=True)
print("Connected successfully.")

step(2, TOTAL_STEPS, "Inventory all tables and columns")
schema = con.execute(
    """
    SELECT
        table_schema,
        table_name,
        column_name,
        data_type,
        ordinal_position
    FROM information_schema.columns
    WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
    ORDER BY table_schema, table_name, ordinal_position
    """
).df()

schema.to_csv(OUT / "01_ALL_COLUMNS.csv", index=False)
print(f"Tables found : {schema[['table_schema', 'table_name']].drop_duplicates().shape[0]}")
print(f"Columns found: {len(schema)}")
print("Saved: 01_ALL_COLUMNS.csv")

step(3, TOTAL_STEPS, "Search column names for diagnosis/comorbidity terms")

name_terms = [
    "icd", "diagn", "diagnosis", "dx", "comorb", "charlson", "elix",
    "phecode", "ccs", "ahrq", "coronary", "ischemic", "heart_failure",
    "heartfailure", "cerebro", "stroke", "diabet", "renal", "kidney",
    "hypertens", "atrial", "fibrillation", "condition", "problem"
]

def name_is_candidate(name):
    s = str(name).lower().replace(" ", "_").replace("-", "_")
    return any(term in s for term in name_terms)

name_hits = schema[
    schema["column_name"].astype(str).apply(name_is_candidate)
].copy()

name_hits.to_csv(
    OUT / "02_DIAGNOSIS_RELATED_COLUMN_NAME_HITS.csv",
    index=False
)
print(f"Diagnosis/comorbidity name hits: {len(name_hits)}")
print("Saved: 02_DIAGNOSIS_RELATED_COLUMN_NAME_HITS.csv")

step(4, TOTAL_STEPS, "Identify strict ICD/diagnosis/code-like field names")

strict_terms = [
    "icd", "diagn", "dx", "comorb", "charlson", "elix",
    "phecode", "ccs", "ahrq"
]

def name_is_strict(name):
    s = str(name).lower()
    return any(term in s for term in strict_terms)

strict_hits = schema[
    schema["column_name"].astype(str).apply(name_is_strict)
].copy()

strict_hits.to_csv(
    OUT / "03_STRICT_DIAGNOSIS_CODE_COLUMN_HITS.csv",
    index=False
)
print(f"Strict diagnosis/code field hits: {len(strict_hits)}")
print("Saved: 03_STRICT_DIAGNOSIS_CODE_COLUMN_HITS.csv")

step(5, TOTAL_STEPS, "Profile candidate fields: counts, distinct values, samples")

audit_rows = []

for _, r in name_hits.iterrows():
    sch = str(r["table_schema"])
    tab = str(r["table_name"])
    col = str(r["column_name"])
    table_ref = fq(sch, tab)
    col_ref = qi(col)

    row = {
        "table_schema": sch,
        "table_name": tab,
        "column_name": col,
        "data_type": str(r["data_type"]),
    }

    try:
        counts = con.execute(
            f"""
            SELECT
                COUNT(*) AS n_total,
                COUNT({col_ref}) AS n_nonmissing,
                COUNT(DISTINCT {col_ref}) AS n_distinct
            FROM {table_ref}
            """
        ).fetchone()

        row["n_total"] = counts[0]
        row["n_nonmissing"] = counts[1]
        row["n_distinct"] = counts[2]

        vals = con.execute(
            f"""
            SELECT DISTINCT CAST({col_ref} AS VARCHAR) AS value
            FROM {table_ref}
            WHERE {col_ref} IS NOT NULL
              AND TRIM(CAST({col_ref} AS VARCHAR)) <> ''
            LIMIT 30
            """
        ).df()

        row["sample_values"] = " | ".join(
            vals["value"].astype(str).tolist()
        )

    except Exception as exc:
        row["error"] = repr(exc)

    audit_rows.append(row)

candidate_audit = pd.DataFrame(audit_rows)
candidate_audit.to_csv(
    OUT / "04_DIAGNOSIS_COLUMN_VALUE_AUDIT.csv",
    index=False
)
print(f"Candidate fields profiled: {len(candidate_audit)}")
print("Saved: 04_DIAGNOSIS_COLUMN_VALUE_AUDIT.csv")

step(6, TOTAL_STEPS, "Scan text-column values for ICD-9/ICD-10-like codes")

text_type_pattern = re.compile("CHAR|VARCHAR|TEXT|STRING", re.IGNORECASE)

text_cols = schema[
    schema["data_type"].astype(str).str.contains(
        text_type_pattern, na=False
    )
].copy()

icd10_re = re.compile(
    "(?<![A-Z0-9])[A-TV-Z][0-9][0-9AB](?:\\.[A-Z0-9]{1,4})?(?![A-Z0-9])",
    re.IGNORECASE
)

icd9_re = re.compile(
    "(?<![A-Z0-9])(?:[0-9]{3}\\.[0-9]{1,2}|V[0-9]{2}(?:\\.[0-9]{1,2})?|E[0-9]{3}(?:\\.[0-9])?)(?![A-Z0-9])",
    re.IGNORECASE
)

value_hit_rows = []

for _, r in text_cols.iterrows():
    sch = str(r["table_schema"])
    tab = str(r["table_name"])
    col = str(r["column_name"])
    table_ref = fq(sch, tab)
    col_ref = qi(col)

    try:
        vals = con.execute(
            f"""
            SELECT DISTINCT CAST({col_ref} AS VARCHAR) AS value
            FROM {table_ref}
            WHERE {col_ref} IS NOT NULL
              AND TRIM(CAST({col_ref} AS VARCHAR)) <> ''
            LIMIT 2000
            """
        ).df()["value"].astype(str).tolist()
    except Exception:
        continue

    matched_values = []
    all_icd10 = set()
    all_icd9 = set()

    for value in vals:
        m10 = icd10_re.findall(value)
        m9 = icd9_re.findall(value)

        if m10 or m9:
            matched_values.append(value[:500])
            all_icd10.update(m10)
            all_icd9.update(m9)

        if len(matched_values) >= 25:
            break

    if matched_values:
        value_hit_rows.append(
            {
                "table_schema": sch,
                "table_name": tab,
                "column_name": col,
                "data_type": str(r["data_type"]),
                "n_values_sampled": len(vals),
                "n_matching_values_saved": len(matched_values),
                "sample_values": " || ".join(matched_values),
                "icd10_like_codes": " | ".join(sorted(all_icd10)),
                "icd9_like_codes": " | ".join(sorted(all_icd9)),
            }
        )

value_hits = pd.DataFrame(
    value_hit_rows,
    columns=[
        "table_schema",
        "table_name",
        "column_name",
        "data_type",
        "n_values_sampled",
        "n_matching_values_saved",
        "sample_values",
        "icd10_like_codes",
        "icd9_like_codes",
    ],
)

value_hits.to_csv(
    OUT / "05_ICD_LIKE_VALUE_HITS.csv",
    index=False
)
print(f"Text columns with ICD-like sampled values: {len(value_hits)}")
print("Saved: 05_ICD_LIKE_VALUE_HITS.csv")

step(7, TOTAL_STEPS, "Write final audit summary and display key results")

table_count = (
    schema[["table_schema", "table_name"]]
    .drop_duplicates()
    .shape[0]
)

summary_lines = [
    "WCMC DUCKDB ICD / DIAGNOSIS AUDIT",
    "=" * 90,
    f"Database: {DB}",
    "",
    "AUDIT SEQUENCE",
    "1. Open exact DuckDB in read-only mode",
    "2. Inventory every table and column",
    "3. Search column names for diagnosis/comorbidity terms",
    "4. Identify strict ICD/diagnosis/code-like field names",
    "5. Profile candidate fields (nonmissing, distinct, sample values)",
    "6. Scan text-column values for ICD-9/ICD-10-like patterns",
    "7. Summarize evidence and save outputs",
    "",
    "RESULT COUNTS",
    f"Tables inspected: {table_count}",
    f"Columns inspected: {len(schema)}",
    f"Diagnosis/comorbidity column-name hits: {len(name_hits)}",
    f"Strict ICD/diagnosis column-name hits: {len(strict_hits)}",
    f"Text columns with sampled ICD-like values: {len(value_hits)}",
    "",
    "NOTE:",
    "The value scan samples up to 2,000 distinct non-empty values per text column.",
    "Any promising diagnosis/code field should be exhaustively profiled next.",
]

summary_text = "\n".join(summary_lines)
(OUT / "06_ICD_AUDIT_SUMMARY.txt").write_text(summary_text)

print()
print(summary_text)

print("\nSTRICT DIAGNOSIS / CODE COLUMNS")
print("-" * 110)
if strict_hits.empty:
    print("NONE")
else:
    print(
        strict_hits[
            ["table_schema", "table_name", "column_name", "data_type"]
        ].to_string(index=False)
    )

print("\nICD-LIKE VALUE COLUMNS")
print("-" * 110)
if value_hits.empty:
    print("NONE")
else:
    print(
        value_hits[
            ["table_schema", "table_name", "column_name", "n_values_sampled"]
        ].to_string(index=False)
    )

print("\nSaved outputs:")
for p in sorted(OUT.iterdir()):
    print(" ", p)

con.close()
print("\n" + "=" * 110)
print("AUDIT COMPLETE")
print("=" * 110)
print("Output directory:", OUT)
print("Main files:")
print("  01_ALL_COLUMNS.csv")
print("  02_DIAGNOSIS_RELATED_COLUMN_NAME_HITS.csv")
print("  03_STRICT_DIAGNOSIS_CODE_COLUMN_HITS.csv")
print("  04_DIAGNOSIS_COLUMN_VALUE_AUDIT.csv")
print("  05_ICD_LIKE_VALUE_HITS.csv")
print("  06_ICD_AUDIT_SUMMARY.txt")
