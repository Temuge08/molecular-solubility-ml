#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem

# ============================================================
# Configuration
# ============================================================

RAW_FILE = Path("data/raw/delaney.csv")
PROCESSED_DIR = Path("data/processed")
REPORT_DIR = Path("reports")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
ID_COL = "Compound ID"
SMILES_COL = "smiles"
TARGET_COL = "measured log solubility in mols per litre"

# ------------------------------------------------------------
# Duplicate consistency threshold
#
# 0.30 log10 units corresponds to roughly a 2-fold
# difference in solubility.
#
# This is a project-specific heuristic, not a universal
# experimental standard.
# ------------------------------------------------------------

MAX_DUPLICATE_LOGS_RANGE = 0.30

# ============================================================
# Load source data
# ============================================================

df = pd.read_csv(RAW_FILE)

df = df[[ID_COL, SMILES_COL, TARGET_COL,]].copy()
df.columns = ["compound_id", "smiles", "measured_logS",]

print("=" * 70)
print("CURATION START")
print("=" * 70)

print(f"Raw rows: {len(df)}")

# ============================================================
# Parse molecules
# ============================================================

def parse_smiles(smiles):
    if pd.isna(smiles):
        return None

    try:
        return Chem.MolFromSmiles(str(smiles))
    except Exception:
        return None

df["mol"] = df["smiles"].apply(parse_smiles)

# ============================================================
# Defensive validation
#
# We already know the current dataset contains no invalid
# structures, but keeping this check makes the pipeline robust.
# ============================================================

invalid = df["mol"].isna()

if invalid.any():

    invalid_rows = df.loc[
        invalid,
        [
            "compound_id",
            "smiles",
            "measured_logS",
        ],
    ]

    invalid_rows.to_csv(
        REPORT_DIR / "invalid_molecules.csv",
        index=False,
    )

    print(f"Invalid molecules excluded: {invalid.sum()}")

    df = df.loc[~invalid].copy()


# ============================================================
# Canonical SMILES
# ============================================================

def canonicalize(mol):
    return Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=True,
    )


df["canonical_smiles"] = df["mol"].apply(canonicalize)

# ============================================================
# Prepare output containers
# ============================================================
curated_rows = []
curation_log = []
conflicting_rows = []


# ============================================================
# Process each unique represented structure
# ============================================================

for canonical_smiles, group in df.groupby(
    "canonical_smiles",
    sort=False,
):
    group = group.copy()
    n_records = len(group)
    labels = group["measured_logS"]

    label_min = labels.min()
    label_max = labels.max()

    label_range = label_max - label_min

    label_mean = labels.mean()

    # ddof=1 gives sample SD.
    # For a singleton there is no replicate SD.
    label_std = (
        labels.std(ddof=1)
        if n_records > 1
        else np.nan
    )

    # ========================================================
    # Case 1: unique structure
    # ========================================================
    if n_records == 1:
        row = group.iloc[0]
        curated_rows.append(
            {
                "compound_ids": row["compound_id"],
                "original_smiles": row["smiles"],
                "canonical_smiles": canonical_smiles,
                "measured_logS": row["measured_logS"],
                "replicate_n": 1,
                "replicate_std": np.nan,
                "replicate_range": 0.0,
                "curation_action": "unique_keep",
            }
        )
        curation_log.append(
            {
                "canonical_smiles": canonical_smiles,
                "n_source_records": 1,
                "label_min": label_min,
                "label_max": label_max,
                "label_range": 0.0,
                "action": "keep_unique",
            }
        )
        continue

    # ========================================================
    # Case 2: compatible duplicate measurements
    # ========================================================

    if label_range <= MAX_DUPLICATE_LOGS_RANGE:
        compound_ids = "|".join(
            group["compound_id"]
            .astype(str)
            .tolist()
        )

        source_smiles = "|".join(
            group["smiles"]
            .astype(str)
            .unique()
            .tolist()
        )

        curated_rows.append(
            {
                "compound_ids": compound_ids,
                "original_smiles": source_smiles,
                "canonical_smiles": canonical_smiles,

                # Aggregate replicate measurements
                "measured_logS": label_mean,
                "replicate_n": n_records,
                "replicate_std": label_std,
                "replicate_range": label_range,
                "curation_action": "duplicate_aggregated",
            }
        )

        curation_log.append(
            {
                "canonical_smiles": canonical_smiles,
                "n_source_records": n_records,
                "label_min": label_min,
                "label_max": label_max,
                "label_range": label_range,
                "action": "aggregate_duplicates",
            }
        )
        continue

    # ========================================================
    # Case 3: conflicting duplicate measurements
    # ========================================================

    for _, row in group.iterrows():
        conflicting_rows.append(
            {
                "compound_id": row["compound_id"],
                "smiles": row["smiles"],
                "canonical_smiles": canonical_smiles,
                "measured_logS": row["measured_logS"],
                "group_label_min": label_min,
                "group_label_max": label_max,
                "group_label_range": label_range,
            }
        )

    curation_log.append(
        {
            "canonical_smiles": canonical_smiles,
            "n_source_records": n_records,
            "label_min": label_min,
            "label_max": label_max,
            "label_range": label_range,
            "action": "exclude_conflicting_duplicate",
        }
    )

# ============================================================
# Create output dataframes
# ============================================================

curated = pd.DataFrame(curated_rows)
log_df = pd.DataFrame(curation_log)
conflicts = pd.DataFrame(conflicting_rows)

# ============================================================
# Sanity checks
# ============================================================

if curated["canonical_smiles"].duplicated().any():
    raise RuntimeError(
        "Duplicate canonical SMILES remain after curation."
    )

if curated["measured_logS"].isna().any():
    raise RuntimeError(
        "Missing target values remain after curation."
    )

# ============================================================
# Save outputs
# ============================================================

curated.to_csv(
    PROCESSED_DIR / "esol_curated.csv",
    index=False,
)

log_df.to_csv(
    REPORT_DIR / "curation_log.csv",
    index=False,
)

if len(conflicts) > 0:
    conflicts.to_csv(
        REPORT_DIR / "conflicting_duplicates.csv",
        index=False,
    )

# ============================================================
# Report
# ============================================================

print()
print("=" * 70)
print("CURATION SUMMARY")
print("=" * 70)

print(f"Raw records:                {len(df)}")
print(f"Final curated structures:   {len(curated)}")
print()
print("Curation actions:")
print(
    curated["curation_action"]
    .value_counts()
)
print()
print(
    "Conflicting source records excluded:",
    len(conflicts),
)
print()
print(
    "Unique canonical structures:",
    curated["canonical_smiles"].nunique(),
)
print("\nSaved:")
print("  data/processed/esol_curated.csv")
print("  reports/curation_log.csv")

if len(conflicts) > 0:
    print("  reports/conflicting_duplicates.csv")