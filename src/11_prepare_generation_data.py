#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
from rdkit import Chem

# ============================================================
# Configuration
# ============================================================

INPUT_FILE = Path(
    "data/generation/raw/moses_train.csv"
)

OUTPUT_FILE = Path(
    "data/generation/processed/moses_100k.csv"
)

REPORT_FILE = Path(
    "reports/generation/generation_data_summary.csv"
)

N_MOLECULES = 100_000
RANDOM_STATE = 42

MIN_SMILES_LENGTH = 5
MAX_SMILES_LENGTH = 120


# ============================================================
# Load
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("GENERATION DATA PREPARATION")
print("=" * 70)

print(f"Raw rows: {len(df)}")
print(f"Columns: {list(df.columns)}")


# ============================================================
# Locate SMILES column
# ============================================================

candidate_columns = [
    c
    for c in df.columns
    if c.lower() == "smiles"
]

if len(candidate_columns) != 1:
    raise RuntimeError(
        f"Could not uniquely identify SMILES column: "
        f"{candidate_columns}"
    )

SMILES_COL = candidate_columns[0]


work = df[
    [SMILES_COL]
].copy()

work.columns = [
    "smiles"
]


# ============================================================
# Missing values
# ============================================================

n_initial = len(work)

work = work.dropna(
    subset=["smiles"]
).copy()


# ============================================================
# Parse + canonicalize
# ============================================================

def canonicalize(smiles):

    try:
        mol = Chem.MolFromSmiles(
            str(smiles)
        )

        if mol is None:
            return None

        return Chem.MolToSmiles(
            mol,
            canonical=True,
            isomericSmiles=True,
        )

    except Exception:
        return None


work[
    "canonical_smiles"
] = work["smiles"].apply(
    canonicalize
)


n_invalid = (
    work["canonical_smiles"]
    .isna()
    .sum()
)


work = work.dropna(
    subset=[
        "canonical_smiles"
    ]
).copy()


# ============================================================
# Remove duplicates
# ============================================================

n_before_duplicates = len(work)

work = work.drop_duplicates(
    subset=[
        "canonical_smiles"
    ]
).copy()

n_duplicates = (
    n_before_duplicates
    - len(work)
)


# ============================================================
# SMILES length
#
# For the first VAE we restrict extremely long strings.
# This simplifies sequence modeling and batching.
# ============================================================

work[
    "smiles_length"
] = (
    work["canonical_smiles"]
    .str.len()
)


length_mask = (
    work["smiles_length"]
    .between(
        MIN_SMILES_LENGTH,
        MAX_SMILES_LENGTH,
    )
)


n_length_removed = (
    (~length_mask).sum()
)


work = work.loc[
    length_mask
].copy()


# ============================================================
# Sample fixed development set
# ============================================================

if len(work) < N_MOLECULES:

    raise RuntimeError(
        f"Only {len(work)} molecules available, "
        f"cannot sample {N_MOLECULES}."
    )


sample = work.sample(
    n=N_MOLECULES,
    random_state=RANDOM_STATE,
).copy()


sample = sample.reset_index(
    drop=True
)


# ============================================================
# Save
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


sample[
    [
        "canonical_smiles",
        "smiles_length",
    ]
].to_csv(
    OUTPUT_FILE,
    index=False,
)


summary = pd.DataFrame(
    [
        {
            "raw_rows":
                n_initial,

            "invalid_smiles":
                n_invalid,

            "duplicates_removed":
                n_duplicates,

            "length_filtered":
                n_length_removed,

            "final_sample_size":
                len(sample),

            "minimum_length":
                sample[
                    "smiles_length"
                ].min(),

            "maximum_length":
                sample[
                    "smiles_length"
                ].max(),

            "mean_length":
                sample[
                    "smiles_length"
                ].mean(),
        }
    ]
)


summary.to_csv(
    REPORT_FILE,
    index=False,
)


print()
print("=" * 70)
print("FINAL GENERATION DATASET")
print("=" * 70)

print(f"Molecules: {len(sample)}")

print(
    "SMILES length:"
)

print(
    sample["smiles_length"]
    .describe()
)

print()
print(
    f"Saved: {OUTPUT_FILE}"
)