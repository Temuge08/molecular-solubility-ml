#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors


# ============================================================
# Paths
# ============================================================

RAW_FILE = Path("data/raw/delaney.csv")
REPORT_DIR = Path("reports")

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Column names
# ============================================================

ID_COL = "Compound ID"
SMILES_COL = "smiles"
TARGET_COL = "measured log solubility in mols per litre"


# ============================================================
# 1. Load raw dataset
# ============================================================

df_raw = pd.read_csv(RAW_FILE)

print("=" * 70)
print("RAW DATASET")
print("=" * 70)

print(f"Rows:    {len(df_raw)}")
print(f"Columns: {len(df_raw.columns)}")

print("\nOriginal columns:")
for col in df_raw.columns:
    print(f"  {col}")


# ============================================================
# 2. Select source columns only
# ============================================================

df = df_raw[
    [
        ID_COL,
        SMILES_COL,
        TARGET_COL,
    ]
].copy()

df.columns = [
    "compound_id",
    "smiles",
    "measured_logS",
]


# ============================================================
# 3. Check missing values
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

print(df.isna().sum())


# ============================================================
# 4. Check target datatype
# ============================================================

df["measured_logS"] = pd.to_numeric(
    df["measured_logS"],
    errors="coerce",
)

print("\nMissing/non-numeric logS values:")
print(df["measured_logS"].isna().sum())


# ============================================================
# 5. Parse SMILES using RDKit
# ============================================================

def parse_smiles(smiles):

    if pd.isna(smiles):
        return None

    try:
        return Chem.MolFromSmiles(str(smiles))
    except Exception:
        return None


df["mol"] = df["smiles"].apply(parse_smiles)

df["valid_smiles"] = df["mol"].notna()

print("\n" + "=" * 70)
print("SMILES VALIDITY")
print("=" * 70)

print("Valid:", df["valid_smiles"].sum())
print("Invalid:", (~df["valid_smiles"]).sum())


# ============================================================
# 6. Canonicalize SMILES
# ============================================================

def canonicalize(mol):

    if mol is None:
        return None

    try:
        return Chem.MolToSmiles(
            mol,
            canonical=True,
            isomericSmiles=True,
        )
    except Exception:
        return None


df["canonical_smiles"] = df["mol"].apply(canonicalize)


# ============================================================
# 7. Find duplicate structures
# ============================================================

duplicate_mask = (
    df["canonical_smiles"].notna()
    & df["canonical_smiles"].duplicated(keep=False)
)

duplicates = df.loc[
    duplicate_mask,
    [
        "compound_id",
        "smiles",
        "canonical_smiles",
        "measured_logS",
    ],
].copy()


print("\n" + "=" * 70)
print("DUPLICATE STRUCTURES")
print("=" * 70)

print("Rows belonging to duplicated structures:", len(duplicates))

print(
    "Unique duplicated structures:",
    duplicates["canonical_smiles"].nunique(),
)


# ============================================================
# 8. Examine disagreements between duplicate labels
# ============================================================

if len(duplicates) > 0:

    duplicate_summary = (
        duplicates
        .groupby("canonical_smiles")
        ["measured_logS"]
        .agg(
            count="count",
            mean="mean",
            std="std",
            minimum="min",
            maximum="max",
        )
    )

    duplicate_summary["range"] = (
        duplicate_summary["maximum"]
        - duplicate_summary["minimum"]
    )

    duplicate_summary = duplicate_summary.sort_values(
        "range",
        ascending=False,
    )

    print("\nLargest disagreements:")
    print(duplicate_summary.head(10))

    duplicate_summary.to_csv(
        REPORT_DIR / "duplicate_label_summary.csv"
    )


# ============================================================
# 9. Detect disconnected fragments / salts
# ============================================================

df["fragment_count"] = df["mol"].apply(
    lambda mol:
        len(Chem.GetMolFrags(mol))
        if mol is not None
        else np.nan
)

df["multi_fragment"] = df["fragment_count"] > 1


print("\n" + "=" * 70)
print("MULTI-FRAGMENT MOLECULES")
print("=" * 70)

print("Multi-fragment structures:", df["multi_fragment"].sum())


if df["multi_fragment"].any():

    print(
        df.loc[
            df["multi_fragment"],
            [
                "compound_id",
                "smiles",
                "fragment_count",
                "measured_logS",
            ],
        ].head(20)
    )


# ============================================================
# 10. Formal charge
# ============================================================

def get_formal_charge(mol):

    if mol is None:
        return np.nan

    return sum(
        atom.GetFormalCharge()
        for atom in mol.GetAtoms()
    )


df["formal_charge"] = df["mol"].apply(get_formal_charge)


print("\n" + "=" * 70)
print("FORMAL CHARGE")
print("=" * 70)

print(df["formal_charge"].value_counts(dropna=False).sort_index())


# ============================================================
# 11. Calculate basic QC descriptors
# ============================================================

def safe_mw(mol):

    if mol is None:
        return np.nan

    return Descriptors.MolWt(mol)


def safe_heavy_atoms(mol):

    if mol is None:
        return np.nan

    return Descriptors.HeavyAtomCount(mol)


df["molecular_weight"] = df["mol"].apply(safe_mw)
df["heavy_atom_count"] = df["mol"].apply(safe_heavy_atoms)


print("\n" + "=" * 70)
print("MOLECULAR SIZE")
print("=" * 70)

print(
    df[
        [
            "molecular_weight",
            "heavy_atom_count",
        ]
    ].describe()
)


# ============================================================
# 12. Examine target distribution
# ============================================================

print("\n" + "=" * 70)
print("SOLUBILITY TARGET")
print("=" * 70)

print(df["measured_logS"].describe())


print("\n10 lowest-solubility compounds:")

print(
    df[
        [
            "compound_id",
            "smiles",
            "measured_logS",
        ]
    ]
    .sort_values("measured_logS")
    .head(10)
)


print("\n10 highest-solubility compounds:")

print(
    df[
        [
            "compound_id",
            "smiles",
            "measured_logS",
        ]
    ]
    .sort_values(
        "measured_logS",
        ascending=False,
    )
    .head(10)
)


# ============================================================
# 13. Save audit result
# ============================================================

audit_output = df.drop(columns=["mol"])

audit_output.to_csv(
    REPORT_DIR / "esol_audit.csv",
    index=False,
)


if len(duplicates) > 0:

    duplicates.to_csv(
        REPORT_DIR / "duplicate_structures.csv",
        index=False,
    )


print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)

print("Saved:")
print("  reports/esol_audit.csv")

if len(duplicates) > 0:
    print("  reports/duplicate_structures.csv")
    print("  reports/duplicate_label_summary.csv")