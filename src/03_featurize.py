#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator

# ============================================================
# Configuration
# ============================================================

INPUT_FILE = Path("data/processed/esol_curated.csv")
OUTPUT_DIR = Path("data/processed")
REPORT_DIR = Path("reports")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Morgan fingerprint parameters
MORGAN_RADIUS = 2
MORGAN_BITS = 2048

# ============================================================
# Load curated dataset
# ============================================================

df = pd.read_csv(INPUT_FILE)
print("=" * 70)
print("MOLECULAR FEATURIZATION")
print("=" * 70)

print(f"Input molecules: {len(df)}")


# ============================================================
# Parse canonical SMILES
# ============================================================

def parse_smiles(smiles):
    try:
        return Chem.MolFromSmiles(str(smiles))
    except Exception:
        return None


df["mol"] = df["canonical_smiles"].apply(parse_smiles)

# Curated data should already be valid.
# This is a defensive check.

invalid = df["mol"].isna()

if invalid.any():
    raise RuntimeError(
        f"{invalid.sum()} invalid molecules found in curated dataset."
    )

# ============================================================
# Physicochemical descriptors
# ============================================================

def calculate_descriptors(mol):
    return {
        "desc_mol_weight": Descriptors.MolWt(mol),
        "desc_logp": Descriptors.MolLogP(mol),
        "desc_tpsa": rdMolDescriptors.CalcTPSA(mol),
        "desc_hbd": rdMolDescriptors.CalcNumHBD(mol),
        "desc_hba": rdMolDescriptors.CalcNumHBA(mol),
        "desc_rotatable_bonds":
            rdMolDescriptors.CalcNumRotatableBonds(mol),
        "desc_ring_count":
            rdMolDescriptors.CalcNumRings(mol),
        "desc_aromatic_ring_count":
            rdMolDescriptors.CalcNumAromaticRings(mol),
        "desc_heavy_atom_count":
            Descriptors.HeavyAtomCount(mol),
        "desc_fraction_csp3":
            rdMolDescriptors.CalcFractionCSP3(mol),
    }

descriptor_records = [
    calculate_descriptors(mol)
    for mol in df["mol"]
]

descriptor_df = pd.DataFrame(descriptor_records)

# ============================================================
# Morgan fingerprint generator
# ============================================================

morgan_generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=MORGAN_RADIUS,
    fpSize=MORGAN_BITS,
    includeChirality=True,
)
def calculate_morgan(mol):
    fingerprint = morgan_generator.GetFingerprint(mol)
    array = np.zeros(MORGAN_BITS, dtype=np.uint8,)
    DataStructs.ConvertToNumpyArray(
        fingerprint,
        array,
    )
    return array

fingerprint_matrix = np.vstack(
    [
        calculate_morgan(mol)
        for mol in df["mol"]
    ]
)

fingerprint_columns = [
    f"fp_{i:04d}"
    for i in range(MORGAN_BITS)
]

fingerprint_df = pd.DataFrame(
    fingerprint_matrix,
    columns=fingerprint_columns,
)


# ============================================================
# Metadata / labels
# ============================================================

metadata_columns = [
    "compound_ids",
    "canonical_smiles",
    "measured_logS",
    "replicate_n",
    "replicate_std",
    "replicate_range",
    "curation_action",
]

metadata = df[metadata_columns].reset_index(drop=True)

# ============================================================
# Combine representations
# ============================================================

descriptor_output = pd.concat(
    [
        metadata,
        descriptor_df,
    ],
    axis=1,
)

fingerprint_output = pd.concat(
    [
        metadata,
        fingerprint_df,
    ],
    axis=1,
)

combined_output = pd.concat(
    [
        metadata,
        descriptor_df,
        fingerprint_df,
    ],
    axis=1,
)

# ============================================================
# Feature quality control
# ============================================================

descriptor_columns = list(descriptor_df.columns)

print()
print("=" * 70)
print("DESCRIPTOR SUMMARY")
print("=" * 70)

print(descriptor_df.describe().T)

# ------------------------------------------------------------
# Missing descriptors
# ------------------------------------------------------------

missing_descriptor_values = (descriptor_df.isna().sum())
print()
print("Missing descriptor values:")
print(missing_descriptor_values)

# ------------------------------------------------------------
# Constant descriptors
# ------------------------------------------------------------

descriptor_unique_counts = (descriptor_df.nunique())
constant_descriptors = (descriptor_unique_counts[descriptor_unique_counts <= 1])

print()
print("Constant descriptor features:")

if len(constant_descriptors) == 0:
    print("None")
else:
    print(constant_descriptors)

# ------------------------------------------------------------
# Fingerprint density
# ------------------------------------------------------------
bits_per_molecule = fingerprint_matrix.sum(axis=1)

print()
print("=" * 70)
print("MORGAN FINGERPRINT SUMMARY")
print("=" * 70)

print(f"Radius: {MORGAN_RADIUS}")
print(f"Bits:   {MORGAN_BITS}")

print(
    f"Average active bits/molecule: "
    f"{bits_per_molecule.mean():.2f}"
)

print(
    f"Minimum active bits: "
    f"{bits_per_molecule.min()}"
)

print(
    f"Maximum active bits: "
    f"{bits_per_molecule.max()}"
)


# ------------------------------------------------------------
# Fingerprint bits that are never active
# ------------------------------------------------------------

bit_frequency = fingerprint_matrix.sum(axis=0)
never_active = np.sum(bit_frequency == 0)

print(
    f"Fingerprint bits never active: "
    f"{never_active}/{MORGAN_BITS}"
)

# ============================================================
# Save feature tables
# ============================================================

descriptor_output.to_csv(
    OUTPUT_DIR / "esol_descriptors.csv",
    index=False,
)

fingerprint_output.to_csv(
    OUTPUT_DIR / "esol_morgan.csv",
    index=False,
)

combined_output.to_csv(
    OUTPUT_DIR / "esol_combined_features.csv",
    index=False,
)


# ============================================================
# Save feature summaries
# ============================================================

descriptor_summary = descriptor_df.describe().T

descriptor_summary.to_csv(
    REPORT_DIR / "descriptor_summary.csv"
)


fingerprint_summary = pd.DataFrame(
    {
        "fingerprint_bit": fingerprint_columns,
        "active_molecule_count": bit_frequency,
        "active_fraction":
            bit_frequency / len(df),
    }
)

fingerprint_summary.to_csv(
    REPORT_DIR / "morgan_bit_summary.csv",
    index=False,
)


# ============================================================
# Final report
# ============================================================

print()
print("=" * 70)
print("FEATURIZATION COMPLETE")
print("=" * 70)
print(f"Molecules:             {len(df)}")
print(f"Descriptor features:   {len(descriptor_columns)}")
print(f"Morgan features:       {MORGAN_BITS}")

print(
    f"Combined features:     "
    f"{len(descriptor_columns) + MORGAN_BITS}"
)

print()
print("Saved:")
print("  data/processed/esol_descriptors.csv")
print("  data/processed/esol_morgan.csv")
print("  data/processed/esol_combined_features.csv")
print("  reports/descriptor_summary.csv")
print("  reports/morgan_bit_summary.csv")