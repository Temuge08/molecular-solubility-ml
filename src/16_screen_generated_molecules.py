#!/usr/bin/env python3

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors, QED, rdMolDescriptors


# ============================================================
# Paths
# ============================================================

GENERATED_FILE = Path("data/generation/processed/vae_generated_novel.csv")
MODEL_FILE = Path("models/final_esol_descriptor_rf.joblib")
ESOL_FEATURE_FILE = Path("data/processed/esol_descriptors.csv")
ESOL_MANIFEST_FILE = Path("data/processed/esol_cluster_split_manifest.csv")
OUTPUT_FILE = Path("data/generation/processed/vae_generated_screened.csv")
SUMMARY_FILE = Path("reports/generation/generated_screening_summary.csv")
TOP_FILE = Path("reports/generation/top_generated_candidates.csv")

# ============================================================
# Descriptor order
#
# MUST match the features used to train the final RF.
# ============================================================

DESCRIPTOR_COLUMNS = [
    "desc_mol_weight",
    "desc_logp",
    "desc_tpsa",
    "desc_hbd",
    "desc_hba",
    "desc_rotatable_bonds",
    "desc_ring_count",
    "desc_aromatic_ring_count",
    "desc_heavy_atom_count",
    "desc_fraction_csp3",
]


# ============================================================
# Load generated molecules
# ============================================================

df = pd.read_csv(GENERATED_FILE)
print("=" * 70)
print("GENERATED MOLECULE SCREENING")
print("=" * 70)
print(f"Novel generated molecules: {len(df)}")

# ============================================================
# Parse molecules
# ============================================================

def parse_smiles(smiles):
    try:
        return Chem.MolFromSmiles(str(smiles))
    except Exception:
        return None


df["mol"] = df["canonical_smiles"].apply(parse_smiles)
invalid = df["mol"].isna()
if invalid.any():
    print(f"Unexpected invalid molecules removed: {invalid.sum()}")
    df = df.loc[~invalid].copy()

# ============================================================
# Same 10 descriptors used for ESOL model
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


descriptor_df = pd.DataFrame([calculate_descriptors(mol)for mol in df["mol"]])
df = pd.concat([df.reset_index(drop=True), descriptor_df], axis=1)

# ============================================================
# Additional drug-like properties
# ============================================================

df["qed"] = df["mol"].apply(QED.qed)

# ============================================================
# Lipinski Rule-of-Five checks
#
# MW <= 500
# LogP <= 5
# HBD <= 5
# HBA <= 10
# ============================================================

df["lipinski_mw_violation"] = (df["desc_mol_weight"] > 500)
df["lipinski_logp_violation"] = (df["desc_logp"] > 5)
df["lipinski_hbd_violation"] = (df["desc_hbd"] > 5)
df["lipinski_hba_violation"] = (df["desc_hba"] > 10)

violation_columns = ["lipinski_mw_violation", "lipinski_logp_violation", "lipinski_hbd_violation", "lipinski_hba_violation"]

df["lipinski_violations"] = (df[violation_columns].sum(axis=1))
df["lipinski_compliant"] = (df["lipinski_violations"] == 0)

# ============================================================
# Load final solubility model
# ============================================================

model = joblib.load(MODEL_FILE)
X_generated = df[DESCRIPTOR_COLUMNS]
df["predicted_logS"] = model.predict(X_generated)

# Optional conversion:
# logS = log10(mol/L)
# therefore:
# molar solubility = 10^logS
# ============================================================

df["predicted_solubility_mol_L"] = (10 ** df["predicted_logS"])

# ============================================================
# Check whether generated molecules fall inside the descriptor
# ranges seen during model development.
#
# This is a simple applicability-domain flag.
# ============================================================

esol_features = pd.read_csv(ESOL_FEATURE_FILE)
manifest = pd.read_csv(ESOL_MANIFEST_FILE)
development_smiles = set(manifest.loc[manifest["split"].isin(["train", "validation"]), "canonical_smiles"])

development = esol_features[esol_features["canonical_smiles"].isin(development_smiles)].copy()
descriptor_min = development[DESCRIPTOR_COLUMNS].min()
descriptor_max = development[DESCRIPTOR_COLUMNS].max()

outside_range_count = np.zeros(len(df), dtype=int)

for col in DESCRIPTOR_COLUMNS:
    outside = ((df[col] < descriptor_min[col]) | (df[col] > descriptor_max[col]))
    outside_range_count += outside.astype(int)

df["n_descriptors_outside_esol_range"] = (outside_range_count)
df["inside_esol_descriptor_range"] = (df["n_descriptors_outside_esol_range"] == 0)

# ============================================================
# Simple candidate filter
#
# NOTE:
# This is only a first screening rule, not an optimized
# medicinal-chemistry objective.
# ============================================================

df["basic_candidate"] = ((df["qed"] >= 0.5) & (df["lipinski_violations"] <= 1) & (df["inside_esol_descriptor_range"]))

# ============================================================
# Candidate ranking
#
# For now:
# 1. basic filter
# 2. higher QED
# 3. higher predicted solubility
#
# We will build proper multi-objective ranking later.
# ============================================================

candidates = (df[df["basic_candidate"]].sort_values(["qed", "predicted_logS"], ascending=[False, False]).copy())

# ============================================================
# Save
# ============================================================

df.drop(columns=["mol"]).to_csv(OUTPUT_FILE, index=False)
candidates.drop(columns=["mol"]).head(100).to_csv(TOP_FILE, index=False)

# ============================================================
# Summary
# ============================================================

summary = pd.DataFrame([{
    "n_novel_generated": len(df),
    "mean_qed": df["qed"].mean(),
    "median_qed": df["qed"].median(),
    "mean_predicted_logS": df["predicted_logS"].mean(),
    "median_predicted_logS": df["predicted_logS"].median(),
    "lipinski_compliant_fraction":
        df["lipinski_compliant"].mean(),
    "inside_esol_range_fraction":
        df["inside_esol_descriptor_range"].mean(),
    "basic_candidate_count":
        df["basic_candidate"].sum(),
    "basic_candidate_fraction":
        df["basic_candidate"].mean(),
}])

summary.to_csv(SUMMARY_FILE, index=False)

# ============================================================
# Report
# ============================================================

print()
print("=" * 70)
print("SCREENING SUMMARY")
print("=" * 70)

print(f"Generated molecules:       {len(df)}")
print(f"Mean QED:                  {df['qed'].mean():.3f}")

print(
    f"Lipinski compliant:        "
    f"{df['lipinski_compliant'].mean():.3f}"
)

print(
    f"Inside ESOL feature range: "
    f"{df['inside_esol_descriptor_range'].mean():.3f}"
)

print(
    f"Mean predicted logS:       "
    f"{df['predicted_logS'].mean():.3f}"
)

print(
    f"Basic candidates:          "
    f"{df['basic_candidate'].sum()}"
)

print()
print("Top 10 candidates:")

print(candidates[["canonical_smiles", "qed", "predicted_logS", "desc_mol_weight", "desc_logp", "desc_tpsa", "lipinski_violations"]]
      .head(10).to_string(index=False))

print()
print("Saved:")
print(f"  {OUTPUT_FILE}")
print(f"  {TOP_FILE}")
print(f"  {SUMMARY_FILE}")