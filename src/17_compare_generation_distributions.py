#!/usr/bin/env python3

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import Descriptors, QED, rdMolDescriptors
from scipy.stats import ks_2samp


# ============================================================
# Configuration
# ============================================================

TRAIN_FILE = Path("data/generation/processed/moses_100k_split.csv")
GENERATED_FILE = Path("data/generation/processed/vae_generated_screened.csv")
MODEL_FILE = Path("models/final_esol_descriptor_rf.joblib")

REPORT_DIR = Path("reports/generation")
FIGURE_DIR = Path("figures/generation")

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_SAMPLE_SIZE = 20_000
RANDOM_STATE = 42


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


COMPARE_PROPERTIES = [
    "desc_mol_weight",
    "desc_logp",
    "desc_tpsa",
    "desc_fraction_csp3",
    "qed",
    "predicted_logS",
]


# ============================================================
# Load generated molecules
# ============================================================

generated = pd.read_csv(GENERATED_FILE)

print("=" * 70)
print("TRAINING VS GENERATED CHEMICAL SPACE")
print("=" * 70)

print(f"Generated novel molecules: {len(generated)}")


# ============================================================
# Load MOSES training molecules
# ============================================================

source = pd.read_csv(TRAIN_FILE)

train = source[source["split"] == "train"].copy()

if len(train) > REFERENCE_SAMPLE_SIZE:
    train = train.sample(
        n=REFERENCE_SAMPLE_SIZE,
        random_state=RANDOM_STATE,
    )

train = train.reset_index(drop=True)

print(f"MOSES reference molecules: {len(train)}")


# ============================================================
# Calculate properties for MOSES reference molecules
# ============================================================

def calculate_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    return {
        "desc_mol_weight": Descriptors.MolWt(mol),
        "desc_logp": Descriptors.MolLogP(mol),
        "desc_tpsa": rdMolDescriptors.CalcTPSA(mol),
        "desc_hbd": rdMolDescriptors.CalcNumHBD(mol),
        "desc_hba": rdMolDescriptors.CalcNumHBA(mol),
        "desc_rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "desc_ring_count": rdMolDescriptors.CalcNumRings(mol),
        "desc_aromatic_ring_count": rdMolDescriptors.CalcNumAromaticRings(mol),
        "desc_heavy_atom_count": Descriptors.HeavyAtomCount(mol),
        "desc_fraction_csp3": rdMolDescriptors.CalcFractionCSP3(mol),
        "qed": QED.qed(mol),
    }


records = []

for smiles in train["canonical_smiles"]:
    record = calculate_properties(smiles)

    if record is None:
        continue

    record["canonical_smiles"] = smiles
    records.append(record)


train_properties = pd.DataFrame(records)


# ============================================================
# Predict solubility for training reference
# ============================================================

model = joblib.load(MODEL_FILE)

X_train_reference = train_properties[DESCRIPTOR_COLUMNS]

train_properties["predicted_logS"] = model.predict(X_train_reference)


# ============================================================
# Add source labels
# ============================================================

train_properties["source"] = "MOSES training"
generated["source"] = "VAE generated"


# ============================================================
# Descriptive statistics
# ============================================================

summary_rows = []

for prop in COMPARE_PROPERTIES:
    for name, df in [
        ("MOSES training", train_properties),
        ("VAE generated", generated),
    ]:
        values = df[prop].dropna()

        summary_rows.append({
            "property": prop,
            "source": name,
            "n": len(values),
            "mean": values.mean(),
            "std": values.std(),
            "median": values.median(),
            "min": values.min(),
            "max": values.max(),
        })


summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(
    REPORT_DIR / "training_vs_generated_property_summary.csv",
    index=False,
)


# ============================================================
# KS tests
#
# KS statistic:
# 0   -> very similar distributions
# 1   -> completely different distributions
# ============================================================

ks_rows = []

for prop in COMPARE_PROPERTIES:
    reference_values = train_properties[prop].dropna().to_numpy()
    generated_values = generated[prop].dropna().to_numpy()

    ks = ks_2samp(
        reference_values,
        generated_values,
    )

    ks_rows.append({
        "property": prop,
        "ks_statistic": ks.statistic,
        "p_value": ks.pvalue,
        "training_mean": reference_values.mean(),
        "generated_mean": generated_values.mean(),
        "mean_difference": generated_values.mean() - reference_values.mean(),
    })


ks_df = pd.DataFrame(ks_rows)

ks_df.to_csv(
    REPORT_DIR / "training_vs_generated_ks_tests.csv",
    index=False,
)


# ============================================================
# Print comparison
# ============================================================

print()
print("=" * 70)
print("DISTRIBUTION COMPARISON")
print("=" * 70)

print(
    ks_df[
        [
            "property",
            "ks_statistic",
            "p_value",
            "training_mean",
            "generated_mean",
            "mean_difference",
        ]
    ].to_string(index=False)
)


# ============================================================
# Plot distributions
# ============================================================

plot_labels = {
    "desc_mol_weight": "Molecular weight",
    "desc_logp": "LogP",
    "desc_tpsa": "TPSA",
    "desc_fraction_csp3": "Fraction Csp3",
    "qed": "QED",
    "predicted_logS": "Predicted logS",
}


for prop in COMPARE_PROPERTIES:
    plt.figure(figsize=(7, 5))

    plt.hist(
        train_properties[prop].dropna(),
        bins=40,
        density=True,
        alpha=0.5,
        label="MOSES training",
    )

    plt.hist(
        generated[prop].dropna(),
        bins=40,
        density=True,
        alpha=0.5,
        label="VAE generated",
    )

    plt.xlabel(plot_labels[prop])
    plt.ylabel("Density")
    plt.title(f"{plot_labels[prop]} Distribution")
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / f"training_vs_generated_{prop}.png",
        dpi=300,
    )

    plt.close()


# ============================================================
# Save reference property table
# ============================================================

train_properties.to_csv(
    REPORT_DIR / "moses_reference_properties.csv",
    index=False,
)

print()
print("=" * 70)
print("STEP 17 COMPLETE")
print("=" * 70)

print()
print("Saved:")
print("  reports/generation/training_vs_generated_property_summary.csv")
print("  reports/generation/training_vs_generated_ks_tests.csv")
print("  reports/generation/moses_reference_properties.csv")
print("  figures/generation/training_vs_generated_*.png")