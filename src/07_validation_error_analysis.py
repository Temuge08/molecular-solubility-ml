#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from scipy.stats import spearmanr


# ============================================================
# Paths
# ============================================================

DATA_DIR = Path("data/processed")
REPORT_DIR = Path("reports")
FIGURE_DIR = Path("figures")

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


MANIFEST_FILE = (
    DATA_DIR / "esol_cluster_split_manifest.csv"
)

DESC_PRED_FILE = (
    REPORT_DIR
    / "tuned_validation_predictions_descriptors.csv"
)

COMBINED_PRED_FILE = (
    REPORT_DIR
    / "tuned_validation_predictions_combined.csv"
)


# ============================================================
# Morgan settings
#
# Same representation used previously.
# ============================================================

MORGAN_RADIUS = 2
MORGAN_BITS = 2048


# ============================================================
# Load data
# ============================================================

manifest = pd.read_csv(MANIFEST_FILE)

desc_pred = pd.read_csv(DESC_PRED_FILE)

combined_pred = pd.read_csv(COMBINED_PRED_FILE)


print("=" * 70)
print("VALIDATION ERROR ANALYSIS")
print("=" * 70)


# ============================================================
# Prepare predictions
# ============================================================

desc_pred = desc_pred.rename(
    columns={
        "predicted_logS":
            "predicted_logS_descriptors",

        "residual":
            "residual_descriptors",

        "absolute_error":
            "absolute_error_descriptors",
    }
)


combined_pred = combined_pred.rename(
    columns={
        "predicted_logS":
            "predicted_logS_combined",

        "residual":
            "residual_combined",

        "absolute_error":
            "absolute_error_combined",
    }
)


# ============================================================
# Merge both model predictions
# ============================================================

comparison = desc_pred.merge(
    combined_pred[
        [
            "canonical_smiles",
            "predicted_logS_combined",
            "residual_combined",
            "absolute_error_combined",
        ]
    ],
    on="canonical_smiles",
    how="inner",
    validate="one_to_one",
)


print(f"Validation molecules: {len(comparison)}")


# ============================================================
# Compare model errors
# ============================================================

comparison["absolute_error_difference"] = (
    comparison["absolute_error_combined"]
    - comparison["absolute_error_descriptors"]
)


# Negative:
# combined model has smaller error
#
# Positive:
# descriptor model has smaller error


comparison["better_model"] = np.where(
    comparison["absolute_error_combined"]
    < comparison["absolute_error_descriptors"],
    "combined",
    np.where(
        comparison["absolute_error_combined"]
        > comparison["absolute_error_descriptors"],
        "descriptors",
        "tie",
    )
)


print()
print("=" * 70)
print("PER-MOLECULE MODEL COMPARISON")
print("=" * 70)

print(
    comparison["better_model"]
    .value_counts()
)


print()
print(
    "Mean absolute-error difference "
    "(combined - descriptors):"
)

print(
    comparison[
        "absolute_error_difference"
    ].mean()
)


print()
print(
    "Median absolute-error difference "
    "(combined - descriptors):"
)

print(
    comparison[
        "absolute_error_difference"
    ].median()
)


# ============================================================
# Build Morgan fingerprints for applicability analysis
# ============================================================

generator = (
    rdFingerprintGenerator.GetMorganGenerator(
        radius=MORGAN_RADIUS,
        fpSize=MORGAN_BITS,
        includeChirality=True,
    )
)


def get_fp(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(
            f"Invalid molecule: {smiles}"
        )

    return generator.GetFingerprint(mol)


# ============================================================
# Training fingerprints
# ============================================================

train_manifest = manifest[
    manifest["split"] == "train"
].copy()


train_fps = [
    get_fp(smiles)
    for smiles in train_manifest[
        "canonical_smiles"
    ]
]


# ============================================================
# Maximum similarity to training set
# ============================================================

def max_training_similarity(smiles):

    fp = get_fp(smiles)

    similarities = (
        DataStructs.BulkTanimotoSimilarity(
            fp,
            train_fps,
        )
    )

    similarities = np.asarray(
        similarities
    )

    best_index = similarities.argmax()

    return (
        similarities[best_index],
        train_manifest.iloc[
            best_index
        ]["canonical_smiles"],
    )


max_similarities = []
nearest_training_smiles = []


print()
print(
    "Calculating nearest training molecule "
    "for each validation compound..."
)


for smiles in comparison["canonical_smiles"]:

    max_similarity, nearest_smiles = (
        max_training_similarity(smiles)
    )

    max_similarities.append(
        max_similarity
    )

    nearest_training_smiles.append(
        nearest_smiles
    )


comparison[
    "max_train_tanimoto"
] = max_similarities


comparison[
    "nearest_train_smiles"
] = nearest_training_smiles


# ============================================================
# Similarity summary
# ============================================================

print()
print("=" * 70)
print("CHEMICAL SIMILARITY TO TRAINING SET")
print("=" * 70)

print(
    comparison[
        "max_train_tanimoto"
    ].describe()
)


# ============================================================
# Correlation:
# similarity versus model error
# ============================================================

rho_desc, p_desc = spearmanr(
    comparison["max_train_tanimoto"],
    comparison["absolute_error_descriptors"],
)


rho_comb, p_comb = spearmanr(
    comparison["max_train_tanimoto"],
    comparison["absolute_error_combined"],
)


print()
print("=" * 70)
print("SIMILARITY VS ERROR")
print("=" * 70)

print(
    "Descriptors:"
)

print(
    f"  Spearman rho = {rho_desc:.4f}"
)

print(
    f"  p-value      = {p_desc:.4g}"
)


print()

print(
    "Combined:"
)

print(
    f"  Spearman rho = {rho_comb:.4f}"
)

print(
    f"  p-value      = {p_comb:.4g}"
)


# ============================================================
# Similarity bins
# ============================================================

bins = [
    0.0,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    1.01,
]


labels = [
    "<0.30",
    "0.30-0.39",
    "0.40-0.49",
    "0.50-0.59",
    "0.60-0.69",
    "0.70-0.79",
    ">=0.80",
]


comparison["similarity_bin"] = pd.cut(
    comparison["max_train_tanimoto"],
    bins=bins,
    labels=labels,
    right=False,
)


similarity_summary = (
    comparison
    .groupby(
        "similarity_bin",
        observed=False,
    )
    .agg(
        n_molecules=(
            "canonical_smiles",
            "size",
        ),

        mean_similarity=(
            "max_train_tanimoto",
            "mean",
        ),

        descriptor_mae=(
            "absolute_error_descriptors",
            "mean",
        ),

        combined_mae=(
            "absolute_error_combined",
            "mean",
        ),
    )
    .reset_index()
)


print()
print("=" * 70)
print("ERROR BY SIMILARITY BIN")
print("=" * 70)

print(
    similarity_summary.to_string(
        index=False
    )
)


# ============================================================
# Worst predictions
# ============================================================

worst_descriptor = (
    comparison.sort_values(
        "absolute_error_descriptors",
        ascending=False,
    )
    .head(15)
)


worst_combined = (
    comparison.sort_values(
        "absolute_error_combined",
        ascending=False,
    )
    .head(15)
)


print()
print("=" * 70)
print("10 WORST DESCRIPTOR PREDICTIONS")
print("=" * 70)

print(
    worst_descriptor[
        [
            "compound_ids",
            "measured_logS",
            "predicted_logS_descriptors",
            "absolute_error_descriptors",
            "max_train_tanimoto",
        ]
    ]
    .head(10)
    .to_string(index=False)
)


print()
print("=" * 70)
print("10 WORST COMBINED PREDICTIONS")
print("=" * 70)

print(
    worst_combined[
        [
            "compound_ids",
            "measured_logS",
            "predicted_logS_combined",
            "absolute_error_combined",
            "max_train_tanimoto",
        ]
    ]
    .head(10)
    .to_string(index=False)
)


# ============================================================
# Save reports
# ============================================================

comparison.to_csv(
    REPORT_DIR
    / "validation_error_analysis.csv",
    index=False,
)


similarity_summary.to_csv(
    REPORT_DIR
    / "validation_error_by_similarity.csv",
    index=False,
)


worst_descriptor.to_csv(
    REPORT_DIR
    / "worst_descriptor_predictions.csv",
    index=False,
)


worst_combined.to_csv(
    REPORT_DIR
    / "worst_combined_predictions.csv",
    index=False,
)


# ============================================================
# Plot 1:
# chemical similarity vs descriptor error
# ============================================================

plt.figure(figsize=(7, 5))

plt.scatter(
    comparison["max_train_tanimoto"],
    comparison["absolute_error_descriptors"],
    alpha=0.7,
)

plt.xlabel(
    "Maximum Tanimoto similarity to training set"
)

plt.ylabel(
    "Absolute prediction error (logS)"
)

plt.title(
    "Descriptor RF: Error vs Chemical Similarity"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "descriptor_error_vs_similarity.png",
    dpi=300,
)

plt.close()


# ============================================================
# Plot 2:
# chemical similarity vs combined error
# ============================================================

plt.figure(figsize=(7, 5))

plt.scatter(
    comparison["max_train_tanimoto"],
    comparison["absolute_error_combined"],
    alpha=0.7,
)

plt.xlabel(
    "Maximum Tanimoto similarity to training set"
)

plt.ylabel(
    "Absolute prediction error (logS)"
)

plt.title(
    "Combined RF: Error vs Chemical Similarity"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "combined_error_vs_similarity.png",
    dpi=300,
)

plt.close()


# ============================================================
# Plot 3:
# measured vs predicted — descriptor
# ============================================================

plt.figure(figsize=(6, 6))

plt.scatter(
    comparison["measured_logS"],
    comparison["predicted_logS_descriptors"],
    alpha=0.7,
)


lims = [
    min(
        comparison["measured_logS"].min(),
        comparison[
            "predicted_logS_descriptors"
        ].min(),
    ),
    max(
        comparison["measured_logS"].max(),
        comparison[
            "predicted_logS_descriptors"
        ].max(),
    ),
]


plt.plot(
    lims,
    lims,
    linestyle="--",
)

plt.xlabel(
    "Measured logS"
)

plt.ylabel(
    "Predicted logS"
)

plt.title(
    "Descriptor RF: Measured vs Predicted"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "descriptor_measured_vs_predicted.png",
    dpi=300,
)

plt.close()


# ============================================================
# Plot 4:
# measured vs predicted — combined
# ============================================================

plt.figure(figsize=(6, 6))

plt.scatter(
    comparison["measured_logS"],
    comparison["predicted_logS_combined"],
    alpha=0.7,
)


lims = [
    min(
        comparison["measured_logS"].min(),
        comparison[
            "predicted_logS_combined"
        ].min(),
    ),
    max(
        comparison["measured_logS"].max(),
        comparison[
            "predicted_logS_combined"
        ].max(),
    ),
]


plt.plot(
    lims,
    lims,
    linestyle="--",
)


plt.xlabel(
    "Measured logS"
)

plt.ylabel(
    "Predicted logS"
)

plt.title(
    "Combined RF: Measured vs Predicted"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "combined_measured_vs_predicted.png",
    dpi=300,
)

plt.close()


print()
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print()
print("Saved reports:")
print(
    "  reports/validation_error_analysis.csv"
)
print(
    "  reports/validation_error_by_similarity.csv"
)
print(
    "  reports/worst_descriptor_predictions.csv"
)
print(
    "  reports/worst_combined_predictions.csv"
)

print()
print("Saved figures:")
print(
    "  figures/descriptor_error_vs_similarity.png"
)
print(
    "  figures/combined_error_vs_similarity.png"
)
print(
    "  figures/descriptor_measured_vs_predicted.png"
)
print(
    "  figures/combined_measured_vs_predicted.png"
)

print()
print(
    "TEST SET HAS NOT BEEN USED."
)