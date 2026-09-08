#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Cluster import Butina


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = Path(
    "data/processed/esol_curated.csv"
)

OUTPUT_DIR = Path(
    "data/processed"
)

REPORT_DIR = Path(
    "reports"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ------------------------------------------------------------
# Morgan fingerprint
# ------------------------------------------------------------

MORGAN_RADIUS = 2
MORGAN_BITS = 2048


# ------------------------------------------------------------
# Butina cutoff
#
# Butina receives DISTANCE cutoff:
#
# distance = 1 - Tanimoto similarity
#
# 0.6 distance therefore corresponds approximately to
# Tanimoto similarity >= 0.4 for cluster neighbors.
# ------------------------------------------------------------

DISTANCE_CUTOFF = 0.60


TRAIN_FRACTION = 0.80
VALID_FRACTION = 0.10
TEST_FRACTION = 0.10


# ============================================================
# Load molecules
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("CHEMICAL SIMILARITY CLUSTER SPLIT")
print("=" * 70)

print(f"Molecules: {len(df)}")


# ============================================================
# Parse SMILES
# ============================================================

molecules = []

for smiles in df["canonical_smiles"]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise RuntimeError(
            f"Invalid curated molecule: {smiles}"
        )
    molecules.append(mol)


# ============================================================
# Morgan fingerprints
# ============================================================

generator = (
    rdFingerprintGenerator
    .GetMorganGenerator(
        radius=MORGAN_RADIUS,
        fpSize=MORGAN_BITS,
        includeChirality=True,
    )
)

fingerprints = [
    generator.GetFingerprint(mol)
    for mol in molecules
]

# ============================================================
# Pairwise Tanimoto distances
#
# Butina expects a lower-triangular distance matrix:
#
# d = 1 - similarity
# ============================================================

print()
print("Calculating pairwise Tanimoto distances...")

distances = []
for i in range(1, len(fingerprints)):
    similarities = (
        DataStructs.BulkTanimotoSimilarity(
            fingerprints[i],
            fingerprints[:i],
        )
    )
    distances.extend(
        [
            1.0 - sim
            for sim in similarities
        ]
    )

# ============================================================
# Butina clustering
# ============================================================

clusters = Butina.ClusterData(
    distances,
    len(fingerprints),
    DISTANCE_CUTOFF,
    isDistData=True,
)


# Convert tuples to lists
clusters = [
    list(cluster)
    for cluster in clusters
]


print()
print("=" * 70)
print("CLUSTER SUMMARY")
print("=" * 70)

print(
    f"Distance cutoff: {DISTANCE_CUTOFF}"
)

print(
    "Equivalent neighbor similarity:",
    f"{1 - DISTANCE_CUTOFF:.2f}",
)

print(
    f"Number of clusters: {len(clusters)}"
)


cluster_sizes = np.array(
    [
        len(cluster)
        for cluster in clusters
    ]
)


print(
    f"Largest cluster: {cluster_sizes.max()}"
)
print(
    f"Median cluster size: "
    f"{np.median(cluster_sizes):.1f}"
)
print(
    f"Singleton clusters: "
    f"{np.sum(cluster_sizes == 1)}"
)

print()
print("15 largest clusters:")

for i, cluster in enumerate(
    clusters[:15],
    start=1,
):

    print(
        f"{i:2d}. n={len(cluster):4d}"
    )


# ============================================================
# Assign cluster ID to every molecule
# ============================================================

df["cluster_id"] = -1


for cluster_id, indices in enumerate(
    clusters
):

    df.loc[
        indices,
        "cluster_id",
    ] = cluster_id


if (df["cluster_id"] < 0).any():

    raise RuntimeError(
        "Some molecules were not assigned a cluster."
    )


# ============================================================
# Target split sizes
# ============================================================

n_total = len(df)

target_sizes = {
    "train": int(
        n_total * TRAIN_FRACTION
    ),

    "validation": int(
        n_total * VALID_FRACTION
    ),
}

target_sizes["test"] = (
    n_total
    - target_sizes["train"]
    - target_sizes["validation"]
)


print()
print("=" * 70)
print("TARGET SPLIT SIZES")
print("=" * 70)

for split_name, n in target_sizes.items():

    print(
        f"{split_name:10s}: {n}"
    )


# ============================================================
# Assign entire clusters
#
# Largest clusters first.
# Each cluster goes to the split with the largest remaining
# target deficit.
# ============================================================

split_indices = {
    "train": [],
    "validation": [],
    "test": [],
}


clusters_sorted = sorted(
    clusters,
    key=len,
    reverse=True,
)


for cluster in clusters_sorted:

    deficits = {}

    for split_name in split_indices:

        current = len(
            split_indices[split_name]
        )

        deficits[split_name] = (
            target_sizes[split_name]
            - current
        )


    best_split = max(
        deficits,
        key=deficits.get,
    )


    split_indices[
        best_split
    ].extend(cluster)


# ============================================================
# Assign split labels
# ============================================================

df["split"] = None


for split_name, indices in split_indices.items():

    df.loc[
        indices,
        "split",
    ] = split_name


if df["split"].isna().any():

    raise RuntimeError(
        "Some molecules have no split assignment."
    )


# ============================================================
# Check cluster leakage
# ============================================================

cluster_split_counts = (
    df.groupby(
        "cluster_id"
    )["split"]
    .nunique()
)


leaking_clusters = (
    cluster_split_counts[
        cluster_split_counts > 1
    ]
)


print()
print("=" * 70)
print("LEAKAGE CHECK")
print("=" * 70)

print(
    "Clusters appearing in multiple splits:",
    len(leaking_clusters),
)


if len(leaking_clusters) > 0:

    raise RuntimeError(
        "Cluster leakage detected."
    )


# ============================================================
# Split statistics
# ============================================================

summary = []


for split_name in [
    "train",
    "validation",
    "test",
]:

    subset = df[
        df["split"] == split_name
    ]


    summary.append(
        {
            "split":
                split_name,

            "n_molecules":
                len(subset),

            "fraction":
                len(subset) / len(df),

            "n_clusters":
                subset[
                    "cluster_id"
                ].nunique(),

            "mean_logS":
                subset[
                    "measured_logS"
                ].mean(),

            "std_logS":
                subset[
                    "measured_logS"
                ].std(),

            "min_logS":
                subset[
                    "measured_logS"
                ].min(),

            "max_logS":
                subset[
                    "measured_logS"
                ].max(),
        }
    )


summary_df = pd.DataFrame(
    summary
)


print()
print("=" * 70)
print("ACTUAL SPLIT SUMMARY")
print("=" * 70)

print(
    summary_df.to_string(
        index=False
    )
)


# ============================================================
# Save split manifest
# ============================================================

manifest = df[
    [
        "compound_ids",
        "canonical_smiles",
        "measured_logS",
        "cluster_id",
        "split",
    ]
].copy()


manifest.to_csv(
    OUTPUT_DIR
    / "esol_cluster_split_manifest.csv",
    index=False,
)


summary_df.to_csv(
    REPORT_DIR
    / "cluster_split_summary.csv",
    index=False,
)


# ============================================================
# Cluster report
# ============================================================

cluster_report = (
    df.groupby(
        [
            "cluster_id",
            "split",
        ]
    )
    .agg(
        n_molecules=(
            "canonical_smiles",
            "size",
        ),

        mean_logS=(
            "measured_logS",
            "mean",
        ),
    )
    .reset_index()
    .sort_values(
        "n_molecules",
        ascending=False,
    )
)


cluster_report.to_csv(
    REPORT_DIR
    / "chemical_clusters.csv",
    index=False,
)


print()
print("=" * 70)
print("CLUSTER SPLIT COMPLETE")
print("=" * 70)

print()
print("Saved:")
print(
    "  data/processed/"
    "esol_cluster_split_manifest.csv"
)
print(
    "  reports/"
    "cluster_split_summary.csv"
)
print(
    "  reports/"
    "chemical_clusters.csv"
)