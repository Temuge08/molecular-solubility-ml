#!/usr/bin/env python3

from pathlib import Path
from collections import defaultdict
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

# ============================================================
# Configuration
# ============================================================

INPUT_FILE = Path("data/processed/esol_curated.csv")

OUTPUT_DIR = Path("data/processed")
REPORT_DIR = Path("reports")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


TRAIN_FRACTION = 0.80
VALID_FRACTION = 0.10
TEST_FRACTION = 0.10


# ============================================================
# Load curated data
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("BEMIS-MURCKO SCAFFOLD SPLIT")
print("=" * 70)

print(f"Total molecules: {len(df)}")


# ============================================================
# Generate Bemis-Murcko scaffold
# ============================================================

def get_scaffold(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(
            f"Could not parse curated SMILES: {smiles}"
        )

    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol,
        includeChirality=True,
    )

    return scaffold


df["scaffold"] = df["canonical_smiles"].apply(
    get_scaffold
)


# ============================================================
# Examine acyclic molecules
#
# RDKit returns an empty string when there is no Murcko
# scaffold, usually for fully acyclic molecules.
# ============================================================

df["is_acyclic_scaffold"] = df["scaffold"] == ""

n_acyclic = df["is_acyclic_scaffold"].sum()

print()
print(f"Acyclic / empty-scaffold molecules: {n_acyclic}")
print(
    f"Acyclic fraction: "
    f"{n_acyclic / len(df):.3f}"
)


# ============================================================
# Group molecules by scaffold
# ============================================================

scaffold_groups = defaultdict(list)

for idx, scaffold in zip(
    df.index,
    df["scaffold"],
):
    scaffold_groups[scaffold].append(idx)


print()
print(f"Unique scaffold groups: {len(scaffold_groups)}")


# ============================================================
# Sort scaffold groups
#
# Largest scaffold families are allocated first.
#
# Tie-breaking by scaffold string keeps the split
# deterministic and reproducible.
# ============================================================

sorted_groups = sorted(
    scaffold_groups.items(),
    key=lambda x: (
        -len(x[1]),
        x[0],
    ),
)


# ============================================================
# Display largest groups
# ============================================================

print()
print("=" * 70)
print("LARGEST SCAFFOLD GROUPS")
print("=" * 70)

for rank, (scaffold, indices) in enumerate(
    sorted_groups[:15],
    start=1,
):

    display_scaffold = (
        scaffold
        if scaffold != ""
        else "<ACYCLIC>"
    )

    print(
        f"{rank:2d}. "
        f"n={len(indices):4d}  "
        f"{display_scaffold}"
    )


# ============================================================
# Desired split sizes
# ============================================================

n_total = len(df)

target_train = int(
    n_total * TRAIN_FRACTION
)

target_valid = int(
    n_total * VALID_FRACTION
)

target_test = (
    n_total
    - target_train
    - target_valid
)


print()
print("=" * 70)
print("TARGET SPLIT SIZES")
print("=" * 70)

print(f"Train target:      {target_train}")
print(f"Validation target: {target_valid}")
print(f"Test target:       {target_test}")


# ============================================================
# Greedy scaffold assignment
#
# Entire scaffold groups always stay together.
#
# We assign each group to the split currently furthest
# below its target fraction.
# ============================================================

split_indices = {
    "train": [],
    "validation": [],
    "test": [],
}

target_sizes = {
    "train": target_train,
    "validation": target_valid,
    "test": target_test,
}


for scaffold, indices in sorted_groups:

    group_size = len(indices)

    # How much capacity remains relative to target?
    deficits = {}

    for split_name in split_indices:

        current_size = len(
            split_indices[split_name]
        )

        target_size = target_sizes[
            split_name
        ]

        deficits[split_name] = (
            target_size - current_size
        )

    # Choose split with largest remaining deficit.
    #
    # Train wins deterministic ties, followed by
    # validation, then test because dict order is fixed.
    best_split = max(
        deficits,
        key=deficits.get,
    )

    split_indices[best_split].extend(
        indices
    )


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
        "Some molecules were not assigned a split."
    )


# ============================================================
# Verify there is no scaffold leakage
# ============================================================

scaffold_to_split_count = (
    df.groupby("scaffold")["split"]
    .nunique()
)

leaking_scaffolds = (
    scaffold_to_split_count[
        scaffold_to_split_count > 1
    ]
)


print()
print("=" * 70)
print("LEAKAGE CHECK")
print("=" * 70)

print(
    "Scaffolds appearing in multiple splits:",
    len(leaking_scaffolds),
)


if len(leaking_scaffolds) > 0:

    raise RuntimeError(
        "Scaffold leakage detected."
    )


# ============================================================
# Split summary
# ============================================================

summary_rows = []

for split_name in [
    "train",
    "validation",
    "test",
]:

    subset = df[
        df["split"] == split_name
    ]

    n_molecules = len(subset)

    n_scaffolds = (
        subset["scaffold"]
        .nunique()
    )

    summary_rows.append(
        {
            "split": split_name,

            "n_molecules":
                n_molecules,

            "fraction":
                n_molecules / len(df),

            "n_scaffolds":
                n_scaffolds,

            "mean_logS":
                subset["measured_logS"].mean(),

            "std_logS":
                subset["measured_logS"].std(),

            "min_logS":
                subset["measured_logS"].min(),

            "max_logS":
                subset["measured_logS"].max(),
        }
    )


summary_df = pd.DataFrame(
    summary_rows
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
# Scaffold-level report
# ============================================================

scaffold_report = (
    df.groupby(
        [
            "scaffold",
            "split",
        ],
        dropna=False,
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
)


scaffold_report = scaffold_report.sort_values(
    "n_molecules",
    ascending=False,
)


# ============================================================
# Save split manifest
#
# This file will become the single source of truth for
# train / validation / test assignment.
# ============================================================

manifest_columns = [
    "compound_ids",
    "canonical_smiles",
    "measured_logS",
    "scaffold",
    "is_acyclic_scaffold",
    "split",
]


manifest = df[
    manifest_columns
].copy()


manifest.to_csv(
    OUTPUT_DIR / "esol_split_manifest.csv",
    index=False,
)


summary_df.to_csv(
    REPORT_DIR / "scaffold_split_summary.csv",
    index=False,
)


scaffold_report.to_csv(
    REPORT_DIR / "scaffold_groups.csv",
    index=False,
)


# ============================================================
# Save explicit molecule lists for convenience
# ============================================================

for split_name in [
    "train",
    "validation",
    "test",
]:

    subset = manifest[
        manifest["split"] == split_name
    ]

    subset.to_csv(
        OUTPUT_DIR
        / f"esol_{split_name}_molecules.csv",
        index=False,
    )


# ============================================================
# Final report
# ============================================================

print()
print("=" * 70)
print("SCAFFOLD SPLIT COMPLETE")
print("=" * 70)

print()
print("Saved:")
print("  data/processed/esol_split_manifest.csv")
print("  data/processed/esol_train_molecules.csv")
print("  data/processed/esol_validation_molecules.csv")
print("  data/processed/esol_test_molecules.csv")
print("  reports/scaffold_split_summary.csv")
print("  reports/scaffold_groups.csv")