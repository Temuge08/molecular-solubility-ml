#!/usr/bin/env python3

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import Draw


# ============================================================
# Paths
# ============================================================

REPORT_DIR = Path("reports/generation")
FIGURE_DIR = Path("figures/generation")

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

RANKED_FILE = REPORT_DIR / "ranked_generated_candidates.csv"
DIVERSE_FILE = REPORT_DIR / "top_diverse_generated_candidates.csv"


# ============================================================
# Load
# ============================================================

ranked = pd.read_csv(RANKED_FILE)
diverse = pd.read_csv(DIVERSE_FILE)

print("=" * 70)
print("GENERATED CANDIDATE VISUALIZATION")
print("=" * 70)

print(f"Ranked candidates:  {len(ranked)}")
print(f"Diverse candidates: {len(diverse)}")


# ============================================================
# 1. Top 20 molecular structures
# ============================================================

top = ranked.head(20).copy()

molecules = []
legends = []

for _, row in top.iterrows():
    mol = Chem.MolFromSmiles(row["canonical_smiles"])

    if mol is None:
        continue

    molecules.append(mol)

    legends.append(
        f"Rank {int(row['overall_rank'])}\n"
        f"QED {row['qed']:.3f}\n"
        f"logS {row['predicted_logS']:.2f}"
    )


grid = Draw.MolsToGridImage(
    molecules,
    molsPerRow=4,
    subImgSize=(350, 280),
    legends=legends,
)

grid.save(
    FIGURE_DIR / "top20_generated_molecules.png"
)


# ============================================================
# 2. QED vs predicted solubility
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    ranked["predicted_logS"],
    ranked["qed"],
    alpha=0.25,
    s=15,
    label="Ranked candidates",
)

pareto = ranked[ranked["pareto_front"]].copy()

plt.scatter(
    pareto["predicted_logS"],
    pareto["qed"],
    s=50,
    label="Pareto front",
)

plt.xlabel("Predicted logS")
plt.ylabel("QED")
plt.title("Generated Molecules: QED vs Predicted Solubility")
plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "qed_vs_predicted_logs.png",
    dpi=300,
)

plt.close()


# ============================================================
# 3. Pareto front with top-ranked molecules labeled
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    ranked["predicted_logS"],
    ranked["qed"],
    alpha=0.15,
    s=15,
)

plt.scatter(
    pareto["predicted_logS"],
    pareto["qed"],
    s=60,
    label="Pareto-optimal",
)


for _, row in ranked.head(10).iterrows():
    plt.annotate(
        str(int(row["overall_rank"])),
        (
            row["predicted_logS"],
            row["qed"],
        ),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=8,
    )


plt.xlabel("Predicted logS")
plt.ylabel("QED")
plt.title("Pareto Trade-off: Drug-likeness vs Solubility")
plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "pareto_front.png",
    dpi=300,
)

plt.close()


# ============================================================
# 4. Diverse top-100 chemical-property map
#
# MW vs LogP, with marker size determined by QED.
# ============================================================

plt.figure(figsize=(8, 6))

marker_sizes = (
    diverse["qed"] * 80
)

plt.scatter(
    diverse["desc_mol_weight"],
    diverse["desc_logp"],
    s=marker_sizes,
    alpha=0.7,
)

plt.xlabel("Molecular weight")
plt.ylabel("LogP")

plt.title(
    "Top Chemically Diverse Generated Candidates"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "diverse_candidates_mw_logp.png",
    dpi=300,
)

plt.close()


# ============================================================
# 5. Top 20 diverse molecules
# ============================================================

top_diverse = diverse.head(20)

molecules = []
legends = []


for _, row in top_diverse.iterrows():
    mol = Chem.MolFromSmiles(
        row["canonical_smiles"]
    )

    if mol is None:
        continue

    molecules.append(mol)

    legends.append(
        f"D{int(row['diverse_rank'])}\n"
        f"QED {row['qed']:.3f}\n"
        f"logS {row['predicted_logS']:.2f}"
    )


grid = Draw.MolsToGridImage(
    molecules,
    molsPerRow=4,
    subImgSize=(350, 280),
    legends=legends,
)

grid.save(
    FIGURE_DIR / "top20_diverse_molecules.png"
)


# ============================================================
# Report
# ============================================================

print()
print("Saved:")
print("  figures/generation/top20_generated_molecules.png")
print("  figures/generation/qed_vs_predicted_logs.png")
print("  figures/generation/pareto_front.png")
print("  figures/generation/diverse_candidates_mw_logp.png")
print("  figures/generation/top20_diverse_molecules.png")

print()
print("=" * 70)
print("STEP 19 COMPLETE")
print("=" * 70)