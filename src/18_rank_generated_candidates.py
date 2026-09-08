#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


# ============================================================
# Paths
# ============================================================

INPUT_FILE = Path(
    "data/generation/processed/vae_generated_screened.csv"
)

OUTPUT_DIR = Path("reports/generation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Configuration
# ============================================================

# Hard filtering rules
MIN_QED = 0.50
MAX_LIPINSKI_VIOLATIONS = 1
REQUIRE_ESOL_DOMAIN = True

# Weighted ranking
WEIGHT_QED = 0.45
WEIGHT_SOLUBILITY = 0.35
WEIGHT_LIPINSKI = 0.20

# Diversity selection
N_DIVERSE_CANDIDATES = 100
MAX_TANIMOTO_BETWEEN_SELECTED = 0.65

MORGAN_RADIUS = 2
MORGAN_BITS = 2048


# ============================================================
# Load
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("MULTI-OBJECTIVE MOLECULAR RANKING")
print("=" * 70)

print(f"Input novel molecules: {len(df)}")


# ============================================================
# 1. Hard filtering
#
# These rules define molecules we are willing to rank.
# ============================================================

mask = (
    (df["qed"] >= MIN_QED)
    &
    (df["lipinski_violations"] <= MAX_LIPINSKI_VIOLATIONS)
)

if REQUIRE_ESOL_DOMAIN:
    mask &= df["inside_esol_descriptor_range"]

candidates = df.loc[mask].copy()


print()
print("=" * 70)
print("HARD FILTER")
print("=" * 70)

print(f"Remaining candidates: {len(candidates)}")
print(
    f"Fraction retained:     "
    f"{len(candidates) / len(df):.3f}"
)


if len(candidates) == 0:
    raise RuntimeError("No molecules passed the hard filter.")


# ============================================================
# 2. Normalize objectives using percentile ranks
#
# Higher is always better.
#
# This prevents numerical scale differences between:
#   QED          ~ 0-1
#   predicted logS ~ negative continuous values
# ============================================================

candidates["qed_score"] = candidates["qed"].rank(
    pct=True,
    method="average",
)

candidates["solubility_score"] = candidates["predicted_logS"].rank(
    pct=True,
    method="average",
)


# ============================================================
# Lipinski score
#
# 0 violations -> 1.00
# 1 violation  -> 0.75
# 2 violations -> 0.50
# ...
# ============================================================

candidates["lipinski_score"] = (
    1.0 - candidates["lipinski_violations"] / 4.0
)

candidates["lipinski_score"] = candidates[
    "lipinski_score"
].clip(lower=0.0, upper=1.0)


# ============================================================
# 3. Weighted multi-objective score
#
# These weights are heuristic.
#
# They are intentionally explicit so they can later be replaced
# by project-specific requirements.
# ============================================================

candidates["multiobjective_score"] = (
    WEIGHT_QED * candidates["qed_score"]
    + WEIGHT_SOLUBILITY * candidates["solubility_score"]
    + WEIGHT_LIPINSKI * candidates["lipinski_score"]
)


# ============================================================
# 4. Pareto front
#
# We consider two continuous objectives here:
#
# maximize QED
# maximize predicted logS
#
# Lipinski/domain constraints were already handled as filters.
#
# A Pareto molecule cannot improve one objective without
# becoming worse in another.
# ============================================================

def mark_pareto_front(data):
    ordered = data.sort_values(
        ["qed", "predicted_logS"],
        ascending=[False, False],
    )

    pareto_indices = []
    best_logS = -np.inf

    for idx, row in ordered.iterrows():
        if row["predicted_logS"] > best_logS:
            pareto_indices.append(idx)
            best_logS = row["predicted_logS"]

    result = pd.Series(False, index=data.index)
    result.loc[pareto_indices] = True

    return result


candidates["pareto_front"] = mark_pareto_front(candidates)

print()
print("=" * 70)
print("PARETO ANALYSIS")
print("=" * 70)

print(
    "Pareto-optimal molecules:",
    int(candidates["pareto_front"].sum()),
)


# ============================================================
# 5. Rank
#
# Pareto candidates are placed first.
# Within each group, use multi-objective score.
# ============================================================

candidates = candidates.sort_values(
    ["pareto_front", "multiobjective_score"],
    ascending=[False, False],
).reset_index(drop=True)

candidates["overall_rank"] = np.arange(
    1,
    len(candidates) + 1,
)


# ============================================================
# 6. Chemical diversity selection
#
# Simply taking the top 100 may return many near-duplicates.
#
# Therefore greedily accept candidates only if their maximum
# Tanimoto similarity to already-selected molecules is <= 0.65.
# ============================================================

generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=MORGAN_RADIUS,
    fpSize=MORGAN_BITS,
    includeChirality=True,
)


def get_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    return generator.GetFingerprint(mol)


selected_rows = []
selected_fps = []


for _, row in candidates.iterrows():

    fp = get_fingerprint(row["canonical_smiles"])

    if fp is None:
        continue

    if len(selected_fps) == 0:
        max_similarity = 0.0

    else:
        similarities = DataStructs.BulkTanimotoSimilarity(
            fp,
            selected_fps,
        )

        max_similarity = max(similarities)


    if max_similarity <= MAX_TANIMOTO_BETWEEN_SELECTED:
        selected = row.copy()

        selected["max_similarity_to_previous_selected"] = (
            max_similarity
        )

        selected_rows.append(selected)
        selected_fps.append(fp)


    if len(selected_rows) >= N_DIVERSE_CANDIDATES:
        break


diverse_candidates = pd.DataFrame(selected_rows)


if len(diverse_candidates) > 0:
    diverse_candidates["diverse_rank"] = np.arange(
        1,
        len(diverse_candidates) + 1,
    )


# ============================================================
# Save outputs
# ============================================================

candidates.to_csv(
    OUTPUT_DIR / "ranked_generated_candidates.csv",
    index=False,
)

candidates[
    candidates["pareto_front"]
].to_csv(
    OUTPUT_DIR / "pareto_generated_candidates.csv",
    index=False,
)

diverse_candidates.to_csv(
    OUTPUT_DIR / "top_diverse_generated_candidates.csv",
    index=False,
)


# ============================================================
# Summary
# ============================================================

summary = pd.DataFrame([{
    "n_input": len(df),
    "n_after_hard_filter": len(candidates),
    "n_pareto": int(candidates["pareto_front"].sum()),
    "n_diverse_selected": len(diverse_candidates),

    "min_qed_filter": MIN_QED,
    "max_lipinski_violations": MAX_LIPINSKI_VIOLATIONS,

    "weight_qed": WEIGHT_QED,
    "weight_solubility": WEIGHT_SOLUBILITY,
    "weight_lipinski": WEIGHT_LIPINSKI,

    "diversity_tanimoto_threshold":
        MAX_TANIMOTO_BETWEEN_SELECTED,
}])

summary.to_csv(
    OUTPUT_DIR / "candidate_ranking_summary.csv",
    index=False,
)


# ============================================================
# Report
# ============================================================

print()
print("=" * 70)
print("MULTI-OBJECTIVE RANKING SUMMARY")
print("=" * 70)

print(f"Input molecules:          {len(df)}")
print(f"After filtering:          {len(candidates)}")

print(
    f"Pareto-optimal:           "
    f"{candidates['pareto_front'].sum()}"
)

print(
    f"Diverse candidates:       "
    f"{len(diverse_candidates)}"
)

print()
print("=" * 70)
print("TOP 10 RANKED MOLECULES")
print("=" * 70)

columns = [
    "overall_rank",
    "canonical_smiles",
    "qed",
    "predicted_logS",
    "lipinski_violations",
    "multiobjective_score",
    "pareto_front",
]

print(
    candidates[columns]
    .head(10)
    .to_string(index=False)
)


print()
print("=" * 70)
print("TOP 10 DIVERSE MOLECULES")
print("=" * 70)

if len(diverse_candidates) > 0:

    columns = [
        "diverse_rank",
        "canonical_smiles",
        "qed",
        "predicted_logS",
        "multiobjective_score",
        "max_similarity_to_previous_selected",
    ]

    print(
        diverse_candidates[columns]
        .head(10)
        .to_string(index=False)
    )


print()
print("Saved:")
print("  reports/generation/ranked_generated_candidates.csv")
print("  reports/generation/pareto_generated_candidates.csv")
print("  reports/generation/top_diverse_generated_candidates.csv")
print("  reports/generation/candidate_ranking_summary.csv")