#!/usr/bin/env python3

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance


# ============================================================
# Paths
# ============================================================

DATA_DIR = Path("data/processed")
REPORT_DIR = Path("reports")
FIGURE_DIR = Path("figures")
MODEL_DIR = Path("models")

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


FEATURE_FILE = (
    DATA_DIR / "esol_descriptors.csv"
)

MANIFEST_FILE = (
    DATA_DIR / "esol_cluster_split_manifest.csv"
)

MODEL_FILE = (
    MODEL_DIR / "rf_tuned_descriptors.joblib"
)


# ============================================================
# Load
# ============================================================

features = pd.read_csv(FEATURE_FILE)
manifest = pd.read_csv(MANIFEST_FILE)

model = joblib.load(MODEL_FILE)


# ============================================================
# Feature columns
# ============================================================

descriptor_columns = [
    c
    for c in features.columns
    if c.startswith("desc_")
]


# ============================================================
# Attach split
# ============================================================

data = features.merge(
    manifest[
        [
            "canonical_smiles",
            "split",
        ]
    ],
    on="canonical_smiles",
    how="inner",
    validate="one_to_one",
)


validation = data[
    data["split"] == "validation"
].copy()

X_val = validation[
    descriptor_columns
].to_numpy()

y_val = validation[
    "measured_logS"
].to_numpy()


# ============================================================
# Baseline validation performance
# ============================================================

pred = model.predict(X_val)

baseline_rmse = np.sqrt(
    mean_squared_error(
        y_val,
        pred,
    )
)


print("=" * 70)
print("DESCRIPTOR FEATURE IMPORTANCE")
print("=" * 70)

print(
    f"Validation RMSE: {baseline_rmse:.4f}"
)


# ============================================================
# Permutation importance
#
# scoring = negative MSE
#
# Higher importance means performance worsens more when
# that feature is randomly shuffled.
# ============================================================

result = permutation_importance(
    model,
    X_val,
    y_val,
    scoring="neg_mean_squared_error",
    n_repeats=50,
    random_state=42,
    n_jobs=-1,
)


importance_df = pd.DataFrame(
    {
        "feature":
            descriptor_columns,

        "importance_mean":
            result.importances_mean,

        "importance_std":
            result.importances_std,
    }
)


importance_df = importance_df.sort_values(
    "importance_mean",
    ascending=False,
)


print()
print("Permutation importance:")
print(
    importance_df.to_string(
        index=False
    )
)


# ============================================================
# Built-in RF importance
#
# Useful as secondary information, but permutation
# importance is our primary interpretation.
# ============================================================

importance_df["rf_impurity_importance"] = [
    model.feature_importances_[
        descriptor_columns.index(feature)
    ]
    for feature in importance_df["feature"]
]


importance_df.to_csv(
    REPORT_DIR
    / "descriptor_feature_importance.csv",
    index=False,
)


# ============================================================
# Plot
# ============================================================

plot_df = importance_df.sort_values(
    "importance_mean",
    ascending=True,
)


plt.figure(
    figsize=(8, 5)
)

plt.barh(
    plot_df["feature"],
    plot_df["importance_mean"],
    xerr=plot_df["importance_std"],
)

plt.xlabel(
    "Permutation importance "
    "(increase in prediction error)"
)

plt.ylabel(
    "Descriptor"
)

plt.title(
    "ESOL Descriptor Random Forest Feature Importance"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "descriptor_feature_importance.png",
    dpi=300,
)

plt.close()


print()
print("Saved:")
print(
    "  reports/descriptor_feature_importance.csv"
)
print(
    "  figures/descriptor_feature_importance.png"
)