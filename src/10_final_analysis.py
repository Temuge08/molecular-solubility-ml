#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

DATA_DIR = Path("data/processed")
REPORT_DIR = Path("reports")
FIGURE_DIR = Path("figures")

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


PREDICTION_FILE = REPORT_DIR / "final_test_predictions.csv"
METRIC_FILE = REPORT_DIR / "final_test_metrics.csv"
IMPORTANCE_FILE = REPORT_DIR / "descriptor_feature_importance.csv"
MANIFEST_FILE = DATA_DIR / "esol_cluster_split_manifest.csv"


# ============================================================
# Load
# ============================================================

pred = pd.read_csv(PREDICTION_FILE)
metrics = pd.read_csv(METRIC_FILE)
importance = pd.read_csv(IMPORTANCE_FILE)
manifest = pd.read_csv(MANIFEST_FILE)


print("=" * 70)
print("FINAL PORTFOLIO ANALYSIS")
print("=" * 70)

print()
print(metrics.to_string(index=False))


# ============================================================
# 1. Measured vs predicted
# ============================================================

plt.figure(figsize=(6, 6))

plt.scatter(
    pred["measured_logS"],
    pred["predicted_logS"],
    alpha=0.7,
)

minimum = min(
    pred["measured_logS"].min(),
    pred["predicted_logS"].min(),
)

maximum = max(
    pred["measured_logS"].max(),
    pred["predicted_logS"].max(),
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--",
)

plt.xlabel("Measured logS")
plt.ylabel("Predicted logS")
plt.title("Held-out Test Set: Measured vs Predicted Solubility")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "final_test_measured_vs_predicted.png",
    dpi=300,
)

plt.close()


# ============================================================
# 2. Residual distribution
# ============================================================

plt.figure(figsize=(7, 5))

plt.hist(
    pred["residual"],
    bins=20,
)

plt.axvline(
    0,
    linestyle="--",
)

plt.xlabel("Residual (measured - predicted logS)")
plt.ylabel("Number of molecules")
plt.title("Held-out Test Residual Distribution")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "final_test_residual_distribution.png",
    dpi=300,
)

plt.close()


# ============================================================
# 3. Absolute error distribution
# ============================================================

plt.figure(figsize=(7, 5))

plt.hist(
    pred["absolute_error"],
    bins=20,
)

plt.xlabel("Absolute prediction error (logS)")
plt.ylabel("Number of molecules")
plt.title("Held-out Test Absolute Error Distribution")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "final_test_absolute_error.png",
    dpi=300,
)

plt.close()


# ============================================================
# 4. Feature importance
# ============================================================

plot_importance = (
    importance
    .sort_values(
        "importance_mean",
        ascending=True,
    )
)

plt.figure(figsize=(8, 5))

plt.barh(
    plot_importance["feature"],
    plot_importance["importance_mean"],
)

plt.xlabel("Permutation importance")
plt.ylabel("RDKit descriptor")
plt.title("Descriptor Importance for Solubility Prediction")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "final_descriptor_importance.png",
    dpi=300,
)

plt.close()


# ============================================================
# 5. Target distribution by split
# ============================================================

for split_name in [
    "train",
    "validation",
    "test",
]:

    subset = manifest[
        manifest["split"] == split_name
    ]

    plt.figure(figsize=(7, 5))

    plt.hist(
        subset["measured_logS"],
        bins=20,
    )

    plt.xlabel("Measured logS")
    plt.ylabel("Number of molecules")

    plt.title(
        f"{split_name.capitalize()} Set Solubility Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / f"{split_name}_target_distribution.png",
        dpi=300,
    )

    plt.close()


# ============================================================
# 6. Worst test predictions
# ============================================================

worst = (
    pred
    .sort_values(
        "absolute_error",
        ascending=False,
    )
    .head(20)
)

worst.to_csv(
    REPORT_DIR / "final_worst_test_predictions.csv",
    index=False,
)


# ============================================================
# Summary
# ============================================================

rf_row = metrics[
    metrics["model"]
    == "descriptor_random_forest"
].iloc[0]

print()
print("=" * 70)
print("FINAL MODEL SUMMARY")
print("=" * 70)

print(f"Test molecules: {len(pred)}")
print(f"Test RMSE:      {rf_row['rmse']:.4f}")
print(f"Test MAE:       {rf_row['mae']:.4f}")
print(f"Test R2:        {rf_row['r2']:.4f}")

print()
print("Saved final portfolio figures to figures/")
print("Saved worst predictions to:")
print("  reports/final_worst_test_predictions.csv")