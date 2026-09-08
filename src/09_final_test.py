#!/usr/bin/env python3

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# Paths
# ============================================================

DATA_DIR = Path("data/processed")
REPORT_DIR = Path("reports")
MODEL_DIR = Path("models")

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


FEATURE_FILE = (
    DATA_DIR / "esol_descriptors.csv"
)

MANIFEST_FILE = (
    DATA_DIR / "esol_cluster_split_manifest.csv"
)


# ============================================================
# FINAL MODEL CONFIGURATION
#
# Frozen before looking at test performance.
# ============================================================

FINAL_PARAMS = {
    "n_estimators": 500,
    "max_depth": 20,
    "max_features": "sqrt",
    "min_samples_leaf": 1,
    "min_samples_split": 2,
    "random_state": 42,
    "n_jobs": -1,
}


# ============================================================
# Metrics
# ============================================================

def metrics(y_true, y_pred):

    return {
        "rmse":
            np.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred,
                )
            ),

        "mae":
            mean_absolute_error(
                y_true,
                y_pred,
            ),

        "r2":
            r2_score(
                y_true,
                y_pred,
            ),
    }


# ============================================================
# Load data
# ============================================================

features = pd.read_csv(
    FEATURE_FILE
)

manifest = pd.read_csv(
    MANIFEST_FILE
)


descriptor_columns = [
    c
    for c in features.columns
    if c.startswith("desc_")
]


data = features.merge(
    manifest[
        [
            "canonical_smiles",
            "cluster_id",
            "split",
        ]
    ],
    on="canonical_smiles",
    how="inner",
    validate="one_to_one",
)


# ============================================================
# Final development data:
# train + validation
# ============================================================

development = data[
    data["split"].isin(
        [
            "train",
            "validation",
        ]
    )
].copy()


test = data[
    data["split"] == "test"
].copy()


X_dev = development[
    descriptor_columns
]

y_dev = development[
    "measured_logS"
]


X_test = test[
    descriptor_columns
]

y_test = test[
    "measured_logS"
]


print("=" * 70)
print("FINAL MODEL")
print("=" * 70)

print(
    f"Development molecules: {len(development)}"
)

print(
    f"Test molecules:        {len(test)}"
)

print(
    f"Features:              {len(descriptor_columns)}"
)


# ============================================================
# Train final model
# ============================================================

model = RandomForestRegressor(
    **FINAL_PARAMS
)

model.fit(
    X_dev,
    y_dev,
)


# ============================================================
# FINAL TEST PREDICTIONS
# ============================================================

test_pred = model.predict(
    X_test
)

test_metrics = metrics(
    y_test,
    test_pred,
)


# ============================================================
# Mean baseline
#
# Uses DEVELOPMENT mean only.
# ============================================================

development_mean = y_dev.mean()

baseline_pred = np.full(
    len(y_test),
    development_mean,
)

baseline_metrics = metrics(
    y_test,
    baseline_pred,
)


# ============================================================
# Report
# ============================================================

print()
print("=" * 70)
print("TEST RESULTS")
print("=" * 70)

print()
print("Mean baseline:")

print(
    f"  RMSE = {baseline_metrics['rmse']:.4f}"
)

print(
    f"  MAE  = {baseline_metrics['mae']:.4f}"
)

print(
    f"  R2   = {baseline_metrics['r2']:.4f}"
)


print()
print("Descriptor Random Forest:")

print(
    f"  RMSE = {test_metrics['rmse']:.4f}"
)

print(
    f"  MAE  = {test_metrics['mae']:.4f}"
)

print(
    f"  R2   = {test_metrics['r2']:.4f}"
)


# ============================================================
# Save model
# ============================================================

joblib.dump(
    model,
    MODEL_DIR
    / "final_esol_descriptor_rf.joblib",
)


# ============================================================
# Save predictions
# ============================================================

predictions = test[
    [
        "compound_ids",
        "canonical_smiles",
        "cluster_id",
        "measured_logS",
    ]
].copy()


predictions[
    "predicted_logS"
] = test_pred


predictions[
    "residual"
] = (
    predictions["measured_logS"]
    - predictions["predicted_logS"]
)


predictions[
    "absolute_error"
] = (
    predictions["residual"]
    .abs()
)


predictions.to_csv(
    REPORT_DIR
    / "final_test_predictions.csv",
    index=False,
)


# ============================================================
# Save final metrics
# ============================================================

metric_df = pd.DataFrame(
    [
        {
            "model":
                "mean_baseline",

            **baseline_metrics,
        },
        {
            "model":
                "descriptor_random_forest",

            **test_metrics,
        },
    ]
)

metric_df.to_csv(REPORT_DIR / "final_test_metrics.csv", index=False)
with open(REPORT_DIR / "final_model_config.json", "w") as f:
    json.dump(FINAL_PARAMS, f, indent=4)

print()
print("Saved:")
print(
    "  models/final_esol_descriptor_rf.joblib"
)
print(
    "  reports/final_test_predictions.csv"
)
print(
    "  reports/final_test_metrics.csv"
)
print()
print(
    "The held-out test set has now been evaluated."
)
print(
    "Do not modify the model based on these test results."
)