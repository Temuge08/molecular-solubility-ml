#!/usr/bin/env python3

from pathlib import Path

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,)

# ============================================================
# Paths
# ============================================================

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_FILE = (
    DATA_DIR / "esol_cluster_split_manifest.csv"
)

FEATURE_FILES = {
    "descriptors":
        DATA_DIR / "esol_descriptors.csv",

    "morgan":
        DATA_DIR / "esol_morgan.csv",

    "combined":
        DATA_DIR / "esol_combined_features.csv",
}


# ============================================================
# Random Forest configuration
#
# Keep EXACTLY the same model for all representations.
# ============================================================
RF_PARAMS = {
    "n_estimators": 500,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": 1.0,
    "random_state": 42,
    "n_jobs": -1,
}

# ============================================================
# Metric function
# ============================================================

def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(
        y_true,
        y_pred,
    )
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(
        y_true,
        y_pred,
    )
    r2 = r2_score(
        y_true,
        y_pred,
    )
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }

# ============================================================
# Load split manifest
# ============================================================

manifest = pd.read_csv(MANIFEST_FILE)
print("=" * 70)
print("DATA SPLIT")
print("=" * 70)
print(manifest["split"].value_counts())

# ============================================================
# Basic validation
# ============================================================
required_splits = {"train", "validation", "test"}
actual_splits = set(manifest["split"].unique())

if required_splits != actual_splits:
    raise RuntimeError(
        f"Unexpected splits: {actual_splits}"
    )

# ============================================================
# Baseline model
#
# Predict mean TRAINING logS for every validation molecule.
#
# This tells us whether ML is actually learning useful signal.
# ============================================================

train_manifest = manifest[manifest["split"] == "train"]
validation_manifest = manifest[manifest["split"] == "validation"]
train_mean = (train_manifest["measured_logS"].mean())

baseline_predictions = np.full(len(validation_manifest), train_mean)
baseline_metrics = calculate_metrics(validation_manifest["measured_logS"],baseline_predictions)

print()
print("=" * 70)
print("MEAN BASELINE")
print("=" * 70)
print(
    f"Train mean logS: {train_mean:.4f}"
)
print(
    f"Validation RMSE: "
    f"{baseline_metrics['rmse']:.4f}"
)
print(
    f"Validation MAE:  "
    f"{baseline_metrics['mae']:.4f}"
)
print(
    f"Validation R2:   "
    f"{baseline_metrics['r2']:.4f}"
)

# ============================================================
# Model results
# ============================================================

results = []

# ============================================================
# Train one model for each representation
# ============================================================

for representation, feature_file in FEATURE_FILES.items():
    print()
    print("=" * 70)
    print(
        f"REPRESENTATION: {representation.upper()}"
    )
    print("=" * 70)
    features = pd.read_csv(
        feature_file
    )

    # --------------------------------------------------------
    # Identify feature columns
    # --------------------------------------------------------

    if representation == "descriptors":
        feature_columns = [
            col
            for col in features.columns
            if col.startswith("desc_")
        ]

    elif representation == "morgan":
        feature_columns = [
            col
            for col in features.columns
            if col.startswith("fp_")
        ]

    elif representation == "combined":
        feature_columns = [
            col
            for col in features.columns
            if (
                col.startswith("desc_")
                or col.startswith("fp_")
            )
        ]

    else:
        raise ValueError(
            f"Unknown representation: {representation}"
        )

    print(
        f"Number of features: "
        f"{len(feature_columns)}"
    )


    # --------------------------------------------------------
    # Merge split labels onto feature table
    #
    # Validate one-to-one matching because each canonical
    # structure should occur exactly once.
    # --------------------------------------------------------

    split_info = manifest[
        [
            "canonical_smiles",
            "split",
        ]
    ]
    data = features.merge(
        split_info,
        on="canonical_smiles",
        how="inner",
        validate="one_to_one",
    )

    if len(data) != len(manifest):

        raise RuntimeError(
            f"{representation}: merge produced "
            f"{len(data)} rows instead of "
            f"{len(manifest)}."
        )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    train = data[data["split"] == "train"]
    validation = data[data["split"] == "validation"]

    X_train = train[feature_columns].to_numpy()
    y_train = train["measured_logS"].to_numpy()

    X_val = validation[feature_columns].to_numpy()
    y_val = validation["measured_logS"].to_numpy()

    print(
        f"Train shape:      {X_train.shape}"
    )
    print(
        f"Validation shape: {X_val.shape}"
    )

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------
    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(X_train, y_train)

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    train_predictions = model.predict(X_train)
    validation_predictions = model.predict(X_val)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    train_metrics = calculate_metrics(y_train, train_predictions)
    validation_metrics = calculate_metrics(y_val, validation_predictions)

    print()
    print("Training:")
    print(
        f"  RMSE = "
        f"{train_metrics['rmse']:.4f}"
    )
    print(
        f"  MAE  = "
        f"{train_metrics['mae']:.4f}"
    )
    print(
        f"  R2   = "
        f"{train_metrics['r2']:.4f}"
    )

    print()
    print("Validation:")
    print(
        f"  RMSE = "
        f"{validation_metrics['rmse']:.4f}"
    )
    print(
        f"  MAE  = "
        f"{validation_metrics['mae']:.4f}"
    )
    print(
        f"  R2   = "
        f"{validation_metrics['r2']:.4f}"
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------
    model_path = (MODEL_DIR / f"rf_{representation}.joblib")
    joblib.dump(model,model_path)

    # --------------------------------------------------------
    # Save validation predictions
    # --------------------------------------------------------

    prediction_output = validation[["compound_ids", "canonical_smiles", "measured_logS"]].copy()
    prediction_output["predicted_logS"] = validation_predictions
    prediction_output["residual"] = (prediction_output["measured_logS"] - prediction_output["predicted_logS"])
    
    prediction_output["absolute_error"] = (prediction_output["residual"].abs())
    prediction_output.to_csv(
        REPORT_DIR
        / (
            f"validation_predictions_"
            f"{representation}.csv"
        ),
        index=False,
    )

    # --------------------------------------------------------
    # Store summary
    # --------------------------------------------------------

    results.append(
        {
            "representation":
                representation,
            "n_features":
                len(feature_columns),
            "train_rmse":
                train_metrics["rmse"],
            "train_mae":
                train_metrics["mae"],
            "train_r2":
                train_metrics["r2"],
            "validation_rmse":
                validation_metrics["rmse"],
            "validation_mae":
                validation_metrics["mae"],
            "validation_r2":
                validation_metrics["r2"]
                })


# ============================================================
# Results table
# ============================================================

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("validation_rmse")

print()
print("=" * 70)
print("REPRESENTATION COMPARISON")
print("=" * 70)
print(results_df.to_string(index=False))

results_df.to_csv(
    REPORT_DIR
    / "rf_representation_comparison.csv",
    index=False,
)

# ============================================================
# Save model configuration
# ============================================================

with open(REPORT_DIR / "rf_config.json", "w",) as f:
    json.dump(RF_PARAMS, f, indent=4)

print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)
print()
print("Important: the TEST set has not been used.")
print()
print("Saved:")
print(
    "  reports/"
    "rf_representation_comparison.csv"
)
print(
    "  reports/"
    "validation_predictions_*.csv"
)
print(
    "  models/rf_*.joblib"
)