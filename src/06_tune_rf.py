#!/usr/bin/env python3

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid
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


MANIFEST_FILE = (
    DATA_DIR / "esol_cluster_split_manifest.csv"
)


FEATURE_FILES = {
    "descriptors":
        DATA_DIR / "esol_descriptors.csv",

    "combined":
        DATA_DIR / "esol_combined_features.csv",
}


# ============================================================
# Hyperparameter search space
#
# We deliberately keep n_estimators fixed.
#
# We mainly want to control tree complexity and the number of
# features considered at each split.
# ============================================================

PARAM_GRID = {
    "max_depth": [
        None,
        10,
        20,
    ],

    "min_samples_split": [
        2,
        5,
    ],

    "min_samples_leaf": [
        1,
        2,
        4,
    ],

    "max_features": [
        1.0,
        0.5,
        "sqrt",
    ],
}


N_ESTIMATORS = 500
RANDOM_STATE = 42


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )
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

manifest = pd.read_csv(
    MANIFEST_FILE
)


print("=" * 70)
print("RANDOM FOREST HYPERPARAMETER TUNING")
print("=" * 70)

print()
print("Split sizes:")

print(
    manifest["split"]
    .value_counts()
)


# ============================================================
# Safety check:
# We will NEVER create X_test or y_test in this script.
# ============================================================

if "test" not in set(manifest["split"]):
    raise RuntimeError(
        "Test split missing from manifest."
    )


# ============================================================
# Tune each representation
# ============================================================

all_results = []
best_models = {}


for representation, feature_file in FEATURE_FILES.items():

    print()
    print("=" * 70)
    print(
        f"TUNING: {representation.upper()}"
    )
    print("=" * 70)

    features = pd.read_csv(
        feature_file
    )


    # --------------------------------------------------------
    # Select representation-specific feature columns
    # --------------------------------------------------------

    if representation == "descriptors":

        feature_columns = [
            c
            for c in features.columns
            if c.startswith("desc_")
        ]

    elif representation == "combined":

        feature_columns = [
            c
            for c in features.columns
            if (
                c.startswith("desc_")
                or c.startswith("fp_")
            )
        ]

    else:

        raise ValueError(
            representation
        )


    # --------------------------------------------------------
    # Merge split labels
    # --------------------------------------------------------

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


    if len(data) != len(manifest):
        raise RuntimeError(
            "Feature/manifest merge mismatch."
        )


    train = data[
        data["split"] == "train"
    ]

    validation = data[
        data["split"] == "validation"
    ]


    X_train = train[
        feature_columns
    ].to_numpy()

    y_train = train[
        "measured_logS"
    ].to_numpy()


    X_val = validation[
        feature_columns
    ].to_numpy()

    y_val = validation[
        "measured_logS"
    ].to_numpy()


    print(
        f"Features: {len(feature_columns)}"
    )

    print(
        f"Train:      {X_train.shape}"
    )

    print(
        f"Validation: {X_val.shape}"
    )


    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    parameter_sets = list(
        ParameterGrid(PARAM_GRID)
    )

    print(
        f"Configurations: {len(parameter_sets)}"
    )


    best_rmse = np.inf
    best_model = None
    best_params = None
    best_metrics = None
    best_predictions = None


    for i, params in enumerate(
        parameter_sets,
        start=1,
    ):

        model = RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **params,
        )


        model.fit(
            X_train,
            y_train,
        )


        train_pred = model.predict(
            X_train
        )

        val_pred = model.predict(
            X_val
        )


        train_metrics = calculate_metrics(
            y_train,
            train_pred,
        )

        val_metrics = calculate_metrics(
            y_val,
            val_pred,
        )


        result = {
            "representation":
                representation,

            "max_depth":
                params["max_depth"],

            "min_samples_split":
                params["min_samples_split"],

            "min_samples_leaf":
                params["min_samples_leaf"],

            "max_features":
                params["max_features"],

            "train_rmse":
                train_metrics["rmse"],

            "train_mae":
                train_metrics["mae"],

            "train_r2":
                train_metrics["r2"],

            "validation_rmse":
                val_metrics["rmse"],

            "validation_mae":
                val_metrics["mae"],

            "validation_r2":
                val_metrics["r2"],
        }


        all_results.append(
            result
        )


        if (
            val_metrics["rmse"]
            < best_rmse
        ):

            best_rmse = (
                val_metrics["rmse"]
            )

            best_model = model
            best_params = params.copy()
            best_metrics = val_metrics
            best_predictions = val_pred.copy()


        if (
            i % 10 == 0
            or i == len(parameter_sets)
        ):

            print(
                f"  completed "
                f"{i}/{len(parameter_sets)}"
            )


    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    joblib.dump(
        best_model,
        MODEL_DIR
        / f"rf_tuned_{representation}.joblib",
    )


    # --------------------------------------------------------
    # Save best validation predictions
    # --------------------------------------------------------

    prediction_df = validation[
        [
            "compound_ids",
            "canonical_smiles",
            "measured_logS",
        ]
    ].copy()


    prediction_df[
        "predicted_logS"
    ] = best_predictions


    prediction_df[
        "residual"
    ] = (
        prediction_df["measured_logS"]
        - prediction_df["predicted_logS"]
    )


    prediction_df[
        "absolute_error"
    ] = (
        prediction_df["residual"]
        .abs()
    )


    prediction_df.to_csv(
        REPORT_DIR
        / f"tuned_validation_predictions_{representation}.csv",
        index=False,
    )


    # --------------------------------------------------------
    # Report best result
    # --------------------------------------------------------

    print()
    print("Best parameters:")

    for key, value in best_params.items():
        print(
            f"  {key}: {value}"
        )


    print()
    print("Best validation metrics:")

    print(
        f"  RMSE = "
        f"{best_metrics['rmse']:.4f}"
    )

    print(
        f"  MAE  = "
        f"{best_metrics['mae']:.4f}"
    )

    print(
        f"  R2   = "
        f"{best_metrics['r2']:.4f}"
    )


    best_models[
        representation
    ] = {
        "params":
            best_params,

        "metrics":
            best_metrics,
    }


# ============================================================
# Save all search results
# ============================================================

results_df = pd.DataFrame(
    all_results
)


results_df = results_df.sort_values(
    [
        "representation",
        "validation_rmse",
    ]
)


results_df.to_csv(
    REPORT_DIR
    / "rf_hyperparameter_search.csv",
    index=False,
)


# ============================================================
# Best-model summary
# ============================================================

summary_rows = []


for representation, info in best_models.items():

    row = {
        "representation":
            representation,

        "validation_rmse":
            info["metrics"]["rmse"],

        "validation_mae":
            info["metrics"]["mae"],

        "validation_r2":
            info["metrics"]["r2"],
    }

    row.update(
        info["params"]
    )

    summary_rows.append(row)


summary_df = pd.DataFrame(
    summary_rows
).sort_values(
    "validation_rmse"
)


summary_df.to_csv(
    REPORT_DIR
    / "rf_tuned_best_models.csv",
    index=False,
)


print()
print("=" * 70)
print("BEST MODEL COMPARISON")
print("=" * 70)

print(
    summary_df.to_string(
        index=False
    )
)


# ============================================================
# Save configuration
# ============================================================

config = {
    "n_estimators":
        N_ESTIMATORS,

    "random_state":
        RANDOM_STATE,

    "parameter_grid":
        PARAM_GRID,
}


with open(
    REPORT_DIR
    / "rf_tuning_config.json",
    "w",
) as f:

    json.dump(
        config,
        f,
        indent=4,
    )


print()
print("=" * 70)
print("TUNING COMPLETE")
print("=" * 70)

print()
print(
    "TEST SET WAS NOT USED."
)