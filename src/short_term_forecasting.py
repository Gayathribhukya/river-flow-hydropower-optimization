import os

# ============================================================
# REPRODUCIBILITY
# ============================================================

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import random
import numpy as np

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

import tensorflow as tf

tf.random.set_seed(SEED)


# ============================================================
# IMPORTS
# ============================================================

import pandas as pd

from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_FILE = (
    BASE_DIR
    / "data"
    / "short_term_train_dataset.csv"
)

TEST_FILE = (
    BASE_DIR
    / "data"
    / "short_term_test_dataset.csv"
)

RESULTS_DIR = (
    BASE_DIR
    / "results"
    / "forecasts"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("Loading forecasting datasets...")

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print("\nTraining samples:", len(train_df))
print("Testing samples :", len(test_df))


# ============================================================
# 3. FEATURES
# ============================================================

FEATURES = [
    "flow_lag_1",
    "flow_lag_2",
    "flow_lag_3",
    "flow_lag_4",
    "flow_lag_5",
    "flow_lag_6",
    "flow_lag_7"
]

X_train = train_df[FEATURES].values
X_test = test_df[FEATURES].values


print("\nFeatures:")
print(FEATURES)

print("\nForecast horizons:")
print("1 to 10 days")


# ============================================================
# 4. EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model_name,
    horizon,
    y_true,
    y_pred
):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    return {
        "Horizon_Days": horizon,
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


# ============================================================
# 5. MODEL FUNCTION
# ============================================================

def create_models():

    models = {

        "Linear Regression":
            LinearRegression(),

        "Polynomial Regression":
            Pipeline([
                (
                    "poly",
                    PolynomialFeatures(
                        degree=2,
                        include_bias=False
                    )
                ),
                (
                    "regressor",
                    LinearRegression()
                )
            ]),

        "KNN Regression":
            Pipeline([
                (
                    "scaler",
                    StandardScaler()
                ),
                (
                    "knn",
                    KNeighborsRegressor(
                        n_neighbors=5
                    )
                )
            ]),

        "Random Forest":
            RandomForestRegressor(
                n_estimators=200,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=SEED,
                n_jobs=-1
            ),

        "XGBoost":
            XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                random_state=SEED,
                n_jobs=-1
            )
    }

    return models


# ============================================================
# 6. DNN FUNCTION
# ============================================================

def create_dnn():

    model = Sequential([

        Input(
            shape=(len(FEATURES),)
        ),

        Dense(
            64,
            activation="relu"
        ),

        Dropout(
            0.2
        ),

        Dense(
            32,
            activation="relu"
        ),

        Dense(
            16,
            activation="relu"
        ),

        Dense(
            1,
            activation="linear"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model


# ============================================================
# 7. STORAGE
# ============================================================

all_results = []

all_predictions = []


# ============================================================
# 8. LOOP THROUGH 1–10 DAY HORIZONS
# ============================================================

for horizon in range(1, 11):

    target = f"flow_t_plus_{horizon}"

    print("\n")
    print("=" * 80)
    print(
        f"FORECAST HORIZON: {horizon} DAY(S)"
    )
    print("=" * 80)

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    y_train = train_df[target].values
    y_test = test_df[target].values

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    models = create_models()

    horizon_predictions = {
        "date": test_df["date"].values,
        "horizon_days": horizon,
        "actual_flow_m3_s": y_test
    }

    # --------------------------------------------------------
    # Traditional / ML Models
    # --------------------------------------------------------

    for model_name, model in models.items():

        print(
            f"\nTraining {model_name}..."
        )

        model.fit(
            X_train,
            y_train
        )

        prediction = model.predict(
            X_test
        )

        # Save metrics

        result = evaluate_model(
            model_name,
            horizon,
            y_test,
            prediction
        )

        all_results.append(
            result
        )

        # Save predictions

        column_name = (
            model_name
            .lower()
            .replace(" ", "_")
            .replace("/", "_")
        )

        horizon_predictions[
            column_name
        ] = prediction

    # --------------------------------------------------------
    # DNN / ANN
    # --------------------------------------------------------

    print("\nTraining DNN / ANN...")

    # Scale input features

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # Reset TensorFlow seed before every DNN

    tf.random.set_seed(
        SEED + horizon
    )

    dnn_model = create_dnn()

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True
    )

    dnn_model.fit(
        X_train_scaled,
        y_train,
        validation_split=0.1,
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=0,
        shuffle=False
    )

    dnn_prediction = (
        dnn_model
        .predict(
            X_test_scaled,
            verbose=0
        )
        .flatten()
    )

    # Metrics

    dnn_result = evaluate_model(
        "DNN / ANN",
        horizon,
        y_test,
        dnn_prediction
    )

    all_results.append(
        dnn_result
    )

    # Predictions

    horizon_predictions[
        "dnn_ann"
    ] = dnn_prediction

    # --------------------------------------------------------
    # Save horizon predictions
    # --------------------------------------------------------

    all_predictions.append(
        pd.DataFrame(
            horizon_predictions
        )
    )

    # --------------------------------------------------------
    # Display horizon results
    # --------------------------------------------------------

    horizon_results = pd.DataFrame(
        [
            r
            for r in all_results
            if r["Horizon_Days"] == horizon
        ]
    )

    print("\nResults:")

    print(
        horizon_results[
            [
                "Model",
                "MAE",
                "RMSE",
                "R2"
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )


# ============================================================
# 9. COMPLETE RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    all_results
)


print("\n")
print("=" * 80)
print("COMPLETE 1–10 DAY FORECASTING RESULTS")
print("=" * 80)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 10. BEST MODEL FOR EACH HORIZON
# ============================================================

best_models = []

for horizon in range(1, 11):

    horizon_df = results_df[
        results_df["Horizon_Days"] == horizon
    ].copy()

    # Rank using all three metrics

    horizon_df["MAE_Rank"] = (
        horizon_df["MAE"]
        .rank(
            ascending=True,
            method="min"
        )
    )

    horizon_df["RMSE_Rank"] = (
        horizon_df["RMSE"]
        .rank(
            ascending=True,
            method="min"
        )
    )

    horizon_df["R2_Rank"] = (
        horizon_df["R2"]
        .rank(
            ascending=False,
            method="min"
        )
    )

    horizon_df["Overall_Score"] = (
        horizon_df["MAE_Rank"]
        + horizon_df["RMSE_Rank"]
        + horizon_df["R2_Rank"]
    )

    best = horizon_df.loc[
        horizon_df["Overall_Score"].idxmin()
    ]

    best_models.append({
        "Horizon_Days":
            horizon,

        "Best_Model":
            best["Model"],

        "MAE":
            best["MAE"],

        "RMSE":
            best["RMSE"],

        "R2":
            best["R2"],

        "Overall_Score":
            best["Overall_Score"]
    })


best_models_df = pd.DataFrame(
    best_models
)


# ============================================================
# 11. DISPLAY BEST MODELS
# ============================================================

print("\n")
print("=" * 80)
print("BEST MODEL FOR EACH FORECAST HORIZON")
print("=" * 80)

print(
    best_models_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 12. MODEL PERFORMANCE ACROSS ALL HORIZONS
# ============================================================

model_summary = (
    results_df
    .groupby("Model")
    .agg({
        "MAE": "mean",
        "RMSE": "mean",
        "R2": "mean"
    })
    .reset_index()
)


model_summary["MAE_Rank"] = (
    model_summary["MAE"]
    .rank(
        ascending=True,
        method="min"
    )
)

model_summary["RMSE_Rank"] = (
    model_summary["RMSE"]
    .rank(
        ascending=True,
        method="min"
    )
)

model_summary["R2_Rank"] = (
    model_summary["R2"]
    .rank(
        ascending=False,
        method="min"
    )
)

model_summary["Overall_Score"] = (
    model_summary["MAE_Rank"]
    + model_summary["RMSE_Rank"]
    + model_summary["R2_Rank"]
)

model_summary = (
    model_summary
    .sort_values("Overall_Score")
    .reset_index(drop=True)
)

model_summary["Overall_Rank"] = (
    np.arange(
        len(model_summary)
    ) + 1
)


# ============================================================
# 13. DISPLAY OVERALL MODEL COMPARISON
# ============================================================

print("\n")
print("=" * 80)
print("OVERALL MODEL PERFORMANCE ACROSS 1–10 DAYS")
print("=" * 80)

print(
    model_summary[
        [
            "Overall_Rank",
            "Model",
            "MAE",
            "RMSE",
            "R2",
            "Overall_Score"
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 14. SAVE COMPLETE METRICS
# ============================================================

comparison_file = (
    RESULTS_DIR
    / "short_term_10day_model_comparison.csv"
)

results_df.to_csv(
    comparison_file,
    index=False
)


# ============================================================
# 15. SAVE BEST MODELS
# ============================================================

best_models_file = (
    RESULTS_DIR
    / "short_term_best_models.csv"
)

best_models_df.to_csv(
    best_models_file,
    index=False
)


# ============================================================
# 16. SAVE OVERALL MODEL SUMMARY
# ============================================================

summary_file = (
    RESULTS_DIR
    / "short_term_model_summary.csv"
)

model_summary.to_csv(
    summary_file,
    index=False
)


# ============================================================
# 17. SAVE ALL PREDICTIONS
# ============================================================

predictions_df = pd.concat(
    all_predictions,
    ignore_index=True
)

predictions_file = (
    RESULTS_DIR
    / "short_term_10day_predictions.csv"
)

predictions_df.to_csv(
    predictions_file,
    index=False
)


# ============================================================
# 18. FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 80)
print("FILES CREATED")
print("=" * 80)

print(
    "\n1. Complete 1–10 day metrics:"
)

print(
    comparison_file
)

print(
    "\n2. Best model for each horizon:"
)

print(
    best_models_file
)

print(
    "\n3. Overall model summary:"
)

print(
    summary_file
)

print(
    "\n4. All model predictions:"
)

print(
    predictions_file
)

print("\n")
print("=" * 80)
print("SHORT-TERM 1–10 DAY FORECASTING COMPLETED")
print("=" * 80)