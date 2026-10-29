import os
import random

# ============================================================
# REPRODUCIBILITY
# ============================================================

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

SEED = 42

random.seed(SEED)

import numpy as np
np.random.seed(SEED)

import pandas as pd

from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

tf.random.set_seed(SEED)


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "DSI_D18A045_Seyhan_daily_flow_2015-2020.csv"
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
# 2. PARAMETERS
# ============================================================

TEST_MONTHS = 12
LOOKBACK = 12
FUTURE_MONTHS = 15


# ============================================================
# 3. LOAD DAILY DATA
# ============================================================

print("Loading daily river-flow data...")

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(df["date"])

df = (
    df
    .sort_values("date")
    .reset_index(drop=True)
)


# ============================================================
# 4. DAILY → MONTHLY AVERAGE
# ============================================================

monthly_df = (
    df
    .set_index("date")["river_flow_m3_s"]
    .resample("MS")
    .mean()
    .reset_index()
)

monthly_df.columns = [
    "date",
    "monthly_flow_m3_s"
]

monthly_df["monthly_flow_m3_s"] = (
    monthly_df["monthly_flow_m3_s"]
    .clip(lower=0)
)

print(
    "\nMonthly observations:",
    len(monthly_df)
)

print(
    "Monthly period:",
    monthly_df["date"].min(),
    "to",
    monthly_df["date"].max()
)


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

train_monthly = (
    monthly_df
    .iloc[:-TEST_MONTHS]
    .copy()
    .reset_index(drop=True)
)

test_monthly = (
    monthly_df
    .iloc[-TEST_MONTHS:]
    .copy()
    .reset_index(drop=True)
)

print(
    "\nTraining months:",
    len(train_monthly)
)

print(
    "Testing months :",
    len(test_monthly)
)

print(
    "\nTraining period:",
    train_monthly["date"].min(),
    "to",
    train_monthly["date"].max()
)

print(
    "Testing period :",
    test_monthly["date"].min(),
    "to",
    test_monthly["date"].max()
)


# ============================================================
# 6. CREATE LAG DATASET
# ============================================================

def create_lag_dataset(values, lookback):

    X = []
    y = []

    for i in range(
        lookback,
        len(values)
    ):

        X.append(
            values[
                i - lookback:i
            ]
        )

        y.append(
            values[i]
        )

    return (
        np.array(X),
        np.array(y)
    )


# ============================================================
# 7. RECURSIVE ML FORECASTING
# ============================================================

def recursive_ml_forecast(
    model,
    history,
    horizon,
    lookback
):

    history = list(
        np.asarray(history)
        .flatten()
    )

    predictions = []

    for _ in range(horizon):

        X_input = np.array(
            history[-lookback:]
        ).reshape(
            1,
            lookback
        )

        prediction = float(
            model.predict(
                X_input
            )[0]
        )

        # River flow cannot be negative.
        prediction = max(
            0.0,
            prediction
        )

        predictions.append(
            prediction
        )

        # Feed prediction back into history.
        history.append(
            prediction
        )

    return np.array(
        predictions
    )


# ============================================================
# 8. EVALUATION
# ============================================================

def evaluate_model(
    model_name,
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
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


results = []

all_predictions = {}


# ============================================================
# 9. LINEAR REGRESSION
# ============================================================

print("\n" + "=" * 70)
print("TRAINING LINEAR REGRESSION")
print("=" * 70)

train_values = (
    train_monthly[
        "monthly_flow_m3_s"
    ]
    .values
)

test_values = (
    test_monthly[
        "monthly_flow_m3_s"
    ]
    .values
)

X_train, y_train = (
    create_lag_dataset(
        train_values,
        LOOKBACK
    )
)

linear_model = LinearRegression()

linear_model.fit(
    X_train,
    y_train
)

linear_pred = recursive_ml_forecast(
    linear_model,
    train_values,
    TEST_MONTHS,
    LOOKBACK
)

results.append(
    evaluate_model(
        "Linear Regression",
        test_values,
        linear_pred
    )
)

all_predictions[
    "Linear Regression"
] = linear_pred


# ============================================================
# 10. RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)

rf_model = RandomForestRegressor(
    n_estimators=200,
    random_state=SEED,
    n_jobs=-1
)

rf_model.fit(
    X_train,
    y_train
)

rf_pred = recursive_ml_forecast(
    rf_model,
    train_values,
    TEST_MONTHS,
    LOOKBACK
)

results.append(
    evaluate_model(
        "Random Forest",
        test_values,
        rf_pred
    )
)

all_predictions[
    "Random Forest"
] = rf_pred


# ============================================================
# 11. XGBOOST
# ============================================================

print("\n" + "=" * 70)
print("TRAINING XGBOOST")
print("=" * 70)

xgb_model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=SEED,
    n_jobs=-1
)

xgb_model.fit(
    X_train,
    y_train
)

xgb_pred = recursive_ml_forecast(
    xgb_model,
    train_values,
    TEST_MONTHS,
    LOOKBACK
)

results.append(
    evaluate_model(
        "XGBoost",
        test_values,
        xgb_pred
    )
)

all_predictions[
    "XGBoost"
] = xgb_pred


# ============================================================
# 12. LSTM
# ============================================================

print("\n" + "=" * 70)
print("TRAINING LSTM")
print("=" * 70)


# ------------------------------------------------------------
# Scale using training data only
# ------------------------------------------------------------

scaler = MinMaxScaler()

train_scaled = scaler.fit_transform(
    train_values.reshape(-1, 1)
)


# ------------------------------------------------------------
# Create LSTM sequences
# ------------------------------------------------------------

X_lstm = []
y_lstm = []

for i in range(
    LOOKBACK,
    len(train_scaled)
):

    X_lstm.append(
        train_scaled[
            i - LOOKBACK:i,
            0
        ]
    )

    y_lstm.append(
        train_scaled[
            i,
            0
        ]
    )

X_lstm = np.array(
    X_lstm
)

y_lstm = np.array(
    y_lstm
)

X_lstm = X_lstm.reshape(
    X_lstm.shape[0],
    LOOKBACK,
    1
)

print(
    "\nLSTM training sequences:",
    len(X_lstm)
)


# ------------------------------------------------------------
# Build LSTM
# ------------------------------------------------------------

def build_lstm():

    model = Sequential([

        Input(
            shape=(LOOKBACK, 1)
        ),

        LSTM(
            32
        ),

        Dense(
            16,
            activation="relu"
        ),

        Dense(
            1
        )
    ])

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model


tf.random.set_seed(SEED)

lstm_model = build_lstm()

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

lstm_model.fit(
    X_lstm,
    y_lstm,
    validation_split=0.2,
    epochs=150,
    batch_size=4,
    callbacks=[
        early_stopping
    ],
    verbose=1,
    shuffle=False
)


# ------------------------------------------------------------
# Recursive LSTM forecasting
# ------------------------------------------------------------

current_sequence = (
    train_scaled[-LOOKBACK:]
    .reshape(
        1,
        LOOKBACK,
        1
    )
)

lstm_predictions_scaled = []

for _ in range(TEST_MONTHS):

    prediction_scaled = float(
        lstm_model.predict(
            current_sequence,
            verbose=0
        )[0][0]
    )

    lstm_predictions_scaled.append(
        prediction_scaled
    )

    current_sequence = np.append(
        current_sequence[
            0,
            1:,
            0
        ],
        prediction_scaled
    ).reshape(
        1,
        LOOKBACK,
        1
    )


lstm_pred = scaler.inverse_transform(
    np.array(
        lstm_predictions_scaled
    ).reshape(-1, 1)
).flatten()

# Physical constraint
lstm_pred = np.clip(
    lstm_pred,
    0,
    None
)

results.append(
    evaluate_model(
        "LSTM",
        test_values,
        lstm_pred
    )
)

all_predictions[
    "LSTM"
] = lstm_pred


# ============================================================
# 13. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
)


# ------------------------------------------------------------
# Ranking
# ------------------------------------------------------------

results_df["MAE_Rank"] = (
    results_df["MAE"]
    .rank(
        ascending=True,
        method="min"
    )
)

results_df["RMSE_Rank"] = (
    results_df["RMSE"]
    .rank(
        ascending=True,
        method="min"
    )
)

results_df["R2_Rank"] = (
    results_df["R2"]
    .rank(
        ascending=False,
        method="min"
    )
)

results_df["Overall_Score"] = (
    results_df["MAE_Rank"]
    + results_df["RMSE_Rank"]
    + results_df["R2_Rank"]
)

results_df = (
    results_df
    .sort_values(
        "Overall_Score"
    )
    .reset_index(drop=True)
)

results_df["Overall_Rank"] = (
    np.arange(
        len(results_df)
    ) + 1
)


# ============================================================
# 14. DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 75)
print("FAIR LONG-TERM FORECASTING MODEL COMPARISON")
print("=" * 75)

print(
    results_df[
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
# 15. BEST MODEL
# ============================================================

best_model_name = (
    results_df
    .iloc[0]["Model"]
)

best_model_row = (
    results_df
    .iloc[0]
)


print("\n" + "=" * 75)
print("BEST LONG-TERM MODEL")
print("=" * 75)

print(
    "Model :",
    best_model_name
)

print(
    "MAE   :",
    f"{best_model_row['MAE']:.4f}"
)

print(
    "RMSE  :",
    f"{best_model_row['RMSE']:.4f}"
)

print(
    "R2    :",
    f"{best_model_row['R2']:.4f}"
)


# ============================================================
# 16. BEST MODEL BY EACH METRIC
# ============================================================

best_mae = results_df.loc[
    results_df["MAE"].idxmin()
]

best_rmse = results_df.loc[
    results_df["RMSE"].idxmin()
]

best_r2 = results_df.loc[
    results_df["R2"].idxmax()
]

print("\n" + "=" * 75)
print("BEST MODEL BY INDIVIDUAL METRIC")
print("=" * 75)

print(
    f"Best MAE  : {best_mae['Model']} "
    f"({best_mae['MAE']:.4f})"
)

print(
    f"Best RMSE : {best_rmse['Model']} "
    f"({best_rmse['RMSE']:.4f})"
)

print(
    f"Best R2   : {best_r2['Model']} "
    f"({best_r2['R2']:.4f})"
)


# ============================================================
# 17. SAVE MODEL COMPARISON
# ============================================================

comparison_file = (
    RESULTS_DIR
    / "long_term_model_comparison.csv"
)

results_df.to_csv(
    comparison_file,
    index=False
)


# ============================================================
# 18. SAVE TEST PREDICTIONS
# ============================================================

test_prediction_df = pd.DataFrame({

    "date":
        test_monthly["date"].values,

    "actual_inflow_m3_s":
        test_values,

    "linear_regression":
        all_predictions[
            "Linear Regression"
        ],

    "random_forest":
        all_predictions[
            "Random Forest"
        ],

    "xgboost":
        all_predictions[
            "XGBoost"
        ],

    "lstm":
        all_predictions[
            "LSTM"
        ]
})


test_predictions_file = (
    RESULTS_DIR
    / "long_term_model_predictions.csv"
)

test_prediction_df.to_csv(
    test_predictions_file,
    index=False
)


# ============================================================
# 19. TRAIN BEST MODEL ON ALL 60 MONTHS
# ============================================================

print("\n" + "=" * 75)
print(
    "TRAINING BEST MODEL ON ALL AVAILABLE DATA"
)
print("=" * 75)

print(
    "Selected model:",
    best_model_name
)


all_values = (
    monthly_df[
        "monthly_flow_m3_s"
    ]
    .values
)


# ============================================================
# 20. FINAL MODEL TRAINING
# ============================================================

if best_model_name == "Linear Regression":

    X_all, y_all = create_lag_dataset(
        all_values,
        LOOKBACK
    )

    final_model = LinearRegression()

    final_model.fit(
        X_all,
        y_all
    )


elif best_model_name == "Random Forest":

    X_all, y_all = create_lag_dataset(
        all_values,
        LOOKBACK
    )

    final_model = RandomForestRegressor(
        n_estimators=200,
        random_state=SEED,
        n_jobs=-1
    )

    final_model.fit(
        X_all,
        y_all
    )


elif best_model_name == "XGBoost":

    X_all, y_all = create_lag_dataset(
        all_values,
        LOOKBACK
    )

    final_model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=SEED,
        n_jobs=-1
    )

    final_model.fit(
        X_all,
        y_all
    )


elif best_model_name == "LSTM":

    final_scaler = MinMaxScaler()

    all_scaled = final_scaler.fit_transform(
        all_values.reshape(-1, 1)
    )

    X_all_lstm = []
    y_all_lstm = []

    for i in range(
        LOOKBACK,
        len(all_scaled)
    ):

        X_all_lstm.append(
            all_scaled[
                i - LOOKBACK:i,
                0
            ]
        )

        y_all_lstm.append(
            all_scaled[
                i,
                0
            ]
        )

    X_all_lstm = np.array(
        X_all_lstm
    )

    y_all_lstm = np.array(
        y_all_lstm
    )

    X_all_lstm = X_all_lstm.reshape(
        X_all_lstm.shape[0],
        LOOKBACK,
        1
    )

    tf.random.set_seed(SEED)

    final_model = build_lstm()

    final_model.fit(
        X_all_lstm,
        y_all_lstm,
        epochs=50,
        batch_size=4,
        verbose=0,
        shuffle=False
    )


# ============================================================
# 21. GENERATE 15-MONTH FORECAST
# ============================================================

print(
    "\nGenerating 15-month forecast..."
)


if best_model_name == "LSTM":

    current_sequence = (
        all_scaled[-LOOKBACK:]
        .reshape(
            1,
            LOOKBACK,
            1
        )
    )

    future_scaled = []

    for _ in range(
        FUTURE_MONTHS
    ):

        prediction_scaled = float(
            final_model.predict(
                current_sequence,
                verbose=0
            )[0][0]
        )

        future_scaled.append(
            prediction_scaled
        )

        current_sequence = np.append(
            current_sequence[
                0,
                1:,
                0
            ],
            prediction_scaled
        ).reshape(
            1,
            LOOKBACK,
            1
        )

    future_predictions = (
        final_scaler
        .inverse_transform(
            np.array(
                future_scaled
            ).reshape(-1, 1)
        )
        .flatten()
    )

else:

    future_predictions = (
        recursive_ml_forecast(
            final_model,
            all_values,
            FUTURE_MONTHS,
            LOOKBACK
        )
    )


# Physical constraint
future_predictions = np.clip(
    future_predictions,
    0,
    None
)


# ============================================================
# 22. FUTURE DATES
# ============================================================

last_date = (
    monthly_df[
        "date"
    ].iloc[-1]
)

future_dates = pd.date_range(
    start=(
        last_date
        + pd.DateOffset(
            months=1
        )
    ),
    periods=FUTURE_MONTHS,
    freq="MS"
)


future_df = pd.DataFrame({

    "date":
        future_dates,

    "predicted_inflow_m3_s":
        future_predictions
})


# ============================================================
# 23. SAVE FINAL 15-MONTH FORECAST
# ============================================================

future_file = (
    RESULTS_DIR
    / "long_term_15month_forecast.csv"
)

future_df.to_csv(
    future_file,
    index=False
)


# ============================================================
# 24. SAVE BEST MODEL
# ============================================================

best_model_file = (
    RESULTS_DIR
    / "long_term_best_model.csv"
)

pd.DataFrame([{

    "best_model":
        best_model_name,

    "MAE":
        best_model_row["MAE"],

    "RMSE":
        best_model_row["RMSE"],

    "R2":
        best_model_row["R2"]

}]).to_csv(
    best_model_file,
    index=False
)


# ============================================================
# 25. DISPLAY 15-MONTH FORECAST
# ============================================================

print("\n")
print("=" * 75)
print("FINAL 15-MONTH FORECAST")
print("=" * 75)

print(
    "Forecasting model:",
    best_model_name
)

print()

print(
    future_df.to_string(
        index=False
    )
)


# ============================================================
# 26. FILES CREATED
# ============================================================

print("\n")
print("=" * 75)
print("FILES CREATED")
print("=" * 75)

print(
    "\n1. Model comparison:"
)

print(
    comparison_file
)

print(
    "\n2. Test predictions:"
)

print(
    test_predictions_file
)

print(
    "\n3. Best model:"
)

print(
    best_model_file
)

print(
    "\n4. Final 15-month forecast:"
)

print(
    future_file
)

print("\n")
print("=" * 75)
print(
    "LONG-TERM FORECASTING COMPLETED"
)
print("=" * 75)