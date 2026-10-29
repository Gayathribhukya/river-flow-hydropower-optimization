import pandas as pd
from pathlib import Path

# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "DSI_D18A045_Seyhan_daily_flow_2015-2020.csv"
)

OUTPUT_DIR = BASE_DIR / "data"

OUTPUT_FILE = (
    OUTPUT_DIR / "short_term_forecasting_dataset.csv"
)

TRAIN_FILE = OUTPUT_DIR / "short_term_train_dataset.csv"
TEST_FILE = OUTPUT_DIR / "short_term_test_dataset.csv"


# ============================================================
# 2. LOAD DATA
# ============================================================

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(df["date"])

# Sort chronologically
df = df.sort_values("date").reset_index(drop=True)

print("\nOriginal dataset:")
print(df.head())

print("\nNumber of records:", len(df))


# ============================================================
# 3. CHECK DATA
# ============================================================

print("\nMissing values:")
print(df.isnull().sum())

print("\nDate range:")
print(df["date"].min(), "to", df["date"].max())


# ============================================================
# 4. CREATE LAG FEATURES
# ============================================================

print("\nCreating 7-day lag features...")

for lag in range(1, 8):
    df[f"flow_lag_{lag}"] = (
        df["river_flow_m3_s"].shift(lag)
    )


# ============================================================
# 5. CREATE 10-DAY FORECAST TARGETS
# ============================================================

print("\nCreating 10-day future flow targets...")

for day in range(1, 11):
    df[f"flow_t_plus_{day}"] = (
        df["river_flow_m3_s"].shift(-day)
    )


# ============================================================
# 6. KEEP REQUIRED COLUMNS
# ============================================================

feature_columns = [
    "date",
    "flow_lag_1",
    "flow_lag_2",
    "flow_lag_3",
    "flow_lag_4",
    "flow_lag_5",
    "flow_lag_6",
    "flow_lag_7",
]

target_columns = [
    f"flow_t_plus_{day}"
    for day in range(1, 11)
]

final_columns = (
    feature_columns
    + target_columns
)

final_df = df[final_columns].copy()


# ============================================================
# 7. REMOVE MISSING VALUES
# ============================================================

# Missing values occur because:
# - the first 7 records do not have enough lag history
# - the last 10 records do not have future targets

final_df = (
    final_df
    .dropna()
    .reset_index(drop=True)
)


# ============================================================
# 8. CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================

# No random shuffling because this is time-series data.

train_df = final_df[
    final_df["date"] < "2020-01-01"
].copy()

test_df = final_df[
    final_df["date"] >= "2020-01-01"
].copy()


# ============================================================
# 9. SAVE DATASETS
# ============================================================

OUTPUT_DIR.mkdir(exist_ok=True)

final_df.to_csv(
    OUTPUT_FILE,
    index=False
)

train_df.to_csv(
    TRAIN_FILE,
    index=False
)

test_df.to_csv(
    TEST_FILE,
    index=False
)


# ============================================================
# 10. DISPLAY RESULTS
# ============================================================

print("\n========================================")
print("SHORT-TERM PREPROCESSING COMPLETED")
print("========================================")

print("\nFinal dataset size:", len(final_df))

print("\nTraining samples:", len(train_df))
print("Testing samples:", len(test_df))

print("\nTraining period:")
print(
    train_df["date"].min(),
    "to",
    train_df["date"].max()
)

print("\nTesting period:")
print(
    test_df["date"].min(),
    "to",
    test_df["date"].max()
)

print("\nInput features:")
print(feature_columns)

print("\nForecast targets:")
print(target_columns)

print("\nFiles created:")
print(OUTPUT_FILE)
print(TRAIN_FILE)
print(TEST_FILE)

print("\nFirst 5 processed records:")
print(final_df.head())

print("\nDataset shape:")
print(final_df.shape)