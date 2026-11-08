"""
Data preparation for the UCI Individual Household Electric Power Consumption dataset.

Pipeline:
  1. Load raw minute-level data (semicolon-separated, '?' for missing)
  2. Parse timestamps, drop/impute missing values
  3. Resample to hourly mean Global_active_power (kW) and sub-meter aggregates
  4. Engineer calendar + lag + rolling features
  5. Chronological split: train (2006-12 .. 2009-06), val (2009-07 .. 2010-03), test (2010-04 .. 2010-11)
  6. Save cleaned artifacts to data/ as CSV + JSON metadata
"""
import json
import numpy as np
import pandas as pd

RAW = "/home/z/my-project/data/household_power_consumption.txt"
OUT_DIR = "/home/z/my-project/data"

# ---------------------------------------------------------------- 1. load
print("Loading raw data ...")
df = pd.read_csv(
    RAW,
    sep=";",
    na_values="?",
    parse_dates={"timestamp": ["Date", "Time"]},
    dayfirst=True,
    low_memory=False,
)
print(f"Raw rows: {len(df):,}")

# ---------------------------------------------------------------- 2. clean
n_missing = int(df.isna().any(axis=1).sum())
print(f"Rows containing missing values: {n_missing} ({n_missing/len(df)*100:.2f}%)")

# Missing values occur in contiguous blocks (known sensor outages).
# Minute-level gaps are short; time-interpolation (linear on time index) is standard.
df = df.set_index("timestamp").sort_index()
df = df.interpolate(method="time", limit=60)  # fill at most 60 min at once
df = df.dropna()
print(f"Rows after cleaning: {len(df):,}")

# Derived: remaining (unsubmetered) power per UCI docs
# remaining = (GAP*1000/60) - SM1 - SM2 - SM3  [watt-minutes -> watts]
df["sub_other"] = (df["Global_active_power"] * 1000 / 60
                   - df["Sub_metering_1"] - df["Sub_metering_2"] - df["Sub_metering_3"]).clip(lower=0)

# ---------------------------------------------------------------- 3. hourly resample
print("Resampling to hourly ...")
h = df.resample("h").agg(
    global_active_power=("Global_active_power", "mean"),   # mean kW
    global_reactive_power=("Global_reactive_power", "mean"),
    voltage=("Voltage", "mean"),
    sub_metering_1=("Sub_metering_1", "mean"),
    sub_metering_2=("Sub_metering_2", "mean"),
    sub_metering_3=("Sub_metering_3", "mean"),
    sub_other=("sub_other", "mean"),
)
# hours where the source had substantial missingness -> drop to avoid fabricated averages
h = h.dropna()
print(f"Hourly rows: {len(h):,}   range: {h.index.min()} .. {h.index.max()}")

# ---------------------------------------------------------------- 4. features
X = pd.DataFrame(index=h.index)
X["gap"] = h["global_active_power"]                       # target: mean hourly kW
X["hour"] = X.index.hour
X["dayofweek"] = X.index.dayofweek
X["month"] = X.index.month
X["is_weekend"] = (X.index.dayofweek >= 5).astype(int)
X["dayofyear"] = X.index.dayofyear

# lag / rolling features (use only past information)
for lag in (1, 2, 3, 24, 48, 168):
    X[f"lag_{lag}"] = X["gap"].shift(lag)
for w in (3, 24):
    X[f"roll_mean_{w}"] = X["gap"].shift(1).rolling(w).mean()
X["roll_std_24"] = X["gap"].shift(1).rolling(24).std()
# previous-day same-hour and previous-week same-hour are lag_24 / lag_168
X = X.dropna()

print(f"Feature rows after lag trimming: {len(X):,}")

# ---------------------------------------------------------------- 5. chronological split
train_end = "2009-06-30 23:00"
val_end = "2010-03-31 23:00"
train = X[X.index <= train_end]
val = X[(X.index > train_end) & (X.index <= val_end)]
test = X[X.index > val_end]
print(f"train: {len(train):,} ({train.index.min()} .. {train.index.max()})")
print(f"val  : {len(val):,} ({val.index.min()} .. {val.index.max()})")
print(f"test : {len(test):,} ({test.index.min()} .. {test.index.max()})")

# ---------------------------------------------------------------- 6. save
train.to_csv(f"{OUT_DIR}/train.csv")
val.to_csv(f"{OUT_DIR}/val.csv")
test.to_csv(f"{OUT_DIR}/test.csv")
h.to_csv(f"{OUT_DIR}/hourly_full.csv")

meta = {
    "raw_rows": int(2_075_259),
    "raw_rows_missing": n_missing,
    "hourly_rows": int(len(h)),
    "feature_rows": int(len(X)),
    "train_rows": int(len(train)),
    "val_rows": int(len(val)),
    "test_rows": int(len(test)),
    "train_range": [str(train.index.min()), str(train.index.max())],
    "val_range": [str(val.index.min()), str(val.index.max())],
    "test_range": [str(test.index.min()), str(test.index.max())],
    "features": list(X.columns),
    "target": "gap",
    "units": "kilowatt (mean hourly global active power)",
}
with open(f"{OUT_DIR}/meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print("Saved train/val/test CSVs + meta.json")
print(X.describe().T[["mean", "std", "min", "max"]])
