import pandas as pd
import numpy as np
from src import config

def load_and_split_data():
    csv_path = config.DATA_RAW_DIR / "ai4i2020.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw data file not found at {csv_path}")

    df = pd.read_csv(csv_path)

    # Validate schema
    if len(df) != 10000:
        raise ValueError(f"Expected 10,000 rows, got {len(df)}")
    if df.isnull().values.any():
        raise ValueError("Dataset contains missing values.")
    
    failure_rate = df[config.TARGET_COLUMN].mean()
    if not (0.033 <= failure_rate <= 0.035):
        raise ValueError(f"Failure rate {failure_rate} is outside the expected range (0.033-0.035)")

    if df['UDI'].nunique() != len(df):
        raise ValueError("UDI is not unique across all rows.")
    if df['Product ID'].nunique() != len(df):
        raise ValueError("Product ID is not unique across all rows.")

    for sensor in config.SENSOR_COLUMNS:
        if (df[sensor] < 0).any():
            raise ValueError(f"Sensor {sensor} contains negative values.")

    if (df['Process temperature [K]'] < df['Air temperature [K]']).any():
        raise ValueError("Process temperature is lower than Air temperature in some rows.")

    # Sort by UDI and assert monotonic
    df = df.sort_values(by="UDI").reset_index(drop=True)
    if not df['UDI'].is_monotonic_increasing:
        raise ValueError("UDI is not monotonically increasing after sort.")

    # Split into ORDERED contiguous chunks
    total_rows = len(df)
    n_train = int(total_rows * config.TRAIN_FRACTION)
    n_val = int(total_rows * config.VALIDATION_FRACTION)
    n_test = total_rows - n_train - n_val  # ensure exact sum

    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_train+n_val].copy()
    test_df = df.iloc[n_train+n_val:].copy()

    # Assert row counts
    assert len(train_df) == int(total_rows * config.TRAIN_FRACTION)
    assert len(val_df) == int(total_rows * config.VALIDATION_FRACTION)

    # Assert time travel / ordering
    assert train_df['UDI'].max() < val_df['UDI'].min()
    assert val_df['UDI'].max() < test_df['UDI'].min()

    # Assert leakage
    for leakage_col in config.LEAKAGE_COLUMNS:
        assert leakage_col not in config.BASE_FEATURE_COLUMNS, f"Leakage column {leakage_col} found in base features!"

    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save splits
    train_df.to_csv(config.DATA_PROCESSED_DIR / "train_raw.csv", index=False)
    val_df.to_csv(config.DATA_PROCESSED_DIR / "val_raw.csv", index=False)
    test_df.to_csv(config.DATA_PROCESSED_DIR / "test_raw.csv", index=False)

    # Generate replay CSV (usually just test, but PRD implies holding out rows or using the whole test set)
    test_df.to_csv(config.DATA_PROCESSED_DIR / "replay_rows.csv", index=False)

    print(f"Data loading and splitting complete. Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}.")

if __name__ == "__main__":
    load_and_split_data()
