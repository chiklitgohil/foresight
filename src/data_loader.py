"""Takes `data/raw/ai4i2020.csv` as input, validates the expected AI4I schema and clean data assumptions from `DATA_PROFILE.md`, sorts by `UDI`, and outputs a cleaned dataframe with identifiers retained only for ordering, diagnostics, and replay metadata."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from src.config import (
        BASE_FEATURE_COLUMNS,
        EXPECTED_COLUMNS,
        FAILURE_MODE_COLUMNS,
        ID_COLUMNS,
        LEAKAGE_COLUMNS,
        PROCESSED_DIR,
        RAW_DATA_PATH,
        SENSOR_COLUMNS,
        TARGET_COLUMN,
        TEST_FRACTION,
        TRAIN_FRACTION,
        VALIDATION_FRACTION,
    )
except ModuleNotFoundError:
    from config import (
        BASE_FEATURE_COLUMNS,
        EXPECTED_COLUMNS,
        FAILURE_MODE_COLUMNS,
        ID_COLUMNS,
        LEAKAGE_COLUMNS,
        PROCESSED_DIR,
        RAW_DATA_PATH,
        SENSOR_COLUMNS,
        TARGET_COLUMN,
        TEST_FRACTION,
        TRAIN_FRACTION,
        VALIDATION_FRACTION,
    )


OUTPUT_COLUMNS = [*ID_COLUMNS, *BASE_FEATURE_COLUMNS, TARGET_COLUMN, *FAILURE_MODE_COLUMNS]
TRAIN_OUTPUT = PROCESSED_DIR / "train_features.csv"
VALIDATION_OUTPUT = PROCESSED_DIR / "validation_features.csv"
TEST_OUTPUT = PROCESSED_DIR / "test_features.csv"
REPLAY_OUTPUT = PROCESSED_DIR / "replay_rows.csv"


def load_data(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw AI4I CSV and return a dataframe with the expected 10,000-row schema."""
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Missing raw dataset at {data_path}")

    df = pd.read_csv(data_path, encoding="utf-8-sig")
    validate_schema(df)
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Validate row count, columns, missingness, class balance, and sensor ranges."""
    missing_columns = [column for column in EXPECTED_COLUMNS if column not in df.columns]
    extra_columns = [column for column in df.columns if column not in EXPECTED_COLUMNS]
    assert not missing_columns, f"Missing expected columns: {missing_columns}"
    assert not extra_columns, f"Unexpected columns found: {extra_columns}"
    assert len(df) == 10_000, f"Expected 10,000 rows, found {len(df)}"
    assert df.isna().sum().sum() == 0, "Expected zero missing values"

    failure_rate = df[TARGET_COLUMN].mean()
    assert 0.033 <= failure_rate <= 0.035, f"Unexpected failure rate: {failure_rate:.4f}"

    assert df["UDI"].is_unique, "UDI must be unique"
    assert df["Product ID"].is_unique, "Product ID must be unique"
    assert set(df[TARGET_COLUMN].unique()).issubset({0, 1}), "Target must be binary"
    for column in FAILURE_MODE_COLUMNS:
        assert set(df[column].unique()).issubset({0, 1}), f"{column} must be binary"

    assert (df["Air temperature [K]"] >= 0).all(), "Air temperature contains negative values"
    assert (df["Process temperature [K]"] >= 0).all(), "Process temperature contains negative values"
    assert (df["Rotational speed [rpm]"] >= 0).all(), "Rotational speed contains negative values"
    assert (df["Torque [Nm]"] >= 0).all(), "Torque contains negative values"
    assert (df["Tool wear [min]"] >= 0).all(), "Tool wear contains negative values"
    assert (df["Process temperature [K]"] >= df["Air temperature [K]"]).all(), (
        "Process temperature should not be below air temperature"
    )


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a duplicate-free dataframe sorted by UDI with architecture-approved output columns."""
    cleaned = df.drop_duplicates().sort_values("UDI").reset_index(drop=True)
    assert len(cleaned) == len(df), "Duplicate rows were removed unexpectedly"
    assert cleaned["UDI"].is_monotonic_increasing, "UDI must be monotonic after sorting"

    feature_columns = BASE_FEATURE_COLUMNS
    leakage_in_features = sorted(set(feature_columns).intersection(LEAKAGE_COLUMNS))
    assert not leakage_in_features, f"Leakage columns present in feature set: {leakage_in_features}"
    assert TARGET_COLUMN not in feature_columns, "Target column must not be in feature columns"

    return cleaned[OUTPUT_COLUMNS]


def split_ordered(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split sorted data into ordered 70/15/15 train, validation, and test dataframes."""
    total_rows = len(df)
    train_end = int(total_rows * TRAIN_FRACTION)
    validation_end = train_end + int(total_rows * VALIDATION_FRACTION)

    train_df = df.iloc[:train_end].copy()
    validation_df = df.iloc[train_end:validation_end].copy()
    test_df = df.iloc[validation_end:].copy()

    expected_test_rows = total_rows - train_end - int(total_rows * VALIDATION_FRACTION)
    assert len(train_df) == 7_000, f"Expected 7,000 train rows, found {len(train_df)}"
    assert len(validation_df) == 1_500, f"Expected 1,500 validation rows, found {len(validation_df)}"
    assert len(test_df) == expected_test_rows, f"Expected {expected_test_rows} test rows, found {len(test_df)}"
    assert train_df["UDI"].max() < validation_df["UDI"].min() < test_df["UDI"].min(), (
        "Ordered split boundaries are invalid"
    )

    return train_df, validation_df, test_df


def save_splits(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str | Path = PROCESSED_DIR,
) -> dict[str, Path]:
    """Write processed train, validation, test, and replay CSVs and return their paths."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    outputs = {
        "train": output_path / "train_features.csv",
        "validation": output_path / "validation_features.csv",
        "test": output_path / "test_features.csv",
        "replay": output_path / "replay_rows.csv",
    }

    train_df.to_csv(outputs["train"], index=False)
    validation_df.to_csv(outputs["validation"], index=False)
    test_df.to_csv(outputs["test"], index=False)
    test_df.to_csv(outputs["replay"], index=False)
    return outputs


def main() -> None:
    """Load, validate, clean, split, and save the initial processed AI4I data files."""
    df = load_data()
    cleaned = clean_data(df)
    train_df, validation_df, test_df = split_ordered(cleaned)
    outputs = save_splits(train_df, validation_df, test_df)

    print("AI4I data_loader completed successfully")
    print(f"Raw rows: {len(df)}")
    print(f"Cleaned rows: {len(cleaned)}")
    print(f"Train rows: {len(train_df)} | failures: {int(train_df[TARGET_COLUMN].sum())}")
    print(f"Validation rows: {len(validation_df)} | failures: {int(validation_df[TARGET_COLUMN].sum())}")
    print(f"Test rows: {len(test_df)} | failures: {int(test_df[TARGET_COLUMN].sum())}")
    print(f"Overall failure rate: {cleaned[TARGET_COLUMN].mean() * 100:.2f}%")
    print(f"Feature columns: {BASE_FEATURE_COLUMNS}")
    print(f"Excluded leakage columns from features: {LEAKAGE_COLUMNS}")
    for name, path in outputs.items():
        print(f"Wrote {name}: {path}")


if __name__ == "__main__":
    main()
