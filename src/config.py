"""Centralizes constants such as file paths, random seed `42`, sensor column names, leakage columns, 50-cycle prediction horizon, 10/50-cycle feature windows, and Green/Amber/Red thresholds for reuse by all pipeline scripts."""

from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "ai4i2020.csv"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"

RANDOM_SEED = 42
PREDICTION_HORIZON_CYCLES = 50
ROLLING_WINDOWS = (10, 50)

TARGET_COLUMN = "Machine failure"
ID_COLUMNS = ["UDI", "Product ID"]
FAILURE_MODE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
LEAKAGE_COLUMNS = [*ID_COLUMNS, *FAILURE_MODE_COLUMNS]
SENSOR_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
BASE_FEATURE_COLUMNS = ["Type", *SENSOR_COLUMNS]
EXPECTED_COLUMNS = [*ID_COLUMNS, "Type", *SENSOR_COLUMNS, TARGET_COLUMN, *FAILURE_MODE_COLUMNS]

GREEN_MAX = 0.30
AMBER_MAX = 0.60
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15
