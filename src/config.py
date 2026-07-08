import os
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"

# Pipeline constants
RANDOM_SEED = 42
PREDICTION_HORIZON_CYCLES = 10
ROLLING_WINDOWS = (10, 50)

# Column groups
TARGET_COLUMN = "Machine failure"
ID_COLUMNS = ["UDI", "Product ID"]
FAILURE_MODE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
LEAKAGE_COLUMNS = ID_COLUMNS + FAILURE_MODE_COLUMNS

SENSOR_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]
BASE_FEATURE_COLUMNS = ["Type"] + SENSOR_COLUMNS
EXPECTED_COLUMNS = ID_COLUMNS + BASE_FEATURE_COLUMNS + [TARGET_COLUMN] + FAILURE_MODE_COLUMNS

# Split fractions
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15

# Evaluation
RECALL_FLOOR = 0.80
AMBER_PRECISION_FLOOR = 0.20
RED_PRECISION_FLOOR = 0.45
