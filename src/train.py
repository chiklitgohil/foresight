"""Takes engineered training and validation feature tables as input, fits class-weighted candidate classifiers plus the preprocessing pipeline, selects the best calibrated model by recall-first validation behavior, and outputs `models/sentinel_model.joblib` plus `models/model_metadata.json`."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def train_model(train_df: pd.DataFrame, validation_df: pd.DataFrame) -> object:
    """Take engineered train/validation dataframes and return a fitted pipeline/model artifact."""
    raise NotImplementedError


def save_model(model: object, metadata: dict, model_path: str | Path) -> None:
    """Take a fitted model plus metadata and write joblib/json artifacts to disk."""
    raise NotImplementedError


def main() -> None:
    """Train and save the Sentinel PdM model artifact."""
    raise NotImplementedError
