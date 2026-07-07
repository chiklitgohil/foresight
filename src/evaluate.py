"""Takes `models/sentinel_model.joblib`, `models/model_metadata.json`, and held-out test features as input, applies the chosen operating point, and outputs metrics and plots for `RESULTS.md` including precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, precision-recall curve, and per-failure-mode breakdown."""

from __future__ import annotations

import pandas as pd


def evaluate_model(model: object, test_df: pd.DataFrame, metadata: dict) -> dict:
    """Take a fitted model, held-out test dataframe, and metadata and return evaluation metrics."""
    raise NotImplementedError


def write_results(metrics: dict) -> None:
    """Take evaluation metrics and update RESULTS.md with final held-out results."""
    raise NotImplementedError


def main() -> None:
    """Run held-out evaluation for the saved Sentinel PdM model."""
    raise NotImplementedError
