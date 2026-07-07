"""Takes a raw or engineered single-row telemetry dict plus the saved model artifact as input and outputs the exact model-to-dashboard prediction dict defined in Section 5."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_artifact(model_path: str | Path) -> object:
    """Take a model artifact path and return the fitted preprocessing/model bundle."""
    raise NotImplementedError


def predict_one(row: dict[str, Any], model: object | None = None) -> dict[str, Any]:
    """Take one telemetry row and return the Section 5 model-to-dashboard prediction dict."""
    raise NotImplementedError


def top_contributing_features(row: dict[str, Any], model: object) -> list[dict[str, float | str]]:
    """Take one row and fitted model and return the top 3-5 feature contribution objects."""
    raise NotImplementedError
