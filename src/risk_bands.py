"""Takes an internal model probability as input and outputs the PRD-defined risk band and recommended action using Green `<0.30`, Amber `0.30-<0.60`, and Red `>=0.60`."""

from __future__ import annotations


def assign_risk_band(risk_score: float) -> tuple[str, str]:
    """Take a probability from 0.0 to 1.0 and return risk_band plus recommended_action."""
    raise NotImplementedError
