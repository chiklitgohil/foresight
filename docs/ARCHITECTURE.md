## 1. SYSTEM OVERVIEW

The system loads the AI4I raw CSV, constructs ordered pseudo-cycle features from past/current rows, trains a class-weighted failure classifier, and saves the fitted feature pipeline plus model as a reproducible artifact. Inference consumes one machine telemetry snapshot, returns a risk-banded prediction with recommended action and top contributing features, and the dashboard displays only operator-safe risk bands and actions while replaying held-out rows as simulated live telemetry.

## 2. FULL FILE/FOLDER TREE

```text
sentinel-pdm/
|-- data/
|   |-- raw/
|   |   `-- ai4i2020.csv
|   `-- processed/
|       |-- train_features.csv
|       |-- validation_features.csv
|       |-- test_features.csv
|       `-- replay_rows.csv
|-- src/
|   |-- __init__.py
|   |-- config.py
|   |-- data_loader.py
|   |-- feature_engineering.py
|   |-- train.py
|   |-- evaluate.py
|   |-- inference.py
|   |-- risk_bands.py
|   `-- simulate_stream.py
|-- models/
|   |-- sentinel_model.joblib
|   `-- model_metadata.json
|-- dashboard/
|   |-- __init__.py
|   `-- app.py
|-- notebooks/
|   |-- eda.py
|   `-- plots/
|-- docs/
|   |-- DATA_PROFILE.md
|   |-- PRD.md
|   |-- ARCHITECTURE.md
|   |-- DECISIONS.md
|   `-- PRESENTATION_OUTLINE.md
|-- tests/
|   |-- test_feature_engineering.py
|   |-- test_risk_bands.py
|   `-- test_inference_contract.py
|-- RESULTS.md
|-- README.md
|-- requirements.txt
`-- .gitignore
```

## 3. PER-FILE RESPONSIBILITY

- `src/__init__.py` - marks `src` as an importable package and produces no runtime output.
- `src/config.py` - centralizes constants such as file paths, random seed `42`, sensor column names, leakage columns, 50-cycle prediction horizon, 10/50-cycle feature windows, and Green/Amber/Red thresholds for reuse by all pipeline scripts.
- `src/data_loader.py` - takes `data/raw/ai4i2020.csv` as input, validates the expected AI4I schema and clean data assumptions from `DATA_PROFILE.md`, sorts by `UDI`, and outputs a cleaned dataframe with identifiers retained only for ordering, diagnostics, and replay metadata.
- `src/feature_engineering.py` - takes the cleaned raw dataframe as input and outputs train/validation/test-ready feature dataframes with the PRD Section 6 interaction, rolling, spectral, anomaly, and forward 50-cycle target features while excluding `UDI`, `Product ID`, `TWF`, `HDF`, `PWF`, `OSF`, and `RNF` from model inputs.
- `src/train.py` - takes engineered training and validation feature tables as input, fits class-weighted candidate classifiers plus the preprocessing pipeline, selects the best calibrated model by recall-first validation behavior, and outputs `models/sentinel_model.joblib` plus `models/model_metadata.json`.
- `src/evaluate.py` - takes `models/sentinel_model.joblib`, `models/model_metadata.json`, and held-out test features as input, applies the chosen operating point, and outputs metrics and plots for `RESULTS.md` including precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, precision-recall curve, and per-failure-mode breakdown.
- `src/inference.py` - takes a raw or engineered single-row telemetry dict plus the saved model artifact as input and outputs the exact model-to-dashboard prediction dict defined in Section 5.
- `src/risk_bands.py` - takes an internal model probability as input and outputs the PRD-defined risk band and recommended action using Green `<0.30`, Amber `0.30-<0.60`, and Red `>=0.60`.
- `src/simulate_stream.py` - takes `data/processed/replay_rows.csv` or held-out test rows as input and yields contract-compliant inference events on a fixed timer for the dashboard demo.
- `dashboard/__init__.py` - marks `dashboard` as an importable package and produces no runtime output.
- `dashboard/app.py` - takes events from `src/simulate_stream.py` or a mocked Section 5 prediction feed as input and outputs an operator dashboard showing risk bands, recommended actions, top contributing features, and machine status without exposing raw risk score in the operator view.
- `notebooks/eda.py` - takes the raw AI4I CSV as input and outputs the already-defined exploration console report plus plots in `notebooks/plots/`.
- `tests/test_feature_engineering.py` - takes small fixture dataframes as input and outputs pass/fail checks that rolling features use only past/current rows and that leakage columns are excluded from model features.
- `tests/test_risk_bands.py` - takes boundary probabilities as input and outputs pass/fail checks for exact Green/Amber/Red threshold behavior and recommended action strings.
- `tests/test_inference_contract.py` - takes a mocked trained model or mocked prediction object as input and outputs pass/fail checks that inference returns every required Section 5 field with the correct types and allowed values.
- `RESULTS.md` - takes evaluation outputs from `src/evaluate.py` as input and records final held-out metrics, chosen operating point, PR curve reference, confusion matrix, and per-failure-mode breakdown.

## 4. DATA FLOW DIAGRAM

```text
data/raw/ai4i2020.csv
  -> src/data_loader.py
  -> src/feature_engineering.py
  -> data/processed/train_features.csv
  -> src/train.py
  -> models/sentinel_model.joblib + models/model_metadata.json
  -> src/inference.py
  -> src/risk_bands.py
  -> dashboard/app.py

Evaluation branch:
src/feature_engineering.py
  -> data/processed/test_features.csv
  -> src/evaluate.py
  -> RESULTS.md

Demo replay branch:
src/feature_engineering.py
  -> data/processed/replay_rows.csv
  -> src/simulate_stream.py
  -> src/inference.py
  -> src/risk_bands.py
  -> dashboard/app.py
```

## 5. THE MODEL <-> DASHBOARD CONTRACT

`src/inference.py` outputs exactly this dict shape, and `dashboard/app.py` consumes exactly this dict shape:

```json
{
  "machine_id": "machine_001",
  "cycle_index": 1234,
  "event_time": "2026-07-07T12:00:00Z",
  "risk_score": 0.74,
  "risk_band": "red",
  "recommended_action": "Immediate inspection, prepare downtime window, and stop the machine if operator safety or production-critical equipment is involved.",
  "top_contributing_features": [
    {
      "feature": "Torque [Nm]",
      "value": 62.4,
      "contribution": 0.18
    },
    {
      "feature": "tool_wear_high_flag",
      "value": 1.0,
      "contribution": 0.11
    },
    {
      "feature": "torque_x_speed",
      "value": 92500.0,
      "contribution": 0.08
    }
  ],
  "predicted_failure_mode": null,
  "diagnostic_failure_mode_context": {
    "TWF": 0,
    "HDF": 0,
    "PWF": 1,
    "OSF": 0,
    "RNF": 0
  }
}
```

Field rules:

- `machine_id`: `str`, simulated because AI4I has no real machine identity; generated as `machine_001` through `machine_020` during replay.
- `cycle_index`: `int`, copied from `UDI` for ordered replay and display context only.
- `event_time`: `str`, ISO 8601 simulated timestamp generated by replay, starting at a fixed demo time and advancing one minute per cycle.
- `risk_score`: `float` from `0.0` to `1.0`, raw model probability, internal only.
- `risk_band`: one of `"green"`, `"amber"`, or `"red"` from `src/risk_bands.py`.
- `recommended_action`: exact action string returned by `src/risk_bands.py`.
- `top_contributing_features`: list of 3-5 objects, each with `feature: str`, `value: float`, and `contribution: float`; first implementation may use model coefficients or feature importance proxy, not full SHAP.
- `predicted_failure_mode`: always `null` in core scope because PRD.md collapses failure modes into the single binary `Machine failure` target.
- `diagnostic_failure_mode_context`: optional demo/debug context copied from held-out labeled replay rows when available; dashboard may show it only as historical context, not as model input or predicted subclass.

`dashboard/app.py` must NEVER display `risk_score` directly to the simulated operator view; it displays only `risk_band`, `recommended_action`, machine status, and contributing feature names/values. A separate debug/analyst view may display `risk_score` if time allows, but that view is optional and must be visually separate from the operator workflow.

## 6. TRAINING <-> INFERENCE CONTRACT

`src/train.py` saves one joblib artifact at `models/sentinel_model.joblib` containing the fitted preprocessing pipeline, encoder/scaler, calibrated classifier, selected feature list, and threshold metadata needed for inference. `src/train.py` also saves `models/model_metadata.json` with `feature_names_in_order`, `target_name`, `horizon_cycles: 50`, `risk_thresholds`, `random_seed: 42`, training timestamp, model type, package versions, and validation operating-point metrics. `src/inference.py` loads `models/sentinel_model.joblib`, applies the bundled preprocessing pipeline, and passes features to the classifier in exactly the `feature_names_in_order` order from metadata; inference fails fast if any required raw column is missing or if a feature order mismatch is detected.

## 7. SIMULATION / DEMO DATA FLOW

`src/simulate_stream.py` reads `data/processed/replay_rows.csv`, which is derived from held-out ordered rows and includes raw display fields, engineered model inputs, true labels for debug context, and original failure-mode labels for historical diagnostics. Because AI4I has no real machine identity and `Product ID` is unique per row, replay assigns synthetic machine IDs by cycling through `machine_001` to `machine_020` in `UDI` order; `Product ID` remains hidden from the operator UI. The replay interval is 1 row per second by default, configurable to 0.25 seconds for fast demos, and simulated `event_time` starts at `2026-07-07T09:00:00Z` with one minute added per cycle.

## 8. DIVISION OF LABOR PROPOSAL

Person A owns the modeling pipeline: `src/config.py`, `src/data_loader.py`, `src/feature_engineering.py`, `src/train.py`, `src/evaluate.py`, `src/risk_bands.py`, `tests/test_feature_engineering.py`, `tests/test_risk_bands.py`, and `RESULTS.md`. Person B owns the product/demo surface: `src/inference.py`, `src/simulate_stream.py`, `dashboard/app.py`, `tests/test_inference_contract.py`, `README.md` demo instructions, and presentation screenshots. Person B starts immediately against a mocked Section 5 prediction dict and a dummy local predictor, then swaps in Person A's `models/sentinel_model.joblib` once the first train artifact exists.

## 9. BUILD ORDER / MILESTONES

- Hours 0-4: create `src/config.py`, `src/data_loader.py`, and a minimal `src/feature_engineering.py` that loads raw data, sorts by `UDI`, excludes leakage columns, creates the 50-cycle target, and writes processed splits.
- Hours 4-8: create a baseline `src/train.py` with class-weighted Logistic Regression and a saved `models/sentinel_model.joblib`, plus a minimal `src/inference.py` that can score one row.
- Hours 8-12: create `src/risk_bands.py`, `src/evaluate.py`, and first `RESULTS.md` metrics with precision, recall, F1, ROC-AUC, PR-AUC, and confusion matrix at the recall-first operating point.
- Hours 12-18: expand feature engineering to the PRD feature set: torque-speed interactions, temperature differential, wear thresholds, 10/50-cycle rolling stats, spectral features, and anomaly z-scores.
- Hours 18-24: create `src/simulate_stream.py` and a dashboard that consumes mocked Section 5 events, displays machine cards, and never shows raw probability in the operator view.
- Hours 24-32: integrate real inference output into the dashboard, add top contributing feature display, add replay controls, and validate the model-dashboard contract with tests.
- Hours 32-40: improve model selection and calibration, update `RESULTS.md`, polish the risk-band operating point chart, and freeze the trained artifact.
- Hours 40-48: finalize README, presentation outline, screenshots, demo script, and contingency fallback using the baseline model if advanced features do not improve held-out recall/precision.

## 10. OPEN RISKS

- The PRD's 50-cycle horizon is a pseudo-sequence assumption based on monotonic `UDI`, not a real timestamped machine history; implementation must avoid claiming true 24-hour prediction.
- Rolling and spectral features may be weak because AI4I rows are synthetic snapshots rather than true per-machine time series; the baseline model must remain runnable even if advanced features add little value.
- Class imbalance plus the forward 50-cycle target may change the positive rate and make the target recall >= 0.80 with precision >= 0.20 difficult; evaluation must report the achieved curve honestly.
- Per-feature contribution scores may be expensive or unstable if implemented with SHAP under time pressure; the fallback is coefficient-based or permutation/importance proxy explanations.
- Dashboard integration can break if inference returns feature names or action strings that drift from this document; `tests/test_inference_contract.py` is mandatory if time is tight.
- Synthetic machine IDs can imply fleet structure that does not exist in the raw data; demo copy must describe them as replay lanes, not real machines.
