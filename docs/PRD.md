## 1. PROBLEM FRAMING

Sentinel PdM is a predictive maintenance system for factory operators that turns machine sensor telemetry into early failure warnings, risk bands, and maintenance actions. The product exists because missed failures carry asymmetric cost: unplanned downtime costs manufacturers roughly USD 50B per year, and a missed critical failure can mean lost throughput, damaged equipment, scrap, and operator injury, while a false alarm usually means an unnecessary inspection.

## 2. DATASET

- Dataset: AI4I 2020 Predictive Maintenance Dataset.
- Source URL: UCI Machine Learning Repository, https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset.
- License: Creative Commons Attribution 4.0 International (CC BY 4.0).
- Size: 10,000 rows and 14 columns.
- Preprocessing before modeling: drop leakage/identifier columns from model inputs (`UDI`, `Product ID`, `TWF`, `HDF`, `PWF`, `OSF`, `RNF`), one-hot encode `Type`, create engineered features from the five numeric sensor columns, and construct a forward-looking target from `Machine failure`.

## 3. TARGET VARIABLE & PREDICTION FRAMING

We use a constructed pseudo-sequence with `UDI` as the operational cycle order and define the target as failure within the next 50 operational cycles. `DATA_PROFILE.md` found that `UDI` is monotonic and unique but that the dataset has no real timestamp, so `UDI` is used only for ordered feature windows, forward target construction, and split discipline, not as a model input. We collapse `TWF`, `HDF`, `PWF`, `OSF`, and `RNF` into the single binary `Machine failure` target because only 339 rows fail and 24 rows have multiple failure-mode flags; the five mode columns are reported as diagnostic breakdowns, never used as prediction-time features.

## 4. OPERATING POINT PHILOSOPHY

We optimize for recall at an acceptable precision floor, not accuracy or F1 alone, because a missed failure carries asymmetric cost (downtime, potential injury) versus a false alarm (an unnecessary inspection). We target recall >= 0.80 with precision >= 0.20 on the held-out test set, then choose the highest-precision threshold that satisfies that recall floor. RESULTS.md will report the full precision-recall curve and explicitly mark the chosen operating point.

## 5. RISK BAND DEFINITION

- Green: risk score < 0.30. Recommended action: no maintenance action; continue normal monitoring.
- Amber: 0.30 <= risk score < 0.60. Recommended action: schedule inspection within 2 working days and review tool wear/load indicators.
- Red: risk score >= 0.60. Recommended action: immediate inspection, prepare downtime window, and stop the machine if operator safety or production-critical equipment is involved.

The dashboard shows Green/Amber/Red bands and recommended actions to the operator, never the raw probability score.

## 6. FEATURE ENGINEERING PLAN

- Interaction features: `torque_x_speed = Torque [Nm] * Rotational speed [rpm]`, approximate mechanical power, and a high-torque/low-speed flag, grounded in the strongest single failure correlation (`Torque [Nm]` = 0.191321) and the strong inverse torque-speed relationship (-0.875027).
- Temperature features: `Process temperature [K] - Air temperature [K]`, rolling mean, rolling standard deviation, rolling trend, and threshold-crossing counts for this differential, grounded in the strong air/process temperature relationship (0.876107).
- Wear features: high-wear flags using the upper quartile at 162 minutes and near-maximum bands, plus rolling max and rolling trend for `Tool wear [min]`, grounded in its positive failure correlation (0.105448).
- Rolling load features: 10-cycle and 50-cycle rolling mean/std/min/max/trend for torque, rotational speed, temperature differential, and mechanical load proxy, computed using past/current rows only.
- Spectral features: FFT dominant frequency bin and spectral energy over 50-cycle windows for torque, rotational speed, and the mechanical load proxy, used as a compact pseudo-time-series signal for unstable operating regimes.
- Anomaly detection ideas: machine-type baseline z-scores for torque, rotational speed, temperature differential, and tool wear; threshold-crossing counts for high torque, low speed, high wear, and low temperature differential.
- RUL regression is a stretch goal gated behind completion of the core classifier, evaluation report, and dashboard demo.

## 7. MODEL & EVALUATION PLAN

Primary imbalance strategy: class-weighted models using `class_weight="balanced"` for Logistic Regression and Random Forest, with calibrated probabilities from the best-performing model. We use an ordered train/validation/test split by `UDI` to avoid future leakage from rolling features and forward labels; `UDI`, `Product ID`, and the five failure-mode flags are excluded from input features because `DATA_PROFILE.md` identified them as leakage or non-generalizable identifiers. RESULTS.md will report precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix at the chosen operating point, and per-failure-mode breakdown on held-out failures. Reproducibility commitments: fixed random seed `42`, pinned package versions in `requirements.txt`, deterministic data processing, saved model artifact, and one-command retrain script.

## 8. SCOPE BOUNDARIES

- No live Kafka or streaming infrastructure; the demo uses simulated replay from CSV.
- No multi-factory support; the demo represents one synthetic factory.
- No NASA C-MAPSS integration in core scope.
- No production database, authentication, user roles, or cloud deployment requirement.
- No use of `TWF`, `HDF`, `PWF`, `OSF`, or `RNF` as model input features.
- No raw probability display to operators.
- No real 24-hour timestamp claim; "next 50 cycles" is the defined hackathon prediction horizon.

## 9. DEMO & PRESENTATION PLAN

The live demo shows an operator dashboard replaying historical AI4I rows on a timer as synthetic factory telemetry; machines move between Green, Amber, and Red bands in real time, and clicking a Red machine shows the top contributing engineered features, the likely failure-mode breakdown for context, and the recommended maintenance action. The deck opens with the USD 50B/year downtime framing, then shows how a recall-first operating point catches failures early enough to schedule inspection before reactive downtime.

## 10. RUBRIC TRACEABILITY TABLE

| Rubric criterion | PRD sections |
|---|---|
| Innovation & Approach - 15 pts | Section 6 (Feature Engineering Plan), Section 9 (Demo & Presentation Plan) |
| Technical Implementation - 20 pts | Section 3 (Target Variable & Prediction Framing), Section 7 (Model & Evaluation Plan), Section 8 (Scope Boundaries) |
| Model Accuracy - 15 pts | Section 7 (Model & Evaluation Plan) |
| Recall (Critical) - 25 pts | Section 4 (Operating Point Philosophy), Section 7 (Model & Evaluation Plan) |
| Feature Engineering Quality - 15 pts | Section 6 (Feature Engineering Plan), Section 3 (Target Variable & Prediction Framing) |
| Presentation & Demo - 10 pts | Section 5 (Risk Band Definition), Section 9 (Demo & Presentation Plan) |
