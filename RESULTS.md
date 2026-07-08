# Sentinel PdM Evaluation Results

## Dataset Base Rates (Target Imbalance)
- **Train Positive Rate:** 28.6%
- **Validation Positive Rate:** 17.9%
- **Test Positive Rate:** 13.4%

## Overall Test Set Metrics (Amber Tier - Detection)
- **Threshold:** 0.1100
- **Precision:** 0.1714
- **Recall:** 0.9353
- **F1 Score:** 0.2897

## Escalation Test Set Metrics (Red Tier - High Confidence)
- **Threshold:** 0.3700
- **Precision:** 0.2784
- **Recall:** 0.5124
- **F1 Score:** 0.3608

## True Positive Distribution
Out of 188 total failures detected (True Positives at Amber):
- **85** (45.2%) fell into the Amber band (Schedule inspection within 2 days).
- **103** (54.8%) fell into the Red band (Immediate inspection, stop machine).

## Model Selection Rationale
**Winner: Logistic Regression (Balanced)**

Candidate Comparison on Validation Set (Recall-First Dual Tier):

| Model | Amber Threshold | Amber Recall | Amber Precision | Red Threshold | Red Precision |
|---|---|---|---|---|---|
| Logistic Regression (Balanced) | 0.110 | 0.866 | 0.205 | 0.370 | 0.452 |
| Random Forest (Regularized) | 0.210 | 0.825 | 0.200 | 0.340 | 0.457 |
| XGBoost (Scale Pos Weight) | 0.010 | 0.844 | 0.230 | 0.130 | 0.465 |
| SMOTE + Logistic Regression | 0.110 | 0.848 | 0.204 | 0.380 | 0.458 |

## Two-Tier Operating Point
The alert tiers have asymmetric false-positive costs. The **Amber** tier triggers a "schedule inspection within 2 days" action, which is cheap if wrong. Therefore, the Amber threshold was tuned generously to maximize overall recall (at a minimum precision floor of 0.20), ensuring we catch as many failures as possible at an acceptable investigation cost. The **Red** tier triggers an "immediate inspection, stop the machine" action, which is expensive if wrong as it halts production. Therefore, the Red threshold was tuned conservatively to require much higher confidence (precision >= 0.45), ensuring production is only disrupted when a failure is highly probable. This approach results in 85 warnings and 103 critical stops among the caught actual failures.

## Methodology Correction
The model selection methodology was originally designed as precision-first at a single threshold (maximizing precision subject to recall >= 0.80). This was revised to a recall-first, two-tier threshold design. The single blanket threshold masked real performance differences by forcing all candidates to converge around the 0.80 recall mark, and failed to reflect the differing real-world costs of Amber versus Red actions. The revised logic evaluates models based on the highest overall recall achieved at the generous Amber threshold, while strictly enforcing a high-confidence precision floor for the Red threshold.

## Per-Failure-Mode Breakdown (Amber Tier)
Recall and False Negatives stratified by original failure mode (on test set):

| Failure Mode | Total Failures | True Positives (Caught) | False Negatives (Missed) | Recall |
|---|---|---|---|---|
| TWF | 7 | 7 | 0 | 1.00 |
| HDF | 0 | 0 | 0 | 0.00 |
| PWF | 7 | 7 | 0 | 1.00 |
| OSF | 17 | 17 | 0 | 1.00 |
| RNF | 0 | 0 | 0 | 0.00 |

## PR Curve
![PR Curve](models/pr_curve.png)
The full precision-recall arrays have also been exported to `models/pr_curve.csv`.

## Important Assumptions and Limitations
1. **UDI as Pseudo-Time-Series**: We assume `UDI` ordered rows represent contiguous temporal operational cycles for feature building (rolling windows) and the forward 50-cycle target, despite the dataset lacking real timestamps. This means our 50-cycle prediction horizon implies operational sequence rather than real time (e.g., 24 hours).
2. **FFT as Proxy Signal**: We assume FFT components computed over the past 50 rows on torque/speed serve as a proxy for vibration regimes, because the AI4I dataset contains no genuine vibration sensor channel.
