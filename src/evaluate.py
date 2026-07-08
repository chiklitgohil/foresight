import json
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, precision_recall_curve

from src import config

def evaluate_model():
    # Load model and metadata
    model = joblib.load(config.MODELS_DIR / "sentinel_model.joblib")
    with open(config.MODELS_DIR / "model_metadata.json", "r") as f:
        metadata = json.load(f)
        
    with open(config.MODELS_DIR / "candidate_comparison.json", "r") as f:
        comparison = json.load(f)

    # Load data and diagnostics
    train_df = pd.read_csv(config.DATA_PROCESSED_DIR / "train_features.csv")
    val_df = pd.read_csv(config.DATA_PROCESSED_DIR / "validation_features.csv")
    test_df = pd.read_csv(config.DATA_PROCESSED_DIR / "test_features.csv")
    test_diag = pd.read_csv(config.DATA_PROCESSED_DIR / "test_diagnostics.csv")
    
    # Calculate base rates
    train_base_rate = train_df['forward_target'].mean()
    val_base_rate = val_df['forward_target'].mean()
    test_base_rate = test_df['forward_target'].mean()
    
    # Assert alignment
    if len(test_df) != len(test_diag):
        raise ValueError("Test features and diagnostics row count mismatch.")
    
    # Align features with training order
    features_in_order = metadata["feature_names_in_order"]
    X_test = test_df[features_in_order]
    y_test = test_df['forward_target']
    
    # Predict probabilities
    y_probs = model.predict_proba(X_test)[:, 1]
    
    # Selected operating points
    amber_thresh = metadata["risk_thresholds"]["amber"]
    red_thresh = metadata["risk_thresholds"]["red"]
    
    y_pred_amber = (y_probs >= amber_thresh).astype(int)
    y_pred_red = (y_probs >= red_thresh).astype(int)
    
    # Compute metrics for Amber (overall detection)
    amber_prec = precision_score(y_test, y_pred_amber, zero_division=0)
    amber_rec = recall_score(y_test, y_pred_amber, zero_division=0)
    amber_f1 = f1_score(y_test, y_pred_amber, zero_division=0)
    cm_amber = confusion_matrix(y_test, y_pred_amber)
    tn_a, fp_a, fn_a, tp_a = cm_amber.ravel() if len(cm_amber.ravel()) == 4 else (0,0,0,0)

    # Compute metrics for Red (escalation)
    red_prec = precision_score(y_test, y_pred_red, zero_division=0)
    red_rec = recall_score(y_test, y_pred_red, zero_division=0)
    red_f1 = f1_score(y_test, y_pred_red, zero_division=0)
    cm_red = confusion_matrix(y_test, y_pred_red)
    tn_r, fp_r, fn_r, tp_r = cm_red.ravel() if len(cm_red.ravel()) == 4 else (0,0,0,0)
    
    roc_auc = roc_auc_score(y_test, y_probs)
    pr_auc = average_precision_score(y_test, y_probs)
    
    # True Positive distribution
    # TP_a is all actual positive cases caught at amber threshold (which includes those caught at red)
    # TP_r is actual positive cases caught at red threshold
    tp_amber_only = tp_a - tp_r
    
    # PR Curve & Plot
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
    pr_curve_df = pd.DataFrame({
        "threshold": np.append(thresholds, [1.0]),
        "precision": precisions,
        "recall": recalls
    })
    pr_curve_df.to_csv(config.MODELS_DIR / "pr_curve.csv", index=False)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, label=f"PR Curve (AUC = {pr_auc:.3f})")
    plt.scatter([amber_rec], [amber_prec], color='orange', s=100, label=f'Amber Tier', zorder=5)
    plt.scatter([red_rec], [red_prec], color='red', s=100, label=f'Red Tier', zorder=5)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Test Set)')
    plt.legend()
    plt.grid(True)
    plt.savefig(config.MODELS_DIR / "pr_curve.png")
    plt.close()
    
    # Per-failure-mode breakdown at Amber tier
    test_diag['pred'] = y_pred_amber
    test_diag['fn'] = ((test_diag['forward_target'] == 1) & (test_diag['pred'] == 0)).astype(int)
    test_diag['tp'] = ((test_diag['forward_target'] == 1) & (test_diag['pred'] == 1)).astype(int)
    
    mode_results = {}
    for mode in config.FAILURE_MODE_COLUMNS:
        mode_mask = test_diag[mode] == 1
        total_mode_failures = mode_mask.sum()
        if total_mode_failures > 0:
            fn_mode = test_diag.loc[mode_mask, 'fn'].sum()
            tp_mode = test_diag.loc[mode_mask, 'tp'].sum()
            rec_mode = tp_mode / total_mode_failures
            mode_results[mode] = {"total": int(total_mode_failures), "fn": int(fn_mode), "tp": int(tp_mode), "recall": float(rec_mode)}
        else:
            mode_results[mode] = {"total": 0, "fn": 0, "tp": 0, "recall": 0.0}
            
    # Generate RESULTS.md
    with open(config.ROOT_DIR / "RESULTS.md", "w") as f:
        f.write("# Sentinel PdM Evaluation Results\n\n")
        
        f.write("## Dataset Base Rates (Target Imbalance)\n")
        f.write(f"- **Train Positive Rate:** {train_base_rate:.1%}\n")
        f.write(f"- **Validation Positive Rate:** {val_base_rate:.1%}\n")
        f.write(f"- **Test Positive Rate:** {test_base_rate:.1%}\n\n")
        
        f.write("## Overall Test Set Metrics (Amber Tier - Detection)\n")
        f.write(f"- **Threshold:** {amber_thresh:.4f}\n")
        f.write(f"- **Precision:** {amber_prec:.4f}\n")
        f.write(f"- **Recall:** {amber_rec:.4f}\n")
        f.write(f"- **F1 Score:** {amber_f1:.4f}\n\n")

        f.write("## Escalation Test Set Metrics (Red Tier - High Confidence)\n")
        f.write(f"- **Threshold:** {red_thresh:.4f}\n")
        f.write(f"- **Precision:** {red_prec:.4f}\n")
        f.write(f"- **Recall:** {red_rec:.4f}\n")
        f.write(f"- **F1 Score:** {red_f1:.4f}\n\n")
        
        f.write("## True Positive Distribution\n")
        if tp_a > 0:
            f.write(f"Out of {tp_a} total failures detected (True Positives at Amber):\n")
            f.write(f"- **{tp_amber_only}** ({tp_amber_only/tp_a:.1%}) fell into the Amber band (Schedule inspection within 2 days).\n")
            f.write(f"- **{tp_r}** ({tp_r/tp_a:.1%}) fell into the Red band (Immediate inspection, stop machine).\n\n")
        else:
            f.write("No true positives detected.\n\n")

        f.write("## Model Selection Rationale\n")
        winner = metadata["model_type"]
        f.write(f"**Winner: {winner}**\n\n")
        f.write("Candidate Comparison on Validation Set (Recall-First Dual Tier):\n\n")
        f.write("| Model | Amber Threshold | Amber Recall | Amber Precision | Red Threshold | Red Precision |\n")
        f.write("|---|---|---|---|---|---|\n")
        for cand, res in comparison.items():
            if res["reached_floors"]:
                f.write(f"| {cand} | {res['amber_threshold']:.3f} | {res['amber_recall']:.3f} | {res['amber_precision']:.3f} | {res['red_threshold']:.3f} | {res['red_precision']:.3f} |\n")
            else:
                f.write(f"| {cand} | - | - | - | - | - |\n")
        
        f.write("\n")
        
        f.write("## Two-Tier Operating Point\n")
        f.write("The alert tiers have asymmetric false-positive costs. The **Amber** tier triggers a \"schedule inspection within 2 days\" action, which is cheap if wrong. Therefore, the Amber threshold was tuned generously to maximize overall recall (at a minimum precision floor of 0.20), ensuring we catch as many failures as possible at an acceptable investigation cost. ")
        f.write(f"The **Red** tier triggers an \"immediate inspection, stop the machine\" action, which is expensive if wrong as it halts production. Therefore, the Red threshold was tuned conservatively to require much higher confidence (precision >= 0.45), ensuring production is only disrupted when a failure is highly probable. ")
        f.write(f"This approach results in {tp_amber_only} warnings and {tp_r} critical stops among the caught actual failures.\n\n")
        
        f.write("## Methodology Correction\n")
        f.write("The model selection methodology was originally designed as precision-first at a single threshold (maximizing precision subject to recall >= 0.80). ")
        f.write("This was revised to a recall-first, two-tier threshold design. The single blanket threshold masked real performance differences by forcing all candidates to converge around the 0.80 recall mark, and failed to reflect the differing real-world costs of Amber versus Red actions. ")
        f.write("The revised logic evaluates models based on the highest overall recall achieved at the generous Amber threshold, while strictly enforcing a high-confidence precision floor for the Red threshold.\n\n")
        
        f.write("## Per-Failure-Mode Breakdown (Amber Tier)\n")
        f.write("Recall and False Negatives stratified by original failure mode (on test set):\n\n")
        f.write("| Failure Mode | Total Failures | True Positives (Caught) | False Negatives (Missed) | Recall |\n")
        f.write("|---|---|---|---|---|\n")
        for mode, res in mode_results.items():
            f.write(f"| {mode} | {res['total']} | {res['tp']} | {res['fn']} | {res['recall']:.2f} |\n")
            
        f.write("\n## PR Curve\n")
        f.write("![PR Curve](models/pr_curve.png)\n")
        f.write("The full precision-recall arrays have also been exported to `models/pr_curve.csv`.\n\n")

        f.write("## Important Assumptions and Limitations\n")
        f.write("1. **UDI as Pseudo-Time-Series**: We assume `UDI` ordered rows represent contiguous temporal operational cycles for feature building (rolling windows) and the forward 50-cycle target, despite the dataset lacking real timestamps. This means our 50-cycle prediction horizon implies operational sequence rather than real time (e.g., 24 hours).\n")
        f.write("2. **FFT as Proxy Signal**: We assume FFT components computed over the past 50 rows on torque/speed serve as a proxy for vibration regimes, because the AI4I dataset contains no genuine vibration sensor channel.\n")

if __name__ == "__main__":
    evaluate_model()
