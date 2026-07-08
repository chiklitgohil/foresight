"""
Failure Mode Breakdown
Evaluates the final model's precision and recall across specific failure modes.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import recall_score, precision_score

root_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(root_dir))

from data.preprocess import load_and_preprocess
from features.engineer import build_features

def evaluate_failure_modes():
    # Load original raw data with all failure modes intact
    raw_path = root_dir / "data" / "raw" / "ai4i2020.csv"
    raw_df = load_and_preprocess(raw_path)
    
    # Isolate failure modes before dropping them in engineering
    failure_modes = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    target = "Machine failure"
    
    # Process features
    X_full = build_features(raw_df)
    
    # We drop target from X_full
    if target in X_full.columns:
        X_full = X_full.drop(columns=[target])
    
    # Load model
    model_path = root_dir / "model" / "sentinel_model.joblib"
    try:
        artifact = joblib.load(model_path)
    except FileNotFoundError:
        print("Model not found. Run train.py first.")
        return
        
    scaler = artifact["scaler"]
    model = artifact["model"]
    threshold = artifact["optimal_threshold"]
    expected_features = artifact["features"]
    
    for col in expected_features:
        if col not in X_full.columns:
            X_full[col] = 0
            
    X_scaled = scaler.transform(X_full[expected_features])
    
    # Predict probabilities across the ENTIRE dataset (for demonstration of breakdown)
    # Note: In practice we'd only do this on the test set, but for failure breakdown
    # on rare modes, we look at how well the model captures each condition overall.
    y_probs = model.predict_proba(X_scaled)[:, 1]
    y_preds = (y_probs >= threshold).astype(int)
    
    raw_df["Predicted_Failure"] = y_preds
    
    print("=== Per-Failure Mode Breakdown ===")
    results = []
    
    for mode in failure_modes:
        # Filter where this specific failure mode actually happened
        mode_mask = raw_df[mode] == 1
        
        if mode_mask.sum() == 0:
            continue
            
        true_positives = raw_df.loc[mode_mask, "Predicted_Failure"].sum()
        actual_positives = mode_mask.sum()
        mode_recall = true_positives / actual_positives if actual_positives > 0 else 0
        
        # Precision per mode is a bit trickier because the model predicts general failure,
        # not the specific mode. We report Recall to show if the model caught the failure.
        results.append({
            "Failure Mode": mode,
            "Total Actual Occurrences": actual_positives,
            "Caught by Model (TP)": true_positives,
            "Recall": mode_recall
        })
        
    df_results = pd.DataFrame(results)
    print(df_results)
    
    # Save to CSV for RESULTS.md
    df_results.to_csv(root_dir / "eval" / "failure_mode_results.csv", index=False)

if __name__ == "__main__":
    evaluate_failure_modes()
