"""
Final Model Training & Threshold Tuning
Trains the selected final model, computes the optimal operating threshold for high recall,
and saves the artifact.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_recall_curve

root_dir = Path(__file__).resolve().parent
sys.path.append(str(root_dir))

from data.preprocess import load_and_preprocess
from features.engineer import build_features

def train_and_save():
    raw_path = root_dir / "data" / "raw" / "ai4i2020.csv"
    df = load_and_preprocess(raw_path)
    df = build_features(df)
    
    target = "Machine failure"
    X = df.drop(columns=[target])
    y = df[target]
    
    # We use a test split here just to find the optimal threshold on unseen data
    # Real-world deployments might use cross-validation for threshold tuning.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Imbalance strategy
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    
    # Selected Model based on evaluation logic (Gradient Boosting had highest Recall at target precision)
    print("Training Final Model (Gradient Boosting)...")
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train_resampled, y_train_resampled)
    
    # Threshold Tuning
    print("Tuning operating threshold for target Recall >= 0.80...")
    y_probs = model.predict_proba(X_test_scaled)[:, 1]
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_probs)
    
    # Find threshold where recall is at least 0.80, maximizing precision
    # thresholds array is len(precision_vals) - 1
    optimal_threshold = 0.5
    target_recall = 0.80
    
    for i in range(len(thresholds)):
        if recall_vals[i] >= target_recall:
            optimal_threshold = thresholds[i]
            # Since recall_vals is monotonically decreasing, the last one that satisfies 
            # >= 0.80 will have the highest precision in that valid range.
    
    print(f"Chosen Operating Threshold: {optimal_threshold:.3f}")
    
    # Save the pipeline/artifacts
    model_dir = root_dir / "model"
    model_dir.mkdir(exist_ok=True)
    
    artifact = {
        "scaler": scaler,
        "model": model,
        "optimal_threshold": optimal_threshold,
        "features": list(X.columns)
    }
    
    joblib.dump(artifact, model_dir / "sentinel_model.joblib")
    print("Model saved to model/sentinel_model.joblib")

if __name__ == "__main__":
    train_and_save()
