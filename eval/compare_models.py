"""
Model Comparison and Evaluation
Trains candidate models, evaluates them on the held-out test set, and generates PR curves.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, auc, precision_recall_curve
from imblearn.over_sampling import SMOTE

# Add root to sys.path to import local modules
root_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(root_dir))

from data.preprocess import load_and_preprocess
from features.engineer import build_features

def run_comparison():
    raw_path = root_dir / "data" / "raw" / "ai4i2020.csv"
    df = load_and_preprocess(raw_path)
    df = build_features(df)
    
    # Target and Features
    target = "Machine failure"
    X = df.drop(columns=[target])
    y = df[target]
    
    # Split chronologically or stratified? 
    # For time-series, chronological split is safer, but AI4I is often treated with stratified shuffle 
    # due to synthetic nature. We'll use stratified split to ensure test set has failures.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Class Imbalance Strategy: SMOTE (Synthetic Minority Over-sampling Technique)
    # Applied ONLY to the training set to prevent data leakage into the test set.
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    
    # Define models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "MLP Neural Net": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    }
    
    results = []
    
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_resampled, y_train_resampled)
        
        # Predict probabilities
        y_probs = model.predict_proba(X_test_scaled)[:, 1]
        y_preds = model.predict(X_test_scaled)
        
        # Compute metrics
        prec = precision_score(y_test, y_preds, zero_division=0)
        rec = recall_score(y_test, y_preds)
        f1 = f1_score(y_test, y_preds)
        roc_auc = roc_auc_score(y_test, y_probs)
        
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_probs)
        pr_auc = auc(recall_vals, precision_vals)
        
        results.append({
            "Model": name,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "ROC-AUC": roc_auc,
            "PR-AUC": pr_auc
        })
        
        # Plot PR curve
        plt.plot(recall_vals, precision_vals, label=f"{name} (AUC = {pr_auc:.3f})")
    
    # Save comparison table
    results_df = pd.DataFrame(results)
    eval_dir = root_dir / "eval"
    eval_dir.mkdir(exist_ok=True)
    results_df.to_csv(eval_dir / "comparison_table.csv", index=False)
    print("\nComparison Table:")
    print(results_df)
    
    # Save PR curve plot
    plt.title("Precision-Recall Curve Comparison")
    plt.xlabel("Recall (Fraction of actual failures caught)")
    plt.ylabel("Precision (Fraction of alarms that are true failures)")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(eval_dir / "pr_curves.png")
    plt.close()
    
if __name__ == "__main__":
    run_comparison()
