import json
import datetime
import sys
import shutil
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import precision_score, recall_score, precision_recall_curve
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import sklearn
import xgboost

from src import config

def train_and_select():
    train_df = pd.read_csv(config.DATA_PROCESSED_DIR / "train_features.csv")
    val_df = pd.read_csv(config.DATA_PROCESSED_DIR / "validation_features.csv")

    X_train = train_df.drop(columns=['forward_target'])
    y_train = train_df['forward_target']
    
    X_val = val_df.drop(columns=['forward_target'])
    y_val = val_df['forward_target']
    
    feature_names_in_order = list(X_train.columns)

    # Preprocessing
    categorical_features = ['Type'] if 'Type' in feature_names_in_order else []
    numeric_features = [col for col in feature_names_in_order if col not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    neg_ratio = (len(y_train) - y_train.sum()) / y_train.sum() if y_train.sum() > 0 else 1.0

    candidates = {
        "Logistic Regression (Balanced)": Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(class_weight='balanced', random_state=config.RANDOM_SEED, max_iter=1000))
        ]),
        "Random Forest (Regularized)": Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                class_weight='balanced',
                max_depth=8,
                min_samples_leaf=10,
                min_samples_split=20,
                random_state=config.RANDOM_SEED
            ))
        ]),
        "XGBoost (Scale Pos Weight)": Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', XGBClassifier(
                scale_pos_weight=neg_ratio,
                random_state=config.RANDOM_SEED,
                use_label_encoder=False,
                eval_metric='logloss'
            ))
        ]),
        "SMOTE + Logistic Regression": ImbPipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=config.RANDOM_SEED)),
            ('classifier', LogisticRegression(random_state=config.RANDOM_SEED, max_iter=1000))
        ])
    }

    thresholds = np.linspace(0, 1, 101)
    
    candidate_results = {}
    pr_curve_data = {}
    best_candidate_name = None
    best_amber_recall = -1
    best_amber_thresh = None
    best_red_thresh = None
    best_amber_precision = None
    best_red_precision = None
    best_model = None

    for name, pipeline in candidates.items():
        print(f"Training {name}...")
        pipeline.fit(X_train, y_train)
        
        # Predict probabilities on validation
        y_val_probs = pipeline.predict_proba(X_val)[:, 1]
        
        # Save PR curve
        precisions, recalls, pr_thresholds = precision_recall_curve(y_val, y_val_probs)
        pr_curve_data[name] = {
            "thresholds": np.append(pr_thresholds, [1.0]).tolist(),
            "precisions": precisions.tolist(),
            "recalls": recalls.tolist()
        }
        
        cand_amber_thresh = None
        cand_red_thresh = None
        
        # Find LOWEST threshold where precision >= floor
        for t in thresholds:
            y_val_pred = (y_val_probs >= t).astype(int)
            prec = precision_score(y_val, y_val_pred, zero_division=0)
            
            if cand_amber_thresh is None and prec >= config.AMBER_PRECISION_FLOOR:
                cand_amber_thresh = t
                
            if cand_red_thresh is None and prec >= config.RED_PRECISION_FLOOR and cand_amber_thresh is not None and t > cand_amber_thresh:
                cand_red_thresh = t
                
        if cand_red_thresh is None or cand_amber_thresh is None:
            print(f"WARNING: {name} could not reach required precision floors. Excluding from selection.", file=sys.stderr)
            candidate_results[name] = {
                "reached_floors": False,
            }
            continue
            
        if cand_amber_thresh < 1e-5:
            print(f"WARNING: {name} amber_threshold is ~0. Floor was already satisfied at baseline — this threshold is not discriminating.")
        if cand_red_thresh < 1e-5:
            print(f"WARNING: {name} red_threshold is ~0. Floor was already satisfied at baseline — this threshold is not discriminating.")
            
        assert cand_red_thresh > cand_amber_thresh, f"{name}: red_threshold ({cand_red_thresh}) not > amber_threshold ({cand_amber_thresh})"
        
        # Evaluate performance at chosen thresholds
        y_val_pred_amber = (y_val_probs >= cand_amber_thresh).astype(int)
        y_val_pred_red = (y_val_probs >= cand_red_thresh).astype(int)
        
        cand_amber_rec = recall_score(y_val, y_val_pred_amber, zero_division=0)
        cand_amber_prec = precision_score(y_val, y_val_pred_amber, zero_division=0)
        
        cand_red_rec = recall_score(y_val, y_val_pred_red, zero_division=0)
        cand_red_prec = precision_score(y_val, y_val_pred_red, zero_division=0)

        candidate_results[name] = {
            "reached_floors": True,
            "amber_threshold": float(cand_amber_thresh),
            "amber_precision": float(cand_amber_prec),
            "amber_recall": float(cand_amber_rec),
            "red_threshold": float(cand_red_thresh),
            "red_precision": float(cand_red_prec),
            "red_recall": float(cand_red_rec)
        }
        
        if cand_amber_rec > best_amber_recall:
            best_amber_recall = cand_amber_rec
            best_amber_thresh = cand_amber_thresh
            best_amber_precision = cand_amber_prec
            best_red_thresh = cand_red_thresh
            best_red_precision = cand_red_prec
            best_candidate_name = name
            best_model = pipeline

    print("\n--- Candidate Comparison (Recall-First Dual Tier) ---")
    for name, res in candidate_results.items():
        if res["reached_floors"]:
            print(f"{name:35} | Amber Thresh: {res['amber_threshold']:.3f} (Rec: {res['amber_recall']:.3f}, Prec: {res['amber_precision']:.3f}) | Red Thresh: {res['red_threshold']:.3f} (Prec: {res['red_precision']:.3f})")
        else:
            print(f"{name:35} | FAILED TO REACH PRECISION FLOORS")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Rename old comparison
    old_comp = config.MODELS_DIR / "candidate_comparison.json"
    if old_comp.exists():
        shutil.move(old_comp, config.MODELS_DIR / "candidate_comparison_single_threshold.json")
        
    with open(config.MODELS_DIR / "candidate_comparison.json", "w") as f:
        json.dump(candidate_results, f, indent=2)
        
    with open(config.MODELS_DIR / "pr_curve_data.json", "w") as f:
        json.dump(pr_curve_data, f, indent=2)

    if best_candidate_name is None:
        raise RuntimeError("No candidate could reach the precision floors. Cannot select a final model.")

    print(f"\nSelected Model: {best_candidate_name} with highest Amber Recall: {best_amber_recall:.3f} (Amber Thresh: {best_amber_thresh:.3f}, Red Thresh: {best_red_thresh:.3f})")

    # Save model artifact
    joblib.dump(best_model, config.MODELS_DIR / "sentinel_model.joblib")

    # Save metadata
    metadata = {
        "feature_names_in_order": feature_names_in_order,
        "target_name": "forward_target",
        "horizon_cycles": config.PREDICTION_HORIZON_CYCLES,
        "risk_thresholds": {
            "amber": float(best_amber_thresh),
            "red": float(best_red_thresh)
        },
        "random_seed": config.RANDOM_SEED,
        "training_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_type": best_candidate_name,
        "package_versions": {
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__
        },
        "validation_operating_point_metrics": {
            "amber_precision": float(best_amber_precision),
            "amber_recall": float(best_amber_recall),
            "red_precision": float(best_red_precision)
        }
    }
    
    with open(config.MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    train_and_select()
