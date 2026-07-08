"""
Foresight Model Training Pipeline
Loads raw AI4I data, engineers simple features, trains a Random Forest model, and saves it.
"""

from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

def main():
    root_dir = Path(__file__).resolve().parents[1]
    data_path = root_dir / "data" / "raw" / "ai4i2020.csv"
    model_dir = root_dir / "models"
    model_dir.mkdir(exist_ok=True)
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, encoding="utf-8-sig")
    
    # Target and Features
    target = "Machine failure"
    features = [
        "Type", 
        "Air temperature [K]", 
        "Process temperature [K]", 
        "Rotational speed [rpm]", 
        "Torque [Nm]", 
        "Tool wear [min]"
    ]
    
    X = df[features]
    y = df[target]
    
    print(f"Data loaded: {len(X)} rows. Failures: {y.sum()}")

    # Define preprocessing
    numeric_features = [
        "Air temperature [K]", 
        "Process temperature [K]", 
        "Rotational speed [rpm]", 
        "Torque [Nm]", 
        "Tool wear [min]"
    ]
    categorical_features = ["Type"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    # Define model with class weighting for imbalanced data
    model = RandomForestClassifier(
        n_estimators=100, 
        class_weight="balanced", 
        random_state=42
    )

    # Bundle preprocessing and modeling code in a pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    print("Training the Foresight model...")
    pipeline.fit(X, y)
    
    model_path = model_dir / "foresight_model.joblib"
    print(f"Saving model to {model_path}...")
    joblib.dump(pipeline, model_path)
    
    # Save the feature columns we used for the app
    metadata = {
        "features": features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features
    }
    joblib.dump(metadata, model_dir / "foresight_metadata.joblib")
    
    print("Training complete!")

if __name__ == "__main__":
    main()
