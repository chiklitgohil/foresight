import sys
import time
import numpy as np
from pathlib import Path
import pandas as pd
import joblib

root_dir = Path(__file__).resolve().parent

def predict_risk_band(input_data: pd.DataFrame, artifact_path: Path):
    """
    Given raw input data, outputs the risk band using the imblearn pipeline.
    """
    # 1. Load pipeline artifact
    pipeline = joblib.load(artifact_path)
    
    # 2. Engineer features matching notebook 03 logic
    df = input_data.copy()
    
    # Type One-hot encoding (L, M, H) - dummy logic
    # In the notebook, 'Type_L' and 'Type_M' were created via drop_first=True
    # Since we only get 1 row often, we must ensure columns exist manually
    if 'Type' in df.columns:
        df['Type_L'] = (df['Type'] == 'L').astype(int)
        df['Type_M'] = (df['Type'] == 'M').astype(int)
        df = df.drop(columns=['Type'])
    
    # Physics features
    df["Temp_Diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["Power"] = df["Torque [Nm]"] * (df["Rotational speed [rpm]"] * 2 * np.pi / 60)
    df["Tool_Wear_Rate"] = df["Tool wear [min]"] * df["Torque [Nm]"]
    
    # Ensure column order matches the training set exactly
    expected_features = [
        'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]',
        'Torque [Nm]', 'Tool wear [min]', 'Type_L', 'Type_M', 'Temp_Diff',
        'Power', 'Tool_Wear_Rate'
    ]
    
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0
            
    X_input = df[expected_features]
    
    # 3. Predict probability using the full pipeline
    start_time = time.time()
    # The pipeline applies scaling internally
    probs = pipeline.predict_proba(X_input)[:, 1]
    latency_ms = (time.time() - start_time) * 1000
    
    # 4. Apply threshold and risk bands
    # We use a default threshold of 0.5, or a tuned threshold if we knew it.
    # In Notebook 03, we didn't tune threshold, we just used predict() which defaults to 0.5
    threshold = 0.5 
    
    results = []
    for p in probs:
        if p >= threshold:
            results.append("High Risk -> MAINTENANCE REQUIRED: Schedule immediate downtime for inspection/replacement")
        elif p >= (threshold * 0.5): 
            results.append("Medium Risk -> ACTION RECOMMENDED: Inspect machine on next shift")
        else:
            results.append("Low Risk -> SYSTEM HEALTHY: Continue normal operations")
            
    return results, latency_ms

if __name__ == "__main__":
    model_path = root_dir / "model" / "foresight_model.joblib"
    
    dummy_input = pd.DataFrame({
        "Type": ["L"],
        "Air temperature [K]": [298.1],
        "Process temperature [K]": [308.6],
        "Rotational speed [rpm]": [1551],
        "Torque [Nm]": [42.8],
        "Tool wear [min]": [0]
    })
    
    print("Running inference benchmark...")
    bands, latency = predict_risk_band(dummy_input, model_path)
    print(f"Predicted Band: {bands[0]}")
    print(f"Inference Latency: {latency:.2f} ms")
