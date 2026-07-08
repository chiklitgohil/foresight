"""
Inference Script
Loads the trained model artifact, takes raw telemetry inputs, and outputs a risk band.
Includes a latency benchmark for scoring.
"""

import sys
import time
from pathlib import Path
import pandas as pd
import joblib

root_dir = Path(__file__).resolve().parent
sys.path.append(str(root_dir))

from features.engineer import build_features

def predict_risk_band(input_data: pd.DataFrame, artifact_path: Path):
    """
    Given raw input data (can be multiple rows, but typical use is 1 row),
    outputs the risk band.
    """
    # 1. Load artifact
    artifact = joblib.load(artifact_path)
    scaler = artifact["scaler"]
    model = artifact["model"]
    threshold = artifact["optimal_threshold"]
    expected_features = artifact["features"]
    
    # 2. Engineer features
    # (Note: In a true streaming environment, rolling features require caching previous state.
    # For this hackathon, we assume the input_data might contain a small buffer of history 
    # if rolling features are needed, or we just compute on the available row which 
    # degrades rolling stats to window=1 if only 1 row is passed. This is acceptable for demo.)
    df_features = build_features(input_data)
    
    # Ensure all expected columns are present (fill missing with 0 for dummy encoding robustness)
    for col in expected_features:
        if col not in df_features.columns:
            df_features[col] = 0
            
    X_input = df_features[expected_features]
    
    # 3. Scale and predict
    start_time = time.time()
    X_scaled = scaler.transform(X_input)
    probs = model.predict_proba(X_scaled)[:, 1]
    
    latency_ms = (time.time() - start_time) * 1000
    
    # 4. Apply threshold and risk bands
    # Constraint: output a risk band (Low/Medium/High) rather than raw probability
    results = []
    for p in probs:
        if p >= threshold:
            results.append("High Risk -> MAINTENANCE REQUIRED: Schedule immediate downtime for inspection/replacement")
        elif p >= (threshold * 0.5): 
            # Simple heuristic for Medium risk
            results.append("Medium Risk -> ACTION RECOMMENDED: Inspect machine on next shift")
        else:
            results.append("Low Risk -> SYSTEM HEALTHY: Continue normal operations")
            
    return results, latency_ms

if __name__ == "__main__":
    model_path = root_dir / "model" / "sentinel_model.joblib"
    
    # Create a dummy row for testing
    dummy_input = pd.DataFrame({
        "Type": ["L"],
        "Air temperature [K]": [298.1],
        "Process temperature [K]": [308.6],
        "Rotational speed [rpm]": [1551],
        "Torque [Nm]": [42.8],
        "Tool wear [min]": [0]
    })
    
    print("Running inference benchmark...")
    
    try:
        bands, latency = predict_risk_band(dummy_input, model_path)
        print(f"Predicted Band: {bands[0]}")
        print(f"Inference Latency: {latency:.2f} ms")
    except FileNotFoundError:
        print("Model artifact not found. Run train.py first.")
