"""
Data Loading & Preprocessing
Loads the AI4I 2020 dataset, cleans it, handles missing values, and enforces
time-series integrity by sorting by UDI (Unique Data Identifier) which acts as a 
chronological proxy for this dataset.
"""

import pandas as pd
from pathlib import Path

def load_and_preprocess(filepath: str | Path) -> pd.DataFrame:
    """
    Load raw dataset, enforce schema, and handle missing values.
    """
    df = pd.read_csv(filepath, encoding="utf-8-sig")
    
    # Handle missing values if any appear as "?"
    df = df.replace("?", pd.NA)
    
    # Convert appropriate columns to numeric
    numeric_cols = [
        "Air temperature [K]", 
        "Process temperature [K]", 
        "Rotational speed [rpm]", 
        "Torque [Nm]", 
        "Tool wear [min]"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # Impute missing values with mean (forward-fill could be used for strict time-series, 
    # but given the dataset's nature, mean is a safe fallback if NA exists).
    # However, to strictly avoid future leakage, we should forward-fill.
    # AI4I typically has no missing values, but we handle it just in case.
    df[numeric_cols] = df[numeric_cols].ffill().bfill() 
    
    # ENFORCE TIME-SERIES INTEGRITY:
    # We must sort by UDI to ensure that any rolling window calculations
    # performed later ONLY use past data for time t, never future data.
    # This prevents data leakage when simulating a real-time predictive maintenance stream.
    df = df.sort_values(by="UDI").reset_index(drop=True)
    
    return df

if __name__ == "__main__":
    # Test execution
    root_dir = Path(__file__).resolve().parents[1]
    raw_path = root_dir / "data" / "raw" / "ai4i2020.csv"
    df = load_and_preprocess(raw_path)
    print(f"Data preprocessed successfully. Shape: {df.shape}")
    print(df.head())
