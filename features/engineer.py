"""
Feature Engineering
Constructs time-series, interaction, and rate-of-change features from raw sensor data.
"""

import pandas as pd
import numpy as np

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a chronologically sorted dataframe (by UDI) and builds predictive features.
    IMPORTANT: We only use rolling windows and shifts that look BACK in time to prevent future leakage.
    """
    df = df.copy()
    
    # 1. Interaction Features (Physics-based)
    # Mechanical Power (Watts): (Torque * RPM * 2 * pi) / 60
    df["Mechanical Power [W]"] = np.round((df["Torque [Nm]"] * df["Rotational speed [rpm]"] * 2 * np.pi) / 60, 4)
    
    # Temperature differential (Heat dissipation proxy)
    df["Temp_Diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    
    # Strain proxy: Torque * Tool Wear
    df["Strain_Proxy"] = df["Torque [Nm]"] * df["Tool wear [min]"]

    # 2. Rolling Time-Series Features
    # Since UDI represents chronological operational cycles, we compute rolling stats 
    # to capture machine degradation over time.
    windows = [5, 10]
    cols_to_roll = [
        "Air temperature [K]", 
        "Process temperature [K]", 
        "Rotational speed [rpm]", 
        "Torque [Nm]", 
        "Tool wear [min]",
        "Mechanical Power [W]",
        "Temp_Diff"
    ]
    
    for w in windows:
        for col in cols_to_roll:
            # Mean over window
            df[f"{col}_roll_mean_{w}"] = df[col].rolling(window=w, min_periods=1).mean()
            # Std over window (captures instability/vibration proxies)
            df[f"{col}_roll_std_{w}"] = df[col].rolling(window=w, min_periods=1).std().fillna(0)
            
    # 3. Rate of Change (Lag Features)
    # How much did the sensor value jump from the previous cycle?
    for col in cols_to_roll:
        df[f"{col}_delta_1"] = df[col].diff(periods=1).fillna(0)
        
    # Categorical Encoding
    # Drop UDI and Product ID as they are unique identifiers and leak information
    # Drop individual failure modes to avoid target leakage (we only predict 'Machine failure')
    leakage_cols = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]
    df = df.drop(columns=[col for col in leakage_cols if col in df.columns])
    
    # One-hot encode 'Type' (L, M, H)
    if "Type" in df.columns:
        df = pd.get_dummies(df, columns=["Type"], drop_first=True)
        
    return df

if __name__ == "__main__":
    pass
