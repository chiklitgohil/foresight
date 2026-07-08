import pandas as pd
import numpy as np
import json
from src import config

def compute_rolling_features(df, column, windows):
    # Ensure no forward-looking data by using min_periods=1 and closed='right' (default)
    # The rolling window includes the current row and (window-1) past rows.
    for w in windows:
        roll = df[column].rolling(window=w, min_periods=1)
        df[f"{column}_roll_{w}_mean"] = roll.mean()
        df[f"{column}_roll_{w}_std"] = roll.std().fillna(0)
        df[f"{column}_roll_{w}_min"] = roll.min()
        df[f"{column}_roll_{w}_max"] = roll.max()
        
        # Trend: current value minus rolling mean
        df[f"{column}_roll_{w}_trend"] = df[column] - df[f"{column}_roll_{w}_mean"]
    return df

def compute_spectral_features(df, column, window=50):
    # ASSUMPTION: FFT on these 50-cycle windows is used as a proxy signal for vibration, 
    # since the AI4I dataset has no true vibration channel.
    
    def dominant_freq_and_energy(series):
        if len(series) < 2:
            return pd.Series({'freq': 0.0, 'energy': 0.0})
        # Subtract mean to remove DC component
        centered = series - series.mean()
        fft_vals = np.fft.rfft(centered)
        freqs = np.fft.rfftfreq(len(series))
        
        energy = np.sum(np.abs(fft_vals)**2)
        dom_freq = freqs[np.argmax(np.abs(fft_vals))] if len(fft_vals) > 0 else 0.0
        return pd.Series({'freq': dom_freq, 'energy': energy})
    
    # We use rolling.apply if possible, but for multiple outputs it's tricky.
    # Alternatively, we can use a rolling window approach manually or with a trick.
    # To strictly ensure backward-looking, we take the last `window` rows ending at `t`.
    
    freq_col = np.zeros(len(df))
    energy_col = np.zeros(len(df))
    
    vals = df[column].values
    for i in range(len(df)):
        start_idx = max(0, i - window + 1)
        slice_vals = vals[start_idx:i+1]
        
        if len(slice_vals) < 2:
            freq_col[i] = 0.0
            energy_col[i] = 0.0
        else:
            centered = slice_vals - np.mean(slice_vals)
            fft_vals = np.fft.rfft(centered)
            energy_col[i] = np.sum(np.abs(fft_vals)**2)
            if len(fft_vals) > 0:
                freqs = np.fft.rfftfreq(len(slice_vals))
                freq_col[i] = freqs[np.argmax(np.abs(fft_vals))]
            else:
                freq_col[i] = 0.0

    df[f"{column}_fft_dom_freq_{window}"] = freq_col
    df[f"{column}_fft_energy_{window}"] = energy_col
    return df

def engineer_features(df, n_train):
    # Ensure DataFrame is sorted by UDI
    df = df.sort_values('UDI').reset_index(drop=True)
    
    # 1. torque_x_speed and high-torque/low-speed flag
    df['torque_x_speed'] = df['Torque [Nm]'] * df['Rotational speed [rpm]']
    
    torque_75 = df['Torque [Nm]'].iloc[:n_train].quantile(0.75)
    speed_25 = df['Rotational speed [rpm]'].iloc[:n_train].quantile(0.25)
    df['high_torque_low_speed'] = ((df['Torque [Nm]'] > torque_75) & (df['Rotational speed [rpm]'] < speed_25)).astype(int)
    
    # 2. temp_diff
    df['temp_diff'] = df['Process temperature [K]'] - df['Air temperature [K]']
    
    # 3. Tool wear high-wear flag
    wear_75 = df['Tool wear [min]'].iloc[:n_train].quantile(0.75)
    df['tool_wear_high_flag'] = (df['Tool wear [min]'] > wear_75).astype(int)
    
    df['Tool wear [min]_roll_10_max'] = df['Tool wear [min]'].rolling(window=10, min_periods=1).max()
    df['Tool wear [min]_roll_10_trend'] = df['Tool wear [min]'] - df['Tool wear [min]'].rolling(window=10, min_periods=1).mean()
    
    # temp_diff rolling threshold crossing (e.g., > 90th percentile)
    temp_diff_90 = df['temp_diff'].iloc[:n_train].quantile(0.90)
    temp_diff_high = (df['temp_diff'] > temp_diff_90).astype(int)
    df['temp_diff_high_crossings_10'] = temp_diff_high.rolling(window=10, min_periods=1).sum()
    
    # 4. 10- and 50-cycle rolling features
    for col in ['Torque [Nm]', 'Rotational speed [rpm]', 'temp_diff', 'torque_x_speed']:
        df = compute_rolling_features(df, col, config.ROLLING_WINDOWS)
    
    # 5. FFT features
    for col in ['Torque [Nm]', 'Rotational speed [rpm]', 'torque_x_speed']:
        df = compute_spectral_features(df, col, window=50)
        
    # 6. per-Type (L/M/H) z-score baselines using ONLY training split
    # Pass n_train as an explicit parameter; compute stats, apply to all.
    type_stats = {}
    train_slice = df.iloc[:n_train]
    
    cols_to_zscore = ['Torque [Nm]', 'Rotational speed [rpm]', 'temp_diff', 'Tool wear [min]']
    
    for machine_type in ['L', 'M', 'H']:
        type_data = train_slice[train_slice['Type'] == machine_type]
        stats = {}
        for col in cols_to_zscore:
            if len(type_data) > 0:
                stats[col] = {'mean': type_data[col].mean(), 'std': type_data[col].std()}
            else:
                stats[col] = {'mean': 0.0, 'std': 1.0}
        type_stats[machine_type] = stats
        
    # Save to train_stats.json
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.MODELS_DIR / "train_stats.json", "w") as f:
        json.dump(type_stats, f, indent=2)
        
    # Apply z-scores
    for col in cols_to_zscore:
        z_col = f"{col}_zscore"
        df[z_col] = 0.0
        for machine_type in ['L', 'M', 'H']:
            mask = df['Type'] == machine_type
            mean = type_stats[machine_type][col]['mean']
            std = type_stats[machine_type][col]['std']
            if std == 0:
                std = 1.0
            df.loc[mask, z_col] = (df.loc[mask, col] - mean) / std

    # 7. Forward target
    # label = 1 if any failure occurs in the next PREDICTION_HORIZON_CYCLES rows
    # This looks forward, but it's the TARGET, not a feature. Features must only look backward.
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=config.PREDICTION_HORIZON_CYCLES)
    df['forward_target'] = df[config.TARGET_COLUMN].rolling(window=indexer, min_periods=1).max().fillna(0).astype(int)
    
    return df

def split_and_save(df, n_train, n_val):
    # Exclude leakage columns from the returned model-input feature set
    # Keep them in a separate diagnostic-only column set for evaluate.py
    
    feature_cols = [c for c in df.columns if c not in config.LEAKAGE_COLUMNS and c != 'forward_target' and c != config.TARGET_COLUMN]
    
    diagnostic_cols = config.LEAKAGE_COLUMNS + ['forward_target', config.TARGET_COLUMN]
    
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train+n_val]
    test_df = df.iloc[n_train+n_val:]
    
    train_features = train_df[feature_cols + ['forward_target']]
    val_features = val_df[feature_cols + ['forward_target']]
    test_features = test_df[feature_cols + ['forward_target']]
    
    test_diagnostics = test_df[diagnostic_cols]
    
    train_features.to_csv(config.DATA_PROCESSED_DIR / "train_features.csv", index=False)
    val_features.to_csv(config.DATA_PROCESSED_DIR / "validation_features.csv", index=False)
    test_features.to_csv(config.DATA_PROCESSED_DIR / "test_features.csv", index=False)
    test_diagnostics.to_csv(config.DATA_PROCESSED_DIR / "test_diagnostics.csv", index=False)
    
    # Replay rows need to contain all info for simulation (features + diagnostics)
    df.iloc[n_train+n_val:].to_csv(config.DATA_PROCESSED_DIR / "replay_rows.csv", index=False)
    
    print("Feature engineering complete.")

if __name__ == "__main__":
    # In a full run, we'd load the full sorted dataset from raw or recombine splits
    # Since data_loader outputted ordered chunks, we can just load the original raw, sort, and process
    csv_path = config.DATA_RAW_DIR / "ai4i2020.csv"
    df = pd.read_csv(csv_path)
    df = df.sort_values(by="UDI").reset_index(drop=True)
    
    total_rows = len(df)
    n_train = int(total_rows * config.TRAIN_FRACTION)
    n_val = int(total_rows * config.VALIDATION_FRACTION)
    
    df_engineered = engineer_features(df, n_train)
    split_and_save(df_engineered, n_train, n_val)
