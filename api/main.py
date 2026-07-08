from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
from pathlib import Path
import sys
import random

# Setup paths to import our predict script
root_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(root_dir))

from predict import predict_risk_band

app = FastAPI(title="Foresight Factory Dashboard API")

# Serve static files for the frontend
static_dir = root_dir / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Load raw dataset for streaming simulation
raw_data_path = root_dir / "data" / "raw" / "ai4i2020.csv"
try:
    df_raw = pd.read_csv(raw_data_path)
except Exception as e:
    df_raw = None
    print(f"Warning: Could not load raw data for simulation: {e}")

class Telemetry(BaseModel):
    machine_type: str  # 'L', 'M', or 'H'
    air_temp: float
    process_temp: float
    rpm: float
    torque: float
    tool_wear: float

@app.post("/api/predict")
async def predict(data: Telemetry):
    # Convert input to the pandas DataFrame format expected by the model
    df_input = pd.DataFrame({
        "Type": [data.machine_type],
        "Air temperature [K]": [data.air_temp],
        "Process temperature [K]": [data.process_temp],
        "Rotational speed [rpm]": [data.rpm],
        "Torque [Nm]": [data.torque],
        "Tool wear [min]": [data.tool_wear]
    })
    
    model_path = root_dir / "model" / "sentinel_model.joblib"
    try:
        bands, latency = predict_risk_band(df_input, model_path)
        risk_str = bands[0]
        
        # Parse the output string "High Risk -> MAINTENANCE REQUIRED: ..."
        risk_level = risk_str.split(" -> ")[0]
        recommendation = risk_str.split(" -> ")[1] if " -> " in risk_str else ""
        
        return {
            "risk_level": risk_level,
            "recommendation": recommendation,
            "latency_ms": latency
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/stream")
async def stream_data():
    """
    Returns a random row from the raw dataset to simulate live telemetry.
    Biased to return failures 10% of the time for the demo.
    """
    if df_raw is None:
        return {"error": "Raw data not loaded"}
        
    if random.random() < 0.1:
        # Pick a failure row
        sample = df_raw[df_raw["Machine failure"] == 1].sample(1).iloc[0]
    else:
        # Pick a healthy row
        sample = df_raw[df_raw["Machine failure"] == 0].sample(1).iloc[0]
        
    # Determine True Failure Mode (for the Incident Log display)
    true_mode = "None"
    if sample["TWF"] == 1: true_mode = "Tool Wear (TWF)"
    elif sample["HDF"] == 1: true_mode = "Heat Dissipation (HDF)"
    elif sample["PWF"] == 1: true_mode = "Power Failure (PWF)"
    elif sample["OSF"] == 1: true_mode = "Overstrain (OSF)"
    elif sample["RNF"] == 1: true_mode = "Random Failure (RNF)"
        
    return {
        "product_id": sample["Product ID"],
        "machine_type": sample["Type"],
        "air_temp": float(sample["Air temperature [K]"]),
        "process_temp": float(sample["Process temperature [K]"]),
        "rpm": float(sample["Rotational speed [rpm]"]),
        "torque": float(sample["Torque [Nm]"]),
        "tool_wear": float(sample["Tool wear [min]"]),
        "true_failure_mode": true_mode
    }

@app.get("/api/fleet")
async def fleet_data():
    """
    Returns predictions for 100 random machines to populate the fleet heatmap.
    """
    if df_raw is None:
        return {"error": "Raw data not loaded"}
        
    # Sample 100 rows (bias slightly to have a few failures so it looks good)
    failures = df_raw[df_raw["Machine failure"] == 1].sample(5, replace=True)
    healthy = df_raw[df_raw["Machine failure"] == 0].sample(95, replace=True)
    sample_df = pd.concat([failures, healthy]).sample(frac=1) # Shuffle
    
    model_path = root_dir / "model" / "sentinel_model.joblib"
    try:
        bands, _ = predict_risk_band(sample_df, model_path)
        
        fleet = []
        for i, (_, row) in enumerate(sample_df.iterrows()):
            risk_str = bands[i]
            risk_level = risk_str.split(" -> ")[0]
            
            fleet.append({
                "product_id": row["Product ID"],
                "machine_type": row["Type"],
                "risk_level": risk_level
            })
            
        return {"fleet": fleet}
    except Exception as e:
        return {"error": str(e)}

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    html_path = root_dir / "static" / "index.html"
    with open(html_path, "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/heatmap", response_class=HTMLResponse)
async def serve_heatmap():
    html_path = root_dir / "static" / "heatmap.html"
    with open(html_path, "r") as f:
        return HTMLResponse(content=f.read())
