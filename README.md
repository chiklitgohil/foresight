# Predictive Maintenance: Early Warning Failure System

## Problem Statement
In industrial milling operations, unanticipated machine failures lead to costly downtime and lost production. This predictive maintenance solution ingests machine sensor telemetry (temperature, rotational speed, torque, and tool wear) to accurately predict whether a machine will fail within its next operational window. By catching failures before they occur while minimizing false alarms, we prioritize maximizing recall while maintaining high precision.

## Dataset
- **Name**: AI4I 2020 Predictive Maintenance Dataset
- **Source URL**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) / [Kaggle](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020)
- **License**: CC BY 4.0 (Creative Commons Attribution 4.0 International)
- **Preprocessing Notes**: Missing values are strictly imputed. A pseudo-time-series chronological ordering is enforced using the `UDI` column to prevent data leakage during rolling window feature engineering.

## Setup Instructions
1. Ensure Python 3.10+ is installed.
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows:
   .\.venv\Scripts\activate
   # Mac/Linux:
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Ensure `ai4i2020.csv` is located in `data/raw/` (download from Kaggle or UCI if missing).

## Training Instructions
To train the final optimized model and output the inference artifact (`model/sentinel_model.joblib`), run:
```bash
python train.py
```
This script handles class imbalance using SMOTE and automatically computes the optimal decision threshold for maximum recall at acceptable precision.

## Evaluation Instructions
To evaluate and compare multiple models (Logistic Regression, Random Forest, XGBoost, MLP Neural Network), run:
```bash
python eval/compare_models.py
```
This generates `eval/comparison_table.csv` and `eval/pr_curves.png`.

## Inference Instructions
To test inference latency and receive a risk band prediction for new telemetry data, run:
```bash
python predict.py
```
*(Ensure `train.py` has been executed first to generate the model artifact).*

## Tech Stack
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn, XGBoost, Imbalanced-Learn (SMOTE)
- **Visualization**: Matplotlib, Seaborn
- **Serialization**: Joblib

## Live Operator Dashboard
To run the live Sentinel Factory Dashboard locally (featuring a real-time simulation engine and live risk-band predictions):
```bash
python -m uvicorn api.main:app --port 8000
```
Then navigate to `http://127.0.0.1:8000` in your browser. The dashboard uses FastAPI to serve both the ML inference API and the glassmorphism UI from a single unified server.
