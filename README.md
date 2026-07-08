# Foresight

Foresight is a predictive maintenance application for factory operators that turns machine sensor telemetry into early failure warnings, risk bands, and maintenance actions.

## Overview
Foresight helps operators catch likely machine failures before reactive downtime. It uses a machine learning model to predict failure risk based on air/process temperature, rotational speed, torque, and tool wear.

## Setup

Ensure you have Python installed, then install the requirements:

```bash
pip install -r requirements.txt
```

## Running the Application

This repository has been simplified to make it easy to train and test the model in just two steps.

### 1. Train the Model

Run the training script to load the AI4I dataset, build the pipeline, and save the model:

```bash
python src/train.py
```

This will create a `foresight_model.joblib` artifact in the `models/` directory.

### 2. Launch the Dashboard

Once the model is trained, launch the interactive web dashboard to test the model:

```bash
python app.py
```

Open your browser to `http://127.0.0.1:8050/` to use the Foresight telemetry dashboard. You can input various sensor readings and see the predicted risk band (Green, Amber, or Red) in real-time.
