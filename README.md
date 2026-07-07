# SentinelPdM

Predictive maintenance for factory operators that turns machine sensor telemetry into early failure warnings, risk bands, and maintenance actions.

## Problem

SentinelPdM helps operators catch likely machine failures before reactive downtime, because missed failures carry much higher cost than false alarms: lost throughput, damaged equipment, scrap, and potential operator injury.

<!-- fill after final demo script -->

## Dataset

- Name: AI4I 2020 Predictive Maintenance Dataset
- Source: UCI Machine Learning Repository, https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)

<!-- fill after README dataset citation is finalized -->

## Setup

```bash
bash setup.sh
```

<!-- fill after confirming final supported shell/OS commands -->

## Training

<!-- fill after training pipeline exists -->

Expected command:

```bash
python -m src.train
```

## Inference

<!-- fill after inference.py is implemented -->

Expected command/API:

```bash
python -m src.inference
```

## Evaluation

See [RESULTS.md](RESULTS.md).

<!-- fill after training run with headline recall/precision numbers -->

## Dashboard / Demo

<!-- fill after dashboard/app.py is implemented -->

Expected command:

```bash
python -m dashboard.app
```

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). The system loads AI4I raw data, creates leakage-safe ordered pseudo-cycle splits, trains a class-weighted failure classifier, serves risk-banded inference events, and displays operator-safe actions through a simulated dashboard replay.

<!-- fill after architecture changes, if any -->

## Team

<!-- fill team names and roles -->

## License

<!-- fill repository license and dataset license summary -->
