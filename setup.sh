#!/usr/bin/env bash
set -euo pipefail

if [[ -f ".venv/Scripts/python.exe" || -f ".venv/bin/python" ]]; then
  echo "Using existing .venv"
elif [[ -n "${PYTHON:-}" ]]; then
  "$PYTHON" -m venv .venv
elif [[ -f "$HOME/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe" ]]; then
  "$HOME/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe" -m venv .venv
elif command -v py >/dev/null 2>&1; then
  py -3 -m venv .venv
elif command -v python >/dev/null 2>&1; then
  python -m venv .venv
elif command -v python3 >/dev/null 2>&1; then
  python3 -m venv .venv
else
  echo "Python was not found on PATH. Install Python 3.14+ or run from a shell where python is available."
  exit 1
fi

if [[ -f ".venv/Scripts/python.exe" ]]; then
  VENV_PYTHON=".venv/Scripts/python.exe"
elif [[ -f ".venv/bin/python" ]]; then
  VENV_PYTHON=".venv/bin/python"
else
  echo "Virtual environment python executable was not found."
  exit 1
fi

if [[ -f ".venv/Scripts/activate" ]]; then
  # Windows Git Bash
  source .venv/Scripts/activate
else
  source .venv/bin/activate
fi

"$VENV_PYTHON" -m pip install --upgrade pip
"$VENV_PYTHON" -m pip install -r requirements.txt
"$VENV_PYTHON" -c "import pandas, numpy, sklearn, matplotlib, seaborn, imblearn, joblib, dash; print('Environment OK')"
