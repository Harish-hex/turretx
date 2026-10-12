#!/usr/bin/env bash
# TurretX setup for the Mac (M1) dev machine. Runs on recorded clips using the Apple GPU (MPS).
# Run from the repo root:  bash setup_mac.sh
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "Creating Python 3.12 virtualenv in .venv (system Python 3.14 is too new for torch/ultralytics)..."
uv python install 3.12
uv venv --python 3.12 --seed .venv
source .venv/bin/activate

echo "Installing PyTorch (default macOS build includes MPS)..."
uv pip install torch torchvision

echo "Installing project requirements..."
uv pip install -r requirements.txt

echo; echo "Checking environment..."
python scripts/check_env.py

echo; echo "Pre-downloading models and lazy dependencies..."
python scripts/prefetch.py

echo; echo "Done. Activate later with:  source .venv/bin/activate"
