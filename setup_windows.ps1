# TurretX setup for the Windows RTX 3050 laptop (main/demo machine).
# Run from the repo root in PowerShell:
#   powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
# Needs internet (~3 GB download for CUDA PyTorch). Run it EARLY, ideally on a phone hotspot.

$ErrorActionPreference = "Stop"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "Git not found. Install it from https://git-scm.com/download/win and re-run." -ForegroundColor Red
    exit 1
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv..."
    powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

Write-Host "Creating Python 3.12 virtualenv in .venv ..."
uv python install 3.12
# --seed installs pip into the venv; Ultralytics uses pip to auto-install extras (e.g. CLIP for YOLO-World)
uv venv --python 3.12 --seed .venv

. .\.venv\Scripts\Activate.ps1

Write-Host "Installing PyTorch (CUDA 12.6 build, works on RTX 30-series)..."
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

Write-Host "Installing project requirements..."
uv pip install -r requirements.txt

Write-Host "`nChecking environment..."
python scripts\check_env.py

Write-Host "`nPre-downloading models and lazy dependencies..."
python scripts\prefetch.py

Write-Host "`nDone. Activate later with:  .\.venv\Scripts\Activate.ps1" -ForegroundColor Green
