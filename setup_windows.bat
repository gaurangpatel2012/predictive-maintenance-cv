@echo off
rem One-time setup on Windows with an NVIDIA GPU. Double-click or run from this folder.
cd /d "%~dp0"
echo === Creating Python environment (.venv) ===
py -3.12 -m venv .venv
if errorlevel 1 (
  echo Python 3.12 not found. Install it from https://www.python.org/downloads/ and tick "Add python.exe to PATH".
  pause
  exit /b 1
)
.venv\Scripts\python -m pip install --upgrade pip
echo === Installing PyTorch with CUDA (about 3 GB download) ===
.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
echo === Installing the other packages ===
.venv\Scripts\pip install -r requirements.txt
echo === Checking the project ===
.venv\Scripts\python scripts\check_setup.py
pause
