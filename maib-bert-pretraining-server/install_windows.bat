@echo off
rem One-time setup on a Windows PC with an NVIDIA GPU (double-click, or run in a Command Prompt).
rem It creates the Python environment in .venv, installs PyTorch for CUDA 12.6 (version 2.7.1: the build
rem that still supports older cards such as the GTX 900 / 1000 series), installs the other libraries and
rem then runs check_setup.py, which tests the GPU and estimates how long the experiment takes here.
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1

where python >nul 2>nul
if errorlevel 1 (
  echo [error] Python is not installed. Install Python 3.12 from https://www.python.org/downloads/windows/
  echo         and tick "Add python.exe to PATH" in the installer. Then run this file again.
  pause
  exit /b 1
)
python -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)"
if errorlevel 1 (
  echo [error] Python 3.10 to 3.13 is needed. Installed:
  python --version
  pause
  exit /b 1
)

echo === NVIDIA driver ===
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
if errorlevel 1 (
  echo [error] nvidia-smi does not work: install the newest NVIDIA driver for this card from
  echo         https://www.nvidia.com/Download/index.aspx , restart Windows and run this file again.
  pause
  exit /b 1
)

if not exist .venv python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip

echo === PyTorch 2.7.1 for CUDA 12.6 (about 2.5 GB) ===
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
if errorlevel 1 (
  echo download.pytorch.org failed, trying the aliyun mirror ...
  pip install torch==2.7.1 -f https://mirrors.aliyun.com/pytorch-wheels/cu126/ -i https://mirrors.aliyun.com/pypi/simple
)
if errorlevel 1 (
  echo [error] PyTorch could not be installed. Check the internet connection and run this file again.
  pause
  exit /b 1
)

echo === other libraries ===
pip install -r requirements.txt
if errorlevel 1 pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
if errorlevel 1 (
  echo [error] The libraries could not be installed. Check the internet connection and run this file again.
  pause
  exit /b 1
)

echo === check and time estimate ===
python check_setup.py --batch-size 8
echo.
echo If every line above starts with [ok], start the experiment with run_windows.bat
pause
