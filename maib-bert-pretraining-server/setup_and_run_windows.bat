@echo off
rem ============================================================================================
rem  EVERYTHING WITH ONE DOUBLE CLICK on a Windows PC with a small NVIDIA GPU (e.g. GTX 960 4 GB)
rem  using uv:   1. Python environment     2. PyTorch 2.7.1 for CUDA 12.6     3. other libraries
rem              4. GPU test and time estimate (saved in check_setup.txt)     5. the experiment
rem  Every download is retried up to 5 times and falls back to a mirror.
rem  It is safe to run this file again at any time (after an error, a power cut, a restart):
rem  finished steps are skipped and the training continues after its last finished epoch.
rem ============================================================================================
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set UV_HTTP_TIMEOUT=600
set MIRROR=https://mirrors.aliyun.com/pypi/simple
set PY=.venv\Scripts\python.exe

where uv >nul 2>nul
if errorlevel 1 goto no_uv

echo.
echo ===== 1/5  Python environment =====
if exist %PY% goto have_venv
call :retry uv venv --python 3.12
if errorlevel 1 goto failed
:have_venv

echo.
echo ===== 2/5  PyTorch 2.7.1 for CUDA 12.6, about 2.5 GB =====
%PY% -c "import torch,sys; sys.exit(0 if torch.__version__=='2.7.1+cu126' else 1)" >nul 2>nul
if not errorlevel 1 goto have_torch
call :retry uv pip install typing-extensions sympy networkx jinja2 fsspec filelock setuptools --index-url %MIRROR%
if errorlevel 1 goto failed
call :retry uv pip install torch==2.7.1+cu126 --no-deps --index-url https://download.pytorch.org/whl/cu126
if not errorlevel 1 goto have_torch
echo download.pytorch.org did not work, trying the aliyun mirror ...
call :retry uv pip install torch==2.7.1+cu126 --no-deps --find-links https://mirrors.aliyun.com/pytorch-wheels/cu126/ --index-url %MIRROR%
if errorlevel 1 goto failed
:have_torch

echo.
echo ===== 3/5  other libraries =====
call :retry uv pip install -r requirements.txt --index-url %MIRROR%
if errorlevel 1 goto failed
%PY% -c "import torch,sys; print('torch', torch.__version__, '| GPU available:', torch.cuda.is_available()); sys.exit(0 if torch.__version__=='2.7.1+cu126' and torch.cuda.is_available() else 1)"
if errorlevel 1 goto bad_torch

rem the original-BERT results of the paper (seeds 1 and 2), if this folder came from GitHub without them
if not exist results mkdir results
for %%R in (bert__ce__seed1 bert__ce__seed2 bert_bilstm_att__wce__seed1 bert_bilstm_att__wce__seed2) do (
  if not exist results\%%R.json if exist ..\maib-thesis\results\%%R.json copy /y ..\maib-thesis\results\%%R.json results\ >nul
)
if exist corpus\external_train.txt goto have_corpus
echo.
echo [note] corpus\ is not in this folder, so the MAIB reports are downloaded and the corpus is built here.
echo        This takes 1 to 2 extra hours. Faster: use maib-server.zip made on the laptop.
:have_corpus

echo.
echo ===== 4/5  GPU test and time estimate, also saved in check_setup.txt =====
%PY% check_setup.py --batch-size 8 > check_setup.txt 2>&1
if not errorlevel 1 goto check_ok
findstr /c:"[missing]" check_setup.txt >nul
if not errorlevel 1 goto check_skipped
echo The check failed. Trying again with the Hugging Face mirror ...
set HF_ENDPOINT=https://hf-mirror.com
%PY% check_setup.py --batch-size 8 > check_setup.txt 2>&1
if not errorlevel 1 goto check_ok
findstr /c:"[missing]" check_setup.txt >nul
if not errorlevel 1 goto check_skipped
goto check_failed
:check_ok
type check_setup.txt
goto run
:check_skipped
type check_setup.txt
echo [note] no time estimate yet: the corpus does not exist. The run builds it first.

:run
echo.
echo ===== 5/5  the experiment; the log is run_full.log =====
powercfg /change standby-timeout-ac 0 >nul 2>nul
powercfg /change hibernate-timeout-ac 0 >nul 2>nul
%PY% run_pipeline.py --corpora task external --external-epochs 3 --mlm-batch-size 8 --log run_full.log %*
echo.
echo Finished or stopped. Results: reports\comparison.md   Log: run_full.log
echo To continue after a stop, double-click this file again.
pause
exit /b 0

:no_uv
echo [error] uv is not installed. In a Command Prompt run:
echo   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
echo then open a NEW window and start this file again.
pause
exit /b 1

:bad_torch
echo [error] PyTorch does not see the GPU. Check that nvidia-smi works and shows a driver version of 560
echo         or newer; if not, install the newest driver for this card, restart, and run this file again.
pause
exit /b 1

:check_failed
type check_setup.txt
echo.
echo [error] The check failed, see the messages above. They are also in check_setup.txt.
pause
exit /b 1

:failed
echo.
echo [error] A download failed 5 times. Check the internet connection and double-click this file again:
echo         what was already downloaded is kept.
pause
exit /b 1

rem ---- run a command, up to 5 times, waiting 30 seconds between tries
:retry
set TRIES=0
:retry_again
set /a TRIES+=1
%*
if not errorlevel 1 exit /b 0
if %TRIES% geq 5 exit /b 1
echo ... failed, try %TRIES% of 5; waiting 30 seconds ...
timeout /t 30 /nobreak >nul
goto retry_again
