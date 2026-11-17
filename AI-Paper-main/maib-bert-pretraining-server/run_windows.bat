@echo off
rem Start OR continue the experiment on a Windows PC with a 4 GB GPU (double-click).
rem Stage 1: TAPT, then DAPT with at most 3 epochs; stage 2: all fine-tunings; then the comparison table.
rem Everything shown here is also written to run_full.log.
rem To stop: press Ctrl+C in this window (or close it). To continue: double-click this file again;
rem finished work is skipped and an interrupted run continues after its last finished epoch.
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
if not exist .venv\Scripts\activate.bat (
  echo Run install_windows.bat first.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat

rem keep Windows awake while plugged in (sleep would pause the training)
powercfg /change standby-timeout-ac 0 >nul 2>nul
powercfg /change hibernate-timeout-ac 0 >nul 2>nul

python run_pipeline.py --corpora task external --external-epochs 3 --mlm-batch-size 8 --log run_full.log %*
echo.
echo Finished or stopped. The results are in reports\comparison.md ; the log is run_full.log
pause
