@echo off
rem Runs the experiment WITHOUT a window, for a Windows Scheduled Task, so it keeps running after you
rem disconnect or sign out of Remote Desktop (README, "Scheduled Task"). Same settings as run_windows.bat
rem (GPU with 4 GB). Everything is written to run_full.log; errors before the log starts go to
rem background_output.txt. Safe to start again at any time: finished work is skipped.
cd /d "%~dp0"
set PYTHONUTF8=1
if exist hf_mirror.txt set HF_ENDPOINT=https://hf-mirror.com
powercfg /change standby-timeout-ac 0 >nul 2>nul
powercfg /change hibernate-timeout-ac 0 >nul 2>nul
.venv\Scripts\python.exe run_pipeline.py --corpora task external --external-epochs 3 --mlm-batch-size 8 --log run_full.log >> background_output.txt 2>&1
