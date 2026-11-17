@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
powercfg /change standby-timeout-ac 0 >nul 2>nul
powercfg /change hibernate-timeout-ac 0 >nul 2>nul
.venv\Scripts\python.exe run_pipeline.py --corpora task external --external-epochs 3 --mlm-batch-size 8 --log run_full.log >> background_output.txt 2>&1