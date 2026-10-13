@echo off
rem Runs every remaining training, evaluation and timing step. Safe to re-run: finished steps are skipped.
cd /d "%~dp0"
.venv\Scripts\python scripts\check_setup.py
if errorlevel 1 (
  pause
  exit /b 1
)
.venv\Scripts\python scripts\run_queue.py
pause
