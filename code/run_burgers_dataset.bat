@echo off
setlocal
cd /d "%~dp0"

if not exist ".tmp" mkdir ".tmp"
set "TEMP=%~dp0.tmp"
set "TMP=%~dp0.tmp"

".venv\Scripts\python.exe" "demo_burgers_dataset.py" --preset balanced %*
endlocal
