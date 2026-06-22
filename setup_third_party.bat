@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
    if errorlevel 1 exit /b 1
)

".venv\Scripts\python.exe" -m pip install -e .[inference]
if errorlevel 1 exit /b 1

".venv\Scripts\colmap-mask-setup-third-party.exe" %*
if errorlevel 1 exit /b 1

echo third_party setup complete.
pause
