@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo .venv not found. Run setup.bat first.
    pause
    exit /b 1
)

set "PATH=%CD%\.venv\Lib\site-packages\nvidia\cudnn\bin;%CD%\.venv\Lib\site-packages\nvidia\cublas\bin;%PATH%"

".venv\Scripts\colmap-mask.exe"
