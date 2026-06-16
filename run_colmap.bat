@echo off
setlocal

cd /d "%~dp0"

if "%~1"=="" (
    echo Usage: run_colmap.bat EXPORT_DIR [extra args]
    echo Example: run_colmap.bat C:\path\to\project\exports --overwrite
    exit /b 1
)

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m colmap_mask.tools.run_colmap %*
) else (
    python -m colmap_mask.tools.run_colmap %*
)
