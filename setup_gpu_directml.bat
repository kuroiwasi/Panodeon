@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    py -3 -m venv .venv
)

".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -e .[dev]
".venv\Scripts\python.exe" -m pip uninstall -y onnxruntime onnxruntime-directml onnxruntime-gpu
".venv\Scripts\python.exe" -m pip install onnxruntime-directml

echo DirectML/NVIDIA/AMD ONNX Runtime installed in .venv
pause
