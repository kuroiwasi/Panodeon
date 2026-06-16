@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    py -3 -m venv .venv
)

".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -e .[dev]
".venv\Scripts\python.exe" -m pip uninstall -y onnxruntime onnxruntime-directml onnxruntime-gpu
".venv\Scripts\python.exe" -m pip install "onnxruntime-gpu[cuda,cudnn]>=1.21"
".venv\Scripts\python.exe" -m pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-runtime-cu12

echo CUDA/NVIDIA ONNX Runtime installed in .venv
pause
