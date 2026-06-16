#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3.10}"
ROCM_ONNXRUNTIME_WHEEL="${ROCM_ONNXRUNTIME_WHEEL:-https://repo.radeon.com/rocm/manylinux/rocm-rel-6.1.3/onnxruntime_rocm-1.17.0-cp310-cp310-linux_x86_64.whl}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "$PYTHON_BIN not found. ROCm 6.1.3 ONNX Runtime wheel requires Python 3.10."
    exit 1
fi

if [[ "$($PYTHON_BIN - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)" != "3.10" ]]; then
    echo "ROCm 6.1.3 ONNX Runtime wheel requires Python 3.10."
    exit 1
fi

if ! command -v rocm-smi >/dev/null 2>&1; then
    echo "rocm-smi not found. Install Radeon Software for Linux with ROCm first."
    exit 1
fi

if ! dpkg -l | grep -q '^ii  migraphx'; then
    echo "MIGraphX is not installed. Install migraphx/migraphx-dev first."
    exit 1
fi

if ! dpkg -l | grep -q '^ii  half'; then
    echo "half library is not installed. Run: sudo apt install half"
    exit 1
fi

if [[ ! -x ".venv_rocm/bin/python" ]]; then
    "$PYTHON_BIN" -m venv .venv_rocm
fi

".venv_rocm/bin/python" -m pip install --upgrade pip
".venv_rocm/bin/python" -m pip install -e '.[dev]'
".venv_rocm/bin/python" -m pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime-directml onnxruntime-rocm numpy
".venv_rocm/bin/python" -m pip install "$ROCM_ONNXRUNTIME_WHEEL" numpy==1.26.4

".venv_rocm/bin/python" - <<'PY'
import onnxruntime as ort
providers = ort.get_available_providers()
print("ONNX Runtime providers:", providers)
required = {"MIGraphXExecutionProvider", "ROCMExecutionProvider", "CPUExecutionProvider"}
missing = required.difference(providers)
if missing:
    raise SystemExit(f"Missing providers: {sorted(missing)}")
PY

echo "ROCm ONNX Runtime installed in .venv_rocm"
