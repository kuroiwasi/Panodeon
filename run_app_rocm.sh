#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -x ".venv_rocm/bin/colmap-mask" ]]; then
    echo ".venv_rocm not found. Run setup_gpu_rocm.sh first."
    exit 1
fi

".venv_rocm/bin/colmap-mask"
