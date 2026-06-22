#!/usr/bin/env bash
set -euo pipefail

# Resolve the project root (this script lives in scripts/).
cd "$(dirname "$0")/.."

if [[ ! -x ".venv_rocm/bin/colmap-mask" ]]; then
    echo ".venv_rocm not found. Run scripts/setup_env_rocm.sh first."
    exit 1
fi

".venv_rocm/bin/colmap-mask"
