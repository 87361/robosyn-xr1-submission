#!/usr/bin/env bash
# Create the separate Python 3.12 XR1 inference environment.
# Usage: install_inference.sh [/absolute/path/to/new/env]   (default: policy/<name>/.xr1_env)
# Without a system python3.12, create one first, e.g.
#   conda create -y -p /path/to/py312 python=3.12 && PYTHON312=/path/to/py312/bin/python install_inference.sh
set -Eeuo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
env_dir=${1:-$root/.xr1_env}
python_bin=${PYTHON312:-python3.12}
if ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "Error: $python_bin not found; set PYTHON312 to a Python 3.12 interpreter (see the header of this script)." >&2
    exit 1
fi
if [[ -e "$env_dir" ]]; then
    echo "Error: $env_dir already exists" >&2
    exit 1
fi
"$python_bin" -m venv "$env_dir"
"$env_dir/bin/python" -m pip install --upgrade pip
# CUDA 12.8 wheels support Blackwell (RTX 5090). No FlashAttention wheel required.
"$env_dir/bin/python" -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
"$env_dir/bin/python" -m pip install -r "$root/requirements-inference.txt"
"$env_dir/bin/python" -m pip check
"$env_dir/bin/python" -m pip freeze > "$env_dir/installed-versions.txt"
echo "XR1 inference Python: $env_dir/bin/python"
