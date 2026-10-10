#!/usr/bin/env bash
set -Eeuo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
env_dir=${1:?Usage: install_inference.sh /absolute/path/to/new/venv}
python_bin=${PYTHON312:-python3.12}
test ! -e "$env_dir"
"$python_bin" -m venv "$env_dir"
"$env_dir/bin/python" -m pip install --upgrade pip
# CUDA 12.8 wheels support Blackwell (RTX 5090). No FlashAttention wheel required.
"$env_dir/bin/python" -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
"$env_dir/bin/python" -m pip install -r "$root/requirements-inference.txt"
"$env_dir/bin/python" -m pip check
"$env_dir/bin/python" -m pip freeze > "$env_dir/installed-versions.txt"
