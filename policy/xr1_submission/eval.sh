#!/bin/bash
# ----------------------------------------------------------------------------
# bash eval.sh <task_name> <setting> <model_dir> [gpu_id] [extra_opts...]
# bash eval.sh click_bell random /path/to/robosyn-xr1-sim10 0 --max_episodes 5
#
# <model_dir> is the complete model repository; task_mapping.json selects the
# checkpoint for <task_name>. deploy_policy.get_model starts the XR1 inference
# process with the separate Python 3.12 interpreter XR1_PYTHON.
# ----------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
EMBODICHAIN_ROOT="${EMBODICHAIN_ROOT:-$WORKSPACE_ROOT/EmbodiChain}"
VENV_DIR="$SCRIPT_DIR/.venv"
if [[ -z "${PYTHON_BIN:-}" && -x "$VENV_DIR/bin/python" ]]; then
    PYTHON_BIN="$VENV_DIR/bin/python"
else
    PYTHON_BIN="${PYTHON_BIN:-python}"
fi
XR1_PYTHON="${XR1_PYTHON:-$SCRIPT_DIR/.xr1_env/bin/python}"

POLICY_NAME="$(basename "$SCRIPT_DIR")"
USAGE="Usage: bash policy/$POLICY_NAME/eval.sh <task_name> <setting> <model_dir> [gpu_id] [extra_opts...]"

TASK_NAME="${1:?$USAGE}"
SETTING="${2:?$USAGE}"
MODEL_DIR="${3:?$USAGE}"
GPU_ID="${4:-0}"

if (($# >= 4)); then shift 4; else shift $#; fi
EXTRA_ARGS=("$@")

for i in "${!EXTRA_ARGS[@]}"; do
    case "${EXTRA_ARGS[$i]}" in
        true) EXTRA_ARGS[$i]=True ;;
        false) EXTRA_ARGS[$i]=False ;;
        none|null) EXTRA_ARGS[$i]=None ;;
    esac
done

if [[ ! -d "$MODEL_DIR" ]]; then
    echo "Error: model directory does not exist: $MODEL_DIR" >&2
    exit 1
fi
MODEL_DIR="$(cd "$MODEL_DIR" && pwd)"

# DexSim uses physical GPU ordinals from --gpu_id. Keep the simulator parent
# unmasked; get_model masks only the XR1 inference process to this GPU.
unset CUDA_VISIBLE_DEVICES
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/policy:$EMBODICHAIN_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "========================================="
echo "  XR1 Policy Evaluation"
echo "  Task:       $TASK_NAME ($SETTING)"
echo "  Models:     $MODEL_DIR"
echo "  GPU:        $GPU_ID"
echo "  Python:     $PYTHON_BIN"
echo "  XR1 Python: $XR1_PYTHON"
echo "========================================="

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: cannot find Python command: $PYTHON_BIN" >&2
    exit 1
fi
if [[ ! -x "$XR1_PYTHON" ]]; then
    echo "Error: cannot find XR1 inference Python: $XR1_PYTHON" >&2
    echo "Install it with: bash policy/$POLICY_NAME/tools/install_inference.sh, or set XR1_PYTHON." >&2
    exit 1
fi

cd "$REPO_ROOT"

PYTHONWARNINGS=ignore::UserWarning \
EMBODICHAIN_SIM_EXIT_PROCESS=0 \
"$PYTHON_BIN" scripts/eval_policy.py \
    --config "policy/$POLICY_NAME/deploy_policy.yml" \
    --overrides \
    --policy_name "$POLICY_NAME" \
    --task_name "$TASK_NAME" \
    --setting "$SETTING" \
    --xr1_model_dir "$MODEL_DIR" \
    --xr1_python "$XR1_PYTHON" \
    --model_name xr1_sim10 \
    --gpu_id "$GPU_ID" \
    "${EXTRA_ARGS[@]}"
