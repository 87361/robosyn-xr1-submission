#!/usr/bin/env bash
# Run the official RoboSynChallenge evaluator for one XR1 task.
# Usage: bash eval.sh TASK SETTING [MODEL_DIR] [GPU_ID] [EVALUATOR_OVERRIDES...]

set -Eeuo pipefail

POLICY_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
POLICY_NAME=$(basename "$POLICY_DIR")
USAGE="Usage: bash eval.sh <task_name> <setting> [model_dir] [gpu_id] [extra_opts...]"

if (($# < 2)); then
    echo "$USAGE" >&2
    exit 2
fi

TASK_NAME=$1
SETTING=$2
shift 2

MODEL_DIR=${XR1_MODEL_DIR:-$POLICY_DIR/models}
GPU_ID=${GPU_ID:-0}
if (($# > 0)) && [[ $1 != --* ]]; then
    MODEL_DIR=$1
    shift
fi
if (($# > 0)) && [[ $1 != --* ]]; then
    GPU_ID=$1
    shift
fi
EXTRA_ARGS=("$@")

if [[ $MODEL_DIR != /* ]]; then
    MODEL_DIR=$POLICY_DIR/$MODEL_DIR
fi

for i in "${!EXTRA_ARGS[@]}"; do
    case "${EXTRA_ARGS[$i]}" in
        true) EXTRA_ARGS[$i]=True ;;
        false) EXTRA_ARGS[$i]=False ;;
        none|null) EXTRA_ARGS[$i]=None ;;
    esac
done

if [[ ! -d $MODEL_DIR ]]; then
    echo "Error: model directory does not exist: $MODEL_DIR" >&2
    echo "Download the pinned checkpoints with: bash $POLICY_DIR/download_checkpoint.sh" >&2
    exit 1
fi
MODEL_DIR=$(cd -- "$MODEL_DIR" && pwd)

if [[ -n ${ROBOSYNCHALLENGE_ROOT:-} ]]; then
    CHALLENGE_ROOT=$(cd -- "$ROBOSYNCHALLENGE_ROOT" && pwd)
else
    CHALLENGE_ROOT=$POLICY_DIR
    while [[ $CHALLENGE_ROOT != / && ! -f $CHALLENGE_ROOT/scripts/eval_policy.py ]]; do
        CHALLENGE_ROOT=$(dirname "$CHALLENGE_ROOT")
    done
fi
if [[ ! -f $CHALLENGE_ROOT/scripts/eval_policy.py ]]; then
    echo "Error: cannot locate the official scripts/eval_policy.py above $POLICY_DIR" >&2
    echo "Clone this repository into RoboSynChallenge/policy/ or set ROBOSYNCHALLENGE_ROOT." >&2
    exit 1
fi

EXPECTED_POLICY_DIR=$CHALLENGE_ROOT/policy/$POLICY_NAME
if [[ $(cd -- "$EXPECTED_POLICY_DIR" 2>/dev/null && pwd || true) != "$POLICY_DIR" ]]; then
    echo "Error: policy must be located at $EXPECTED_POLICY_DIR" >&2
    exit 1
fi

PYTHON_BIN=${PYTHON_BIN:-python}
XR1_PYTHON=${XR1_PYTHON:-$POLICY_DIR/.xr1_env/bin/python}
if [[ $XR1_PYTHON != /* ]]; then
    XR1_PYTHON=$POLICY_DIR/$XR1_PYTHON
fi
EMBODICHAIN_ROOT=${EMBODICHAIN_ROOT:-$(dirname "$CHALLENGE_ROOT")/EmbodiChain}

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: cannot find simulator Python command: $PYTHON_BIN" >&2
    exit 1
fi
if [[ ! -x $XR1_PYTHON ]]; then
    echo "Error: cannot find XR1 inference Python: $XR1_PYTHON" >&2
    echo "Install it with: bash $POLICY_DIR/install.sh, or set XR1_PYTHON." >&2
    exit 1
fi

# DexSim uses the physical ordinal passed through --gpu_id. Only the child model
# process is masked by deploy_policy.py, so the simulator parent stays unmasked.
unset CUDA_VISIBLE_DEVICES
export PYTHONPATH="$CHALLENGE_ROOT:$CHALLENGE_ROOT/policy:$EMBODICHAIN_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "========================================="
echo "  XR1 Policy Evaluation"
echo "  Policy:      $POLICY_NAME"
echo "  Task:        $TASK_NAME ($SETTING)"
echo "  Models:      $MODEL_DIR"
echo "  GPU:         $GPU_ID"
echo "  Challenge:   $CHALLENGE_ROOT"
echo "  Python:      $PYTHON_BIN"
echo "  XR1 Python:  $XR1_PYTHON"
echo "========================================="

cd "$CHALLENGE_ROOT"
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
