#!/usr/bin/env bash
set -Eeuo pipefail
: "${EXPERIMENT_ROOT:?}" "${RUN_ROOT:?}" "${TASK:?}" "${STEP:?}" "${K:?}" "${P:?}" "${SERVER_PORT:?}"
EPISODES=${EPISODES:-20}
VIDEO=${VIDEO:-True}
export XR1_FULL_RECORD=${XR1_FULL_RECORD:-1}
export XR1_RECORD_ROOT="$RUN_ROOT/recordings"
RUNNER_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ "$XR1_FULL_RECORD" == 1 ]]; then
  for dependency in full_recording.py scene_recording.py validate_recordings.py; do test -s "$RUNNER_ROOT/$dependency"; done
fi
EFS=/data/efs/users/lingyue.yang
PFS=/mnt/pfs/public
SOURCE_ROOT=${XR1_EVAL_SOURCE_ROOT:-$EFS/RoboSynChallenge_github_main_8818c82}
SIM_ENV=${XR1_SIM_ENV:-$EFS/envs/robosyn-eval-py311}
EMBODICHAIN_SOURCE=${XR1_EMBODICHAIN_SOURCE:-$EFS/EmbodiChain-v0.2.4}
DATA_ROOT=${XR1_DATA_ROOT:-$PFS/data/datasets/robosynchallenge/embodichain_data}
RUN_STORAGE_ROOT=${XR1_RUN_STORAGE_ROOT:-$PFS/models/xr1/evals}
RUNTIME=/tmp/xrp_$$
[[ "$RUN_ROOT" == "$RUN_STORAGE_ROOT"/* ]]
(( K > 0 && K < 30 && P >= 0 && P <= 6 && K+P <= 30 ))
test ! -e "$RUN_ROOT"
actual=$(sha256sum "$SOURCE_ROOT/scripts/eval_policy.py" | cut -d' ' -f1)
[[ "$actual" == 09e727d416aedec843b481246adc810f528470080f8db3a4efc4937e6c156c45 ]]
mkdir -p "$RUN_ROOT" "$RUNTIME/d/assembled"
cleanup() {
  status=$?
  trap - EXIT
  printf '%s\n' "$status" > "$RUN_ROOT/run.exit"
  rm -rf -- "$RUNTIME"
  exit "$status"
}
trap cleanup EXIT
cp -a "$SOURCE_ROOT" "$RUNTIME/c"
cp -a "$EMBODICHAIN_SOURCE" "$RUNTIME/e"
mkdir -p "$RUNTIME/c/policy/xr1"
cp "$EXPERIMENT_ROOT/vendor/robot_bridge.py" "$RUNTIME/c/policy/robot_bridge.py"
cp "$EXPERIMENT_ROOT/vendor/baseline_policy.py" "$RUNTIME/c/policy/xr1/baseline.py"
cp "$EXPERIMENT_ROOT/deploy_policy.py" "$RUNTIME/c/policy/xr1/deploy_policy.py"
if [[ "$XR1_FULL_RECORD" == 1 ]]; then
  cp "$RUNNER_ROOT/full_recording.py" "$RUNTIME/c/policy/xr1/full_recording.py"
  cp "$RUNNER_ROOT/scene_recording.py" "$RUNTIME/c/policy/xr1/scene_recording.py"
fi
if [[ -f "$EXPERIMENT_ROOT/handover_geometry.py" ]]; then
  cp "$EXPERIMENT_ROOT/handover_geometry.py" "$RUNTIME/c/policy/xr1/handover_geometry.py"
fi
printf '%s\n' 'from .deploy_policy import eval, get_model, reset_model, get_episode_diagnostics' > "$RUNTIME/c/policy/xr1/__init__.py"
cp "$EXPERIMENT_ROOT/vendor/deploy_policy.yml" "$RUNTIME/c/policy/xr1/deploy_policy.yml"
cat >> "$RUNTIME/c/policy/xr1/deploy_policy.yml" <<EOF
xr1_prefix_steps: $P
xr1_policy_seed: 20260908
xr1_trace_root: $RUN_ROOT/traces
xr1_handover_geometry: ${HANDOVER_GEOMETRY:-False}
EOF
if [[ -n "${EPISODE_MANIFEST:-}" ]]; then
  printf 'eval_episode_manifest: %s\n' "$EPISODE_MANIFEST" >> "$RUNTIME/c/policy/xr1/deploy_policy.yml"
fi
for item in "$DATA_ROOT"/*; do
  name=${item##*/}
  [[ "$name" == assembled ]] && continue
  ln -s "$item" "$RUNTIME/d/$name"
done
if [[ -d "$DATA_ROOT/extract" ]]; then
  for item in "$DATA_ROOT/extract"/*; do
    name=${item##*/}
    [[ -e "$RUNTIME/d/$name" ]] || ln -s "$item" "$RUNTIME/d/$name"
  done
fi
if [[ -n "${XR1_SIM_CONDA_ROOT:-}" ]]; then
  source "$XR1_SIM_CONDA_ROOT/etc/profile.d/conda.sh"
  conda activate "$SIM_ENV"
else
  source "$SIM_ENV/bin/activate"
fi
export EMBODICHAIN_ROOT=$RUNTIME/e
export EMBODICHAIN_DATA_ROOT=$RUNTIME/d
export EMBODICHAIN_DATASET_ROOT=${XR1_DATASET_ROOT:-$PFS/data/datasets/robosynchallenge/embodichain_datasets}
export EMBODICHAIN_SIM_EXIT_PROCESS=0
export PYTHONPATH="$RUNTIME/c:$RUNTIME/e:$RUNTIME/e/embodichain_tasks"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TMPDIR=/tmp TMP=/tmp TEMP=/tmp SQLITE_TMPDIR=/tmp
export XDG_CACHE_HOME=$RUNTIME/cache
export XR1_SIM_CACHE_DIR=$RUNTIME/cache/embodichain
mkdir -p "$XDG_CACHE_HOME"
patch_args=( --runtime "$RUNTIME" --output "$RUN_ROOT/runtime_patch.json" )
if [[ -n "${EPISODE_MANIFEST:-}" ]]; then patch_args+=( --manifest "$EPISODE_MANIFEST" ); fi
python "$RUNNER_ROOT/prepare_runtime.py" "${patch_args[@]}"
python - "$RUN_ROOT" "$EXPERIMENT_ROOT" "$TASK" "$STEP" "$K" "$P" "$EPISODES" "$RUNNER_ROOT" <<'PY'
import hashlib,json,pathlib,sys,os
root,code=map(pathlib.Path,sys.argv[1:3]);task,step,k,p,n=sys.argv[3:8];runner=pathlib.Path(sys.argv[8])
files={str(x.relative_to(code)):hashlib.sha256(x.read_bytes()).hexdigest() for x in code.rglob('*') if x.is_file() and x.suffix in ('.py','.sh','.yml')}
(root/'evaluation_contract.json').write_text(json.dumps({'upstream_commit':'8818c82','upstream_eval_sha256':'09e727d416aedec843b481246adc810f528470080f8db3a4efc4937e6c156c45','task':task,'checkpoint_step':int(step),'k':int(k),'prefix_length':int(p),'episodes':int(n),'environment_seed':0,'candidate_rng_offset':int(os.environ.get('XR1_EVAL_CANDIDATE_OFFSET','0')),'policy_seed_reset_each_episode':20260908,'execution':'synchronous','prefix_semantics':'first p unexecuted previous actions, re-normalized at the new horizon positions, native model inpainting','source_hashes':files,'runner_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in runner.iterdir() if p.suffix in ('.py','.sh')},'process_exit_strategy':os.environ.get('XR1_EVAL_PROCESS_EXIT','interpreter_shutdown')},indent=2))
PY
cd "$RUNTIME/c"
if [[ "${XR1_SCENE_VERIFY:-0}" == 1 ]]; then
  scene_args=( --output "$RUN_ROOT/restore_validation.json" --task "$TASK" )
  if [[ -n "${XR1_SCENE_VERIFY_INPUT:-}" ]]; then scene_args+=( --recording "$XR1_SCENE_VERIFY_INPUT" ); fi
  if [[ -n "${XR1_SCENE_VERIFY_ACTIONS:-}" ]]; then scene_args+=( --actions "$XR1_SCENE_VERIFY_ACTIONS" ); fi
  if [[ "${XR1_SCENE_DEBUG:-0}" == 1 ]]; then scene_args+=( --debug-on-error ); fi
  python -u "$RUNNER_ROOT/verify_scene_restore.py" "${scene_args[@]}" > "$RUN_ROOT/run.log" 2>&1
  exit 0
fi
status=0
entry=( python -u scripts/eval_policy.py )
if [[ "${XR1_EVAL_PROCESS_EXIT:-}" == verified_fast_exit ]]; then
  entry=( python -u "$RUNNER_ROOT/finish_eval_process.py" "$RUN_ROOT" "$EPISODES" scripts/eval_policy.py )
fi
if [[ "${INIT_AUDIT:-False}" == True ]]; then
  entry=( python -u "$EXPERIMENT_ROOT/init_audit.py" scripts/eval_policy.py )
fi
if [[ -n "${GDB_ROOT:-}" ]]; then
  runtime_library_path=${LD_LIBRARY_PATH:-}
  entry=( env "LD_LIBRARY_PATH=$GDB_ROOT/usr/lib/x86_64-linux-gnu:$runtime_library_path" "$GDB_ROOT/usr/bin/gdb" --batch --return-child-result -ex "set environment LD_LIBRARY_PATH $runtime_library_path" -ex 'set pagination off' -ex 'run' -ex 'thread apply all bt 12' --args "${entry[@]}" )
fi
timeout --signal=TERM --kill-after=30 "$(( EPISODES * 1800 + 600 ))" "${entry[@]}" \
  --config policy/xr1/deploy_policy.yml --overrides \
  --task_name "$TASK" --setting random --model_name "xr1_h30_step${STEP}_k${K}_p${P}" \
  --train_config_name xr1_h30_replan_cost --max_episodes "$EPISODES" --seed 0 \
  --num_envs 1 --headless True --device cpu --gpu_id 0 \
  --xr1_host 127.0.0.1 --xr1_port "$SERVER_PORT" --xr1_action_representation joint_absolute \
  --xr1_execute_steps "$K" --xr1_prefix_steps "$P" --xr1_policy_seed 20260908 \
  --xr1_trace_root "$RUN_ROOT/traces" --xr1_transport_retries 2 --xr1_transport_retry_delay 2 \
  --eval_expert_check True --filter_dataset_saving True --eval_reset_sync_steps 10 \
  --eval_video_log "$VIDEO" --eval_video_obs_keys "['cam_high','cam_left_wrist','cam_right_wrist']" \
  --eval_result_dir "$RUN_ROOT/raw" > "$RUN_ROOT/run.log" 2>&1 || status=$?
printf '%s\n' "$status" > "$RUN_ROOT/eval.exit"
if [[ "${INIT_AUDIT:-False}" == True ]]; then
  if (( status != 0 )); then tail -n 90 "$RUN_ROOT/run.log" >&2; exit "$status"; fi
  python - "$RUN_ROOT/init_audit.json" <<'PY'
import json,sys
assert json.load(open(sys.argv[1]))['passed']
PY
  printf '0\n' > "$RUN_ROOT/INIT_COMPLETE"
  exit 0
fi
shopt -s nullglob
metrics=( "$RUN_ROOT/raw/$TASK/xr1/random/xr1_h30_replan_cost/xr1_h30_step${STEP}_k${K}_p${P}"/*/evaluation_metrics.json )
if (( ${#metrics[@]} == 1 )); then cp "${metrics[0]}" "$RUN_ROOT/metrics.json"; fi
if (( status != 0 )); then tail -n 70 "$RUN_ROOT/run.log" >&2; exit "$status"; fi
if [[ "${XR1_EVAL_PROCESS_EXIT:-}" == verified_fast_exit ]]; then
  python - "$RUN_ROOT/process_boundary.json" "$EPISODES" <<'PYBOUNDARY'
import json,sys
record=json.load(open(sys.argv[1]))
assert record['evaluator_returned'] and record['episodes']==int(sys.argv[2])
PYBOUNDARY
fi
python "$EXPERIMENT_ROOT/validate.py" --root "$RUN_ROOT" --episodes "$EPISODES"
if [[ "$XR1_FULL_RECORD" == 1 ]]; then
  # Recording is accepted separately: a diagnostic failure must not erase a valid evaluation.
  python "$RUNNER_ROOT/validate_recordings.py" --root "$RUN_ROOT" || true
  if [[ "${XR1_VERIFY_RESTORE:-0}" == 1 ]]; then
    recordings=( "$RUN_ROOT"/recordings/episode_*.h5 )
    python -u "$RUNNER_ROOT/verify_scene_restore.py" --recording "${recordings[0]}" \
      --output "$RUN_ROOT/restore_validation.json" > "$RUN_ROOT/restore.log" 2>&1
  fi
fi
printf '0\n' > "$RUN_ROOT/COMPLETE"
