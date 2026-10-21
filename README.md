# XR1 policy for RoboSynChallenge (ten simulation tasks)

A [Xiaomi-Robotics-1](https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-1-5B) policy adapted
to all ten RoboSynChallenge simulation tasks, with one selected checkpoint per task.
It reads three RGB views, the 14D joint/gripper state and the task instruction, predicts
30 absolute joint actions per call and executes up to 20 (items handover) or 29 (all
other tasks) before replanning.

- Code: https://github.com/87361/robosyn-xr1-submission
- Weights: https://huggingface.co/87361Yly/robosyn-xr1-sim10, revision
  `fdc1c2f53a0eeb61b02459ef4fca41c7e2aa0dc4`

## Repository layout

| Path | Purpose |
| --- | --- |
| `policy/xr1_submission/` | **The complete runtime unit.** Entry point `eval.sh`, adapter (`deploy_policy.py`, `deploy_policy.yml`), task mapping (`task_mapping.json`), installer (`tools/install_inference.sh`), vendored model implementation (`runtime/`) and inference requirements. |
| `tests/` | Unit and packaging tests. |
| `verification/` | Validation evidence (metrics, audits, checksums). |
| `tools/` | Checkpoint export, audit and release-build tools. Not used at evaluation time. |
| `MODEL_CARD.md` | Copy of the model card published with the pinned Hugging Face revision. |
| `SUBMISSION_TEXT.md` | Portal field text. |

Only `policy/xr1_submission/` is needed for evaluation. `eval.sh` calls nothing
outside that directory except the organizer's own `scripts/eval_policy.py`, task
configs and assets.

## Task mapping

`policy/xr1_submission/task_mapping.json` is authoritative. Each checkpoint folder
contains `model.pt` and its `config.yaml`. All tasks use horizon 30 and no action prefix.

| Task | Checkpoint folder | Execute steps |
| --- | --- | ---: |
| click_bell | `click_bell_160k_ft3k` | 29 |
| handle_basket | `handle_basket_140k` | 29 |
| water_pouring | `water_pouring_160k_ft5k` | 29 |
| table_rearrangement | `table_rearrangement_160k_ft5k` | 29 |
| items_handover | `items_handover_180k` | 20 |
| drawer_open_place | `drawer_open_place_160k_ft10k` | 29 |
| mixer_operating | `mixer_operating_terminal_press_ft4k` | 29 |
| item_assembly | `item_assembly_160k_ft10k` | 29 |
| manipulate_pipette | `manipulate_pipette_160k_ft20k` | 29 |
| sample_loading | `sample_loading_160k_ft20k` | 29 |

## Deployment

### 1. Copy the policy directory

Copy the entire `policy/xr1_submission/` directory into the `policy/` directory of the
organizer's RoboSynChallenge checkout. The entry point detects its directory name,
so the organizer may rename it; use that name in the commands below.

### 2. Simulator environment

Use the organizers' RoboSynChallenge/EmbodiChain environment
([installation guide](https://edem-ai.github.io/RoboSynChallenge/html/getting_started/installation.html)).
The policy does not modify task configs, success predicates, seeds or the official
evaluator. Validated combination: EmbodiChain v0.2.4 with DexSim 0.4.3.

### 3. Inference environment

The model runs in a separate Python 3.12 environment. The default installation
location is `policy/xr1_submission/.xr1_env`; an explicit new path is also supported:

```bash
bash policy/xr1_submission/tools/install_inference.sh
# Or: bash policy/xr1_submission/tools/install_inference.sh /absolute/path/to/new-env
```

The installer pins PyTorch 2.8.0 / torchvision 0.23.0 CUDA 12.8 wheels and the
packages in `policy/xr1_submission/requirements-inference.txt`. Attention uses SDPA;
no FlashAttention build is required. If `python3.12` is unavailable, provide an
existing Python 3.12 interpreter through `PYTHON312`. Existing compatible environments
can be selected through `XR1_PYTHON`; the entry point installs or upgrades nothing.

### 4. Model weights

Download the **complete** model repository at the pinned revision. A single task
folder is not sufficient: the adapter requires `manifest.json` and the shared
`processor/`, then checks the selected task's `model.pt` and `config.yaml` against
the sizes and SHA-256 digests in the manifest before loading that model.

```bash
hf download 87361Yly/robosyn-xr1-sim10 --revision fdc1c2f53a0eeb61b02459ef4fca41c7e2aa0dc4 --local-dir /absolute/path/to/models
```

```text
models/
  manifest.json
  processor/...
  click_bell_160k_ft3k/{model.pt,config.yaml}
  handle_basket_140k/{model.pt,config.yaml}
  water_pouring_160k_ft5k/{model.pt,config.yaml}
  table_rearrangement_160k_ft5k/{model.pt,config.yaml}
  items_handover_180k/{model.pt,config.yaml}
  drawer_open_place_160k_ft10k/{model.pt,config.yaml}
  mixer_operating_terminal_press_ft4k/{model.pt,config.yaml}
  item_assembly_160k_ft10k/{model.pt,config.yaml}
  manipulate_pipette_160k_ft20k/{model.pt,config.yaml}
  sample_loading_160k_ft20k/{model.pt,config.yaml}
  shared_160k/{model.pt,config.yaml}
```

The eleven model files total about 119 GB. `shared_160k` is the common parent
checkpoint, kept for provenance; no task selects it. Each evaluation loads only the
checkpoint mapped to its task. The export bundles the training normalization
statistics, tokenizer and processor; inference is fully offline.

### 5. Run

Activate the simulator environment, then run the official-style entry point:

```bash
export PYTHON_BIN=/absolute/path/to/simulator-env/bin/python
export XR1_PYTHON=/absolute/path/to/xr1-inference-env/bin/python
export EMBODICHAIN_ROOT=/absolute/path/to/EmbodiChain  # defaults to ../EmbodiChain
export EMBODICHAIN_DATA_ROOT=/absolute/path/to/embodichain_data
export EMBODICHAIN_DATASET_ROOT=/absolute/path/to/embodichain_datasets

cd /absolute/path/to/RoboSynChallenge
bash policy/xr1_submission/eval.sh click_bell random /absolute/path/to/models 0 \
  --max_episodes 1 --seed 0 --eval_result_dir /absolute/path/to/new-results
```

For all ten tasks, choose the episode count, seed, setting and fresh output root.
Use absolute paths for `MODEL_DIR` and `OUTPUT_ROOT`:

```bash
for task in click_bell handle_basket water_pouring table_rearrangement items_handover \
            drawer_open_place mixer_operating item_assembly manipulate_pipette sample_loading; do
  bash policy/xr1_submission/eval.sh "$task" "$EVAL_SETTING" "$MODEL_DIR" "$GPU_ID" \
    --max_episodes "$EPISODES" --seed "$EVAL_SEED" \
    --eval_result_dir "$OUTPUT_ROOT/$task" --xr1_log_root "$OUTPUT_ROOT/$task/server"
done
```

| Argument or setting | Default | Meaning |
| --- | --- | --- |
| `<task_name> <setting> <model_dir>` | required | Official task, setting, and complete downloaded model repository. |
| `[gpu_id]` | 0 | Physical GPU index, passed to the simulator and selected for the inference worker. |
| `PYTHON_BIN` | policy-local `.venv/bin/python`, otherwise `python` | Simulator Python. |
| `XR1_PYTHON` | policy-local `.xr1_env/bin/python` | Separate model Python 3.12; an executable path. |
| `EMBODICHAIN_ROOT` | sibling `EmbodiChain/` checkout | Existing simulator source. |
| `--max_episodes` / `--seed` | 20 / 0 | Official evaluator overrides. |
| `--eval_result_dir` | official evaluator default | Root under which the evaluator creates its task/config/timestamp result directory. |
| `--xr1_log_root` | new temporary directory | Inference logs: `server.log`, `READY`, `requests.jsonl`. Use a fresh directory. |
| `--xr1_startup_timeout` | 900 seconds | Wait for the model worker to become ready. |
| Other overrides | official evaluator defaults | Passed after the positional GPU argument; booleans/null are normalized for its parser. |

The official evaluator writes `evaluation_metrics.json` below `--eval_result_dir`.
A completed result requires that metrics file and a successful process exit. The
adapter prints the inference log directory when it starts the model worker.

## Adapter lifecycle

1. `eval.sh` runs the organizer's unchanged `scripts/eval_policy.py` with the policy's
   `deploy_policy.yml`. It sets the official `EMBODICHAIN_SIM_EXIT_PROCESS=0` switch,
   which lets native cleanup return so the evaluator can save its metrics.
2. The evaluator creates the simulation environment, then calls `get_model()`.
   The adapter resolves the selected checkpoint and verifies its model/config files,
   the model manifest and processor directory before model startup.
3. Following the official SmolVLA worker pattern, `get_model()` starts a separate
   Python process from `policy/xr1_submission/runtime/server.py`, using a free
   loopback port. Only the vendored runtime is put on that worker's `PYTHONPATH`;
   Hugging Face and Transformers run offline.
4. Readiness requires a matching `READY` port. Early exit and startup timeout report
   the end of the worker log. `atexit` cleanup terminates and waits for the worker;
   Linux parent-death signaling also stops it if the evaluator dies unexpectedly.
5. After metrics are written, an adapter exit hook drains the existing official
   `SimulationManager.flush_cleanup_queue()` after the inference workers stop.
   It first removes registered rigid constraints through the existing official
   `remove_rigid_constraint()` API while their simulation world is still alive.
   This completes the deferred destruction queued by `env.close()` and avoids
   leaving native simulation objects for interpreter garbage collection.
6. An explicitly supplied `xr1_port` connects to an externally managed service.
   Otherwise the adapter owns the service lifecycle.

The policy does not copy or patch the organizer checkout or EmbodiChain, wrap
`gym.make`, or insert policy-directory symlinks. On the validated Baidu machine,
existing Python source packages were registered in the simulator environment and a
short `EMBODICHAIN_DATA_ROOT` alias pointed to the same PFS assets. This is an external
environment setting. The original robot meshes were retained; no CUDA, driver or
Python package upgrade was needed. The observed material-creation assertion was
resolved with the short data path, which also regenerated the simulator's assembled
cache; those two effects have not been isolated. Ensure the simulator environment
can import `embodichain_tasks.tableware.base_agent_env` before launching a task.

## Interface and evaluation behavior

- Inputs: three RGB views (`cam_high`, `cam_left_wrist`, `cam_right_wrist`), 14D
  joint/gripper state and the task instruction supplied by the evaluator.
- Output: 30 absolute 14D joint targets per call; up to 20 (handover) or 29 (others)
  are executed, stopping early on success or termination. The eight
  normalized-gripper tasks and the two raw-gripper tasks keep their own data contracts.
- Each episode clears client history and restarts the policy RNG at seed 20260908,
  independent of the evaluator's environment seed. The policy does not read seed IDs,
  object poses, depth or camera intrinsics.
- Reset synchronization is 0, the official evaluator default.
- Model and simulator share the selected GPU (peak usage 13,665 MiB in validation);
  physics runs on CPU as configured in `deploy_policy.yml`.
- Reported inference time covers preprocessing, transport, model computation and
  action decoding, not simulator stepping.
- Request traces are best-effort diagnostics; the official metrics and the process exit
  code are the result.

## Validation

The historical N20/N50 results below are from the public evaluator (official revision
`8818c82df5fa2522809b8105e7018950dcc9a9a8`, random setting, reset synchronization 0),
with policy failures kept in the denominator. They are not hidden-test scores.

| Task | N=20, selected checkpoint | N=50, selected checkpoint | Peak GPU memory (MiB) |
| --- | ---: | ---: | ---: |
| click_bell | 20/20 | 50/50 | 13405 |
| handle_basket | 19/20 | 47/50 | 13665 |
| water_pouring | 20/20 | 50/50 | 13434 |
| table_rearrangement | 20/20 | 50/50 | 13464 |
| items_handover | 19/20 | 48/50 | 13408 |
| drawer_open_place | 16/20 | 45/50 | 13559 |
| mixer_operating | 20/20 | — | 13434 |
| item_assembly | 20/20 | 50/50 | 13406 |
| manipulate_pipette | 19/20 | — | 13406 |
| sample_loading | 11/20 | — | 13407 |
| **Total** | **184/200** | 340/350 (7 tasks) | |

The N=20 results come from per-task batches with identical model, config, processor and
policy runtime bytes (`verification/original3_20261012/`); the N=50 runs are in
`verification/selected_official_n50/`. They ran on NVIDIA L4 with simulator Python 3.11
(PyTorch 2.10.0, NumPy 1.26.4, DexSim 0.4.3) and inference Python 3.12 (PyTorch 2.8.0,
NumPy 2.1.3).

The current official-structure adapter was validated against official revision
`4b6a6584c8d68d98b200b80d93df5993eeca03fb` on Baidu L20 with simulator Python 3.11
(PyTorch 2.7.1+cu126, DexSim 0.4.3) and model Python 3.12 (PyTorch 2.8.0+cu126).
All 10 tasks completed one episode and passed runtime/cleanup acceptance; 9/10 episodes succeeded. Every evaluator exited 0, wrote official metrics, matched request counts and execution contracts, and released the worker, port and GPU memory; native-fault logs were empty. All ten episodes ran afresh with this exact policy revision.

| Task | Episode success | Official steps | Actual trace steps | Calls | k |
| --- | ---: | ---: | ---: | ---: | ---: |
| click_bell | 1/1 | 47 | 47 | 2 | 29 |
| handle_basket | 1/1 | 351 | 351 | 13 | 29 |
| water_pouring | 1/1 | 178 | 178 | 7 | 29 |
| table_rearrangement | 1/1 | 163 | 163 | 6 | 29 |
| items_handover | 1/1 | 332 | 332 | 17 | 20 |
| drawer_open_place | 1/1 | 404 | 404 | 14 | 29 |
| mixer_operating | 1/1 | 257 | 257 | 9 | 29 |
| item_assembly | 1/1 | 172 | 172 | 6 | 29 |
| manipulate_pipette | 1/1 | 290 | 290 | 10 | 29 |
| sample_loading | 0/1 | 500 | 500 | 18 | 29 |

The final evidence is in `verification/policy_structure_20261021/` and is bound
to the exact tested policy bytes. `policy_layout_20261021/` is historical evidence
for the superseded launcher version. The historical 184/200 is not a new score for
this adapter.

Not verified: a fresh installation of both environments from scratch, RTX 5090
hardware, and the organizers' held-out scenes. A rare native crash during simulator
initialization was observed in earlier runs (`verification/runtime_diagnostics_20261012/`);
its root cause is unknown, and an unchanged retry succeeded.

## Training data and checkpoints

All checkpoints start from Xiaomi-Robotics-1 pretrained weights and descend from one
ten-task run on the official pre-collected Sim release (ten tasks × 1,000 episodes).

| Task | Checkpoint |
| --- | --- |
| handle_basket, items_handover | Ten-task run, steps 140,000 and 180,000. |
| click_bell, water_pouring, table_rearrangement, drawer_open_place | Ten-task step 160,000, fine-tuned 3,000 / 5,000 / 5,000 / 10,000 steps on the task's official 1,000 episodes. |
| item_assembly | Ten-task step 160,000, fine-tuned 10,000 steps on 1,000 generated assembly expert trajectories. |
| manipulate_pipette | Ten-task step 160,000, fine-tuned 20,000 steps on generated expert trajectories (`expert_fulldata_20261007`). |
| sample_loading | Ten-task step 160,000, fine-tuned 20,000 steps on a redesigned expert dataset (`expert_redesign_20261007_add800_v1`). |
| mixer_operating | Single-task mixer model (ten-task step 160,000 + 10,000 steps), fine-tuned 4,000 more steps on the official 1,000 episodes plus 100 generated terminal-press episodes, sampled 1:1 by frames. |

Exports match their source tensors exactly and keep each run's training normalization;
see `verification/*_export.json` and `verification/original3_20261012/*_export.json`.

## License and provenance

Model architecture and inference utilities derive from Xiaomi-Robotics-1 and keep its
Apache-2.0 license. `provenance.json` records the original file hashes and deployment
modifications. `python tools/build_release.py` builds a source archive with a checksum
manifest; weights and caches are not included.
