# XR1 policy for RoboSynChallenge (ten simulation tasks)

A [Xiaomi-Robotics-1](https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-1-5B) policy adapted
to all ten RoboSynChallenge simulation tasks, with one selected checkpoint per task.
It reads three RGB views, the 14D joint/gripper state and the task instruction.
Each inference call produces a decoded sequence of 30 action steps, each containing
14 joint/gripper targets. The adapter executes the first 20 steps for items handover
or the first 29 for every other task, then replans from the latest observation.
Success or episode termination can stop execution earlier; unused predictions are discarded.

- Code: https://github.com/87361/robosyn-xr1-submission
- Weights: https://huggingface.co/87361Yly/robosyn-xr1-sim10, revision
  `fdc1c2f53a0eeb61b02459ef4fca41c7e2aa0dc4`
- [Model card and training details](https://huggingface.co/87361Yly/robosyn-xr1-sim10/blob/fdc1c2f53a0eeb61b02459ef4fca41c7e2aa0dc4/README.md)

## Repository layout

| Path | Purpose |
| --- | --- |
| `policy/xr1_submission/` | **The complete runtime unit.** Entry point `eval.sh`, adapter (`deploy_policy.py`, `deploy_policy.yml`), task mapping (`task_mapping.json`), installer (`tools/install_inference.sh`), vendored model implementation (`runtime/`) and inference requirements. |
| `README.md` | Installation, model download and evaluation instructions. |
| `LICENSE` | Apache-2.0 license. |
| `provenance.json` | Upstream revision, original file hashes and deployment modifications. |

## Task mapping

`policy/xr1_submission/task_mapping.json` is authoritative. Each checkpoint folder
contains `model.pt` and its `config.yaml`. All tasks predict 30 steps per inference
call (prediction horizon), with no action prefix carried over from the previous call.
The execution horizon below is the maximum number of those steps used before replanning.

| Task | Checkpoint folder | Max steps executed per inference call |
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

### 1. Install the policy

Start in the directory that contains the organizer's existing
`RoboSynChallenge/` checkout. Clone this repository beside the official checkout,
copy its runtime directory into `RoboSynChallenge/policy/`, then enter the official
checkout:

```bash
git clone https://github.com/87361/robosyn-xr1-submission.git
cp -a robosyn-xr1-submission/policy/xr1_submission RoboSynChallenge/policy/
cd RoboSynChallenge
```

The official evaluator loads policies from
`RoboSynChallenge/policy/<policy_name>/`, which is why the runtime directory is copied
there. Run all remaining commands from `RoboSynChallenge/` in the same terminal.

After installation, the relevant layout is:

```text
workspace/
  RoboSynChallenge/              # run the remaining commands here
    policy/xr1_submission/       # installed policy runtime
  robosyn-xr1-submission/        # cloned submission repository
    policy/xr1_submission/
    models/                      # created when the weights are downloaded
```

If the official checkout has a different directory name, use that name in the
copy and `cd` commands.

The entry point detects its policy directory name, so the organizer may rename it;
use that name in the commands below if renamed.

### 2. Simulator environment

Use the organizers' RoboSynChallenge/EmbodiChain environment
([installation guide](https://edem-ai.github.io/RoboSynChallenge/html/getting_started/installation.html)).
The policy does not modify task configs, success predicates, seeds or the official
evaluator. Validated combination: EmbodiChain v0.2.4 with DexSim 0.4.3.

### 3. Inference environment

**The model runs in a separate Python 3.12 environment.** Install it inside the copied
policy directory in the organizer's checkout. An explicit new environment directory
can also be passed as the installer's first argument:

```bash
bash policy/xr1_submission/tools/install_inference.sh
```

The installer pins PyTorch 2.8.0 / torchvision 0.23.0 CUDA 12.8 wheels and the
packages in the copied policy's `requirements-inference.txt`. Attention uses SDPA;
no FlashAttention build is required. If `python3.12` is unavailable, provide an
existing Python 3.12 interpreter through `PYTHON312`. Existing compatible environments
can be selected through `XR1_PYTHON`; the entry point installs or upgrades nothing.

### 4. Model weights

Download the **complete** model repository at the pinned revision. A single task
folder is not sufficient: the adapter requires `manifest.json` and the shared
`processor/`, then checks the selected task's `model.pt` and `config.yaml` against
the sizes and SHA-256 digests in the manifest before loading that model.

```bash
policy/xr1_submission/.xr1_env/bin/hf download \
  87361Yly/robosyn-xr1-sim10 \
  --revision fdc1c2f53a0eeb61b02459ef4fca41c7e2aa0dc4 \
  --local-dir ../robosyn-xr1-submission/models
```

This uses the Hugging Face CLI installed in the inference environment from step 3.
By default, the downloaded files are directly visible in this submission repository's
`models/` directory, at `../robosyn-xr1-submission/models` from the current working
directory. Both run examples below use this same location. To store weights elsewhere,
replace that model path in the download and run commands with your chosen absolute
path. Downloaded weights are ignored by Git and are not part of the source submission.

```text
../robosyn-xr1-submission/models/
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

Activate the organizer's simulator environment and keep its configured EmbodiChain
source and asset paths. From the same `RoboSynChallenge/` directory, run one
click-bell episode:

```bash
bash policy/xr1_submission/eval.sh \
  click_bell random ../robosyn-xr1-submission/models 0 --max_episodes 1 --seed 0 \
  --eval_result_dir outputs/xr1
```

For all ten tasks, use one episode per task, GPU 0, the `random` setting and seed 0.
Increase `--max_episodes` for a longer evaluation. The loop stops on a task error:

```bash
for task in click_bell handle_basket water_pouring table_rearrangement items_handover \
            drawer_open_place mixer_operating item_assembly manipulate_pipette sample_loading; do
  bash policy/xr1_submission/eval.sh \
    "$task" random ../robosyn-xr1-submission/models 0 --max_episodes 1 --seed 0 \
    --eval_result_dir outputs/xr1 || break
done
```

These commands read the models from the submission repository and write results
under `RoboSynChallenge/outputs/xr1/`. Inference log directories are created automatically.
The simulator's `EMBODICHAIN_ROOT`, `EMBODICHAIN_DATA_ROOT` and
`EMBODICHAIN_DATASET_ROOT` remain environment-specific; configure them as part of
the official simulator installation rather than replacing them with example paths.

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

`eval.sh` runs the unchanged official `scripts/eval_policy.py` with
`EMBODICHAIN_SIM_EXIT_PROCESS=0`, allowing the evaluator to save its metrics after
closing the environment. The evaluator creates the simulation, then `get_model()`
checks the selected model/config against the model manifest and starts the separate
inference worker on a free loopback port. The worker uses only the policy-local
runtime and runs Hugging Face and Transformers offline.

The adapter waits for readiness and reports startup failures with the worker's log.
On exit, it stops and waits for the worker, removes registered rigid constraints
through the official `remove_rigid_constraint()` API, then calls the official
`SimulationManager.flush_cleanup_queue()` to complete deferred world destruction.
This cleanup runs at process exit, after normal evaluation has written its metrics.
Linux parent-death signaling also stops the worker if the evaluator dies unexpectedly.
An explicitly supplied `xr1_port` instead connects to an externally managed service.

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
- Decoded policy response: shape `(30, 14)` — 30 future action steps, each containing
  left-arm joint targets (6), left-gripper target (1), right-arm joint targets (6)
  and right-gripper target (1). Arm targets are absolute positions. The eight
  normalized-gripper tasks and the two raw-gripper tasks keep their own data contracts;
  the adapter converts these targets to simulator commands.
- Execution: use only the first 20 predicted steps for `items_handover`, or the first
  29 for every other task. Then obtain the latest observation and predict a new
  30-step sequence, discarding the unused steps. Stop early on success or termination.
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

## Validation and limitations

The current adapter completed one public-setting episode per task against official
revision `4b6a6584c8d68d98b200b80d93df5993eeca03fb` on Baidu L20. The simulator used
Python 3.11, EmbodiChain v0.2.4, PyTorch 2.7.1+cu126 and DexSim 0.4.3; the model used
Python 3.12 and PyTorch 2.8.0+cu126.

- All ten tasks passed runtime and cleanup checks: exit code 0, official metrics
  written, no native faults, and worker, port and GPU memory released.

A fresh installation of both environments, RTX 5090 hardware and the organizers'
held-out scenes have not been validated. The fresh installer selects CUDA 12.8 wheels,
whereas acceptance used existing CUDA 12.6 PyTorch environments. Earlier tests also
observed a rare native crash during simulator initialization; its cause remains
unknown, and an unchanged retry succeeded.

## License and provenance

Model architecture and inference utilities derive from Xiaomi-Robotics-1 and keep its
Apache-2.0 license. `provenance.json` records the original file hashes and deployment
modifications. Weights are distributed separately through the pinned model repository.
