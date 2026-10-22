# XR1 policy for RoboSynChallenge

This repository is a complete RoboSynChallenge policy for all ten simulation tasks.
Clone it directly below the organizer checkout's `policy/` directory; no files need
to be copied, moved, patched, or installed into the organizer repository.

The policy is based on
[Xiaomi-Robotics-1](https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-1-5B).
Its checkpoints are published at
[`87361Yly/robosyn-xr1-sim10`](https://huggingface.co/87361Yly/robosyn-xr1-sim10)
and are pinned to revision
`fdc1c2f53a0eeb61b02459ef4fca41c7e2aa0dc4`.

## Quick start

The organizer's **RoboSynChallenge, EmbodiChain, simulator dependencies, and assets
must already be installed.** The block below is intended to be copied and run as one
sequence from the directory containing the official `RoboSynChallenge/` checkout.
Each step states its working directory:

```bash
cd RoboSynChallenge/policy
git clone https://github.com/87361/robosyn-xr1-submission.git

cd robosyn-xr1-submission

bash install.sh

bash download_checkpoint.sh

# Before the final command, make sure the organizer's simulator environment is active and its existing EmbodiChain data variables are configured. `eval.sh` uses that environment's `python` for simulation and `.xr1_env/bin/python` for inference.

bash eval.sh click_bell random
```

`install.sh` creates the **separate policy-local `.xr1_env/` with Python 3.12 and
the model dependencies**. `download_checkpoint.sh` downloads the complete pinned model release
directly to `robosyn-xr1-submission/models/` and checks every file listed by
`robosyn-xr1-submission/models/manifest.json`, including size and SHA-256. The download is
approximately 119 GB.

## Installation details

Run this step from `RoboSynChallenge/policy/robosyn-xr1-submission/`. The default installer
requires a `python3.12` executable and creates `.xr1_env/`:

```bash
bash install.sh
```

To select a Python 3.12 interpreter or a different environment directory:

```bash
PYTHON312=/path/to/python3.12 bash install.sh /path/to/xr1_env
```

The installer pins PyTorch 2.8.0, torchvision 0.23.0, CUDA 12.8 wheels, and the
packages in `requirements-inference.txt`. Attention uses SDPA; FlashAttention is
not required. If a non-default environment is used, set
`XR1_PYTHON=/path/to/xr1_env/bin/python` for download and evaluation.

## Checkpoint download

Run this step from `RoboSynChallenge/policy/robosyn-xr1-submission/`. The normal command writes
the Hugging Face repository contents directly into `robosyn-xr1-submission/models/`:

```bash
bash download_checkpoint.sh
```

The resulting paths are `models/manifest.json`, `models/processor/`, and the ten
checkpoint directories. There is no additional `models/robosyn-xr1-sim10/` nesting.

A custom destination is also supported. Pass it to both download and evaluation:

```bash
bash download_checkpoint.sh /absolute/path/to/xr1_models
bash eval.sh click_bell random /absolute/path/to/xr1_models
```

An already populated custom model directory can be used without downloading again:

```bash
bash eval.sh click_bell random /absolute/path/to/xr1_models
```

The downloader always requests the fixed revision shown above. It requires
`manifest.json`, a non-empty `processor/`, and the model/config pair selected for
each of the ten tasks. It then verifies every manifest entry. At evaluation startup,
the adapter verifies the selected task's checkpoint and config again before any GPU
initialization.

## Evaluation

Run evaluation from `RoboSynChallenge/policy/robosyn-xr1-submission/` after activating the
organizer's existing simulator environment:

The public interface is:

```text
bash eval.sh TASK SETTING [MODEL_DIR] [GPU_ID] [OFFICIAL_EVALUATOR_OVERRIDES...]
```

The normal invocation needs only the task and setting because the model directory
defaults to `models/` and the GPU defaults to `0`:

```bash
bash eval.sh click_bell random
```

To use a custom model directory, GPU, episode count, seed, or output directory:

```bash
bash eval.sh click_bell random /absolute/path/to/models 1 \
  --max_episodes 1 --seed 0 --eval_result_dir outputs/xr1
```

Run all ten task entry points with the default local models and GPU 0:

```bash
for task in click_bell handle_basket water_pouring table_rearrangement items_handover \
            drawer_open_place mixer_operating item_assembly manipulate_pipette sample_loading; do
  bash eval.sh "$task" random || break
done
```

The task-to-checkpoint mapping is authoritative in `task_mapping.json`:

| Task | Checkpoint | Steps used before replanning |
| --- | --- | ---: |
| `click_bell` | `click_bell_160k_ft3k` | 29 |
| `handle_basket` | `handle_basket_140k` | 29 |
| `water_pouring` | `water_pouring_160k_ft5k` | 29 |
| `table_rearrangement` | `table_rearrangement_160k_ft5k` | 29 |
| `items_handover` | `items_handover_180k` | 20 |
| `drawer_open_place` | `drawer_open_place_160k_ft10k` | 29 |
| `mixer_operating` | `mixer_operating_terminal_press_ft4k` | 29 |
| `item_assembly` | `item_assembly_160k_ft10k` | 29 |
| `manipulate_pipette` | `manipulate_pipette_160k_ft20k` | 29 |
| `sample_loading` | `sample_loading_160k_ft20k` | 29 |

Useful environment overrides are:

| Variable | Default | Purpose |
| --- | --- | --- |
| `PYTHON_BIN` | `python` | Python from the organizer's active simulator environment |
| `XR1_PYTHON` | `.xr1_env/bin/python` | Separate Python 3.12 inference interpreter |
| `XR1_MODEL_DIR` | `models/` | Model repository when no positional model path is supplied |
| `GPU_ID` | `0` | GPU when no positional GPU is supplied |
| `ROBOSYNCHALLENGE_ROOT` | auto-detected | Optional explicit official checkout root |
| `EMBODICHAIN_ROOT` | sibling `EmbodiChain/` | Existing EmbodiChain source checkout |

All remaining arguments are forwarded to the unchanged official evaluator. The
policy does not replace task configuration, seeds, success predicates, expert
filtering, or scoring.

## Runtime behavior

Each inference request uses three RGB views, the 14-dimensional joint/gripper state,
and the evaluator's task instruction. The model decodes 30 future 14-dimensional
absolute joint/gripper targets. The adapter executes at most 20 targets for
`items_handover` and 29 for every other task, then replans from the latest observation;
success or termination can stop a chunk earlier.

`eval.sh` invokes the official `scripts/eval_policy.py`. The adapter starts the
separate inference process on loopback, waits for its readiness marker, and owns its
lifecycle. Normal and abnormal evaluator exits stop the worker; normal shutdown also
releases registered rigid constraints through the official API before flushing the
simulator cleanup queue.

## Validation boundary

The closed-loop tests used existing CUDA 12.6 PyTorch inference environments; the
fresh installer intentionally selects CUDA 12.8 wheels. A fresh full simulator
installation, RTX 5090, held-out organizer scenes, and a new 119 GB transfer from
Hugging Face have not been validated.

## License and provenance

The model architecture and inference utilities derive from Xiaomi-Robotics-1 and
retain the Apache-2.0 license. Model weights are distributed separately at the
pinned Hugging Face revision shown at the top of this README.
