# XR1 RoboSynChallenge submission

This repository contains a standalone RoboSynChallenge policy adapter, an isolated
inference runtime, and launch/validation tools. Eight tasks use the shared XR1 H30
checkpoint at step 160000. `item_assembly` uses the user-selected checkpoint after
10,000 further training steps on 1,000 additional generated assembly expert
trajectories. `water_pouring` uses the shared 160k checkpoint fine-tuned for
5,000 steps on the official 1,000-episode water dataset. `task_mapping.json` is
the authoritative mapping. Existing bell acceptance evidence applies to the
unchanged shared checkpoint.

Code: https://github.com/87361/robosyn-xr1-submission

Weights: https://huggingface.co/87361Yly/robosyn-xr1-sim10

## Install

1. Install the organizers' RoboSynChallenge and EmbodiChain simulator environment
   using their [official instructions](https://edem-ai.github.io/RoboSynChallenge/html/getting_started/installation.html).
   Keep its Python separate from the model Python. The policy does not modify task
   configs, success predicates, evaluation seeds, or the official evaluator.
2. Create the inference environment with Python 3.12:

   ```bash
   bash tools/install_inference.sh /absolute/path/to/xr1-inference-env
   ```

   The installer pins [PyTorch 2.8/CUDA 12.8](https://pytorch.org/get-started/previous-versions/)
   for [Blackwell support](https://pytorch.org/blog/pytorch-2-7/) and uses SDPA.
   The acceptance report records the actual tested hardware and package versions;
   installation instructions alone do not constitute an RTX 5090 verification.
   Fresh simulator installation is also pending: EmbodiChain 0.2.4.post1's
   LeRobot/rerun dependencies require NumPy 2 on Python 3.11. Do not impose the
   conflicting `numpy<2` instruction from older setup notes. Dependency resolution
   alone does not verify native-library compatibility or simulator initialization.
3. Download the complete model repository into a local directory:

   ```bash
   hf download 87361Yly/robosyn-xr1-sim10 --local-dir /absolute/path/to/models
   ```

   Its layout is:

   ```text
   models/
     manifest.json
     shared_160k/model.pt
     shared_160k/config.yaml
     item_assembly_160k_ft10k/model.pt
     item_assembly_160k_ft10k/config.yaml
     water_pouring_160k_ft5k/model.pt
     water_pouring_160k_ft5k/config.yaml
     processor/...
   ```

   The export includes model tensors, exact training normalization statistics,
   tokenizer and processor/model configuration. Inference uses local files only
   and does not download the Qwen backbone weights. Do not substitute a different
   normalization file or tokenizer snapshot.

## Evaluate

Activate the official simulator environment first, including its asset environment
variables. Set the following paths for the current machine:

```bash
export EMBODICHAIN_ROOT=/absolute/path/to/EmbodiChain
export EMBODICHAIN_DATA_ROOT=/absolute/path/to/embodichain_data
export EMBODICHAIN_DATASET_ROOT=/absolute/path/to/embodichain_datasets

bash policy/xr1_submission/eval.sh click_bell \
  --challenge-root /absolute/path/to/RoboSynChallenge \
  --model-dir /absolute/path/to/models \
  --sim-python /absolute/path/to/simulator-env/bin/python \
  --model-python /absolute/path/to/xr1-inference-env/bin/python \
  --output /absolute/path/to/new-evaluation-directory \
  --episodes 2 --video
```

The launcher verifies model checksums and referenced task assets before allocating
the model GPU, installs a symlink for `policy/xr1_submission`, starts a loopback-only inference process,
waits for readiness, invokes the organizer's evaluator, and cleans up its child
processes on completion or failure. It refuses to overwrite an existing output or
another policy installation. Run `--preflight-only` for artifact checks without GPU
startup. An extra `--` forwards organizer overrides to the evaluator.

The official evaluator file runs unchanged through `tools/run_official_evaluator.py`.
This wrapper defers its final `env.close()` until the evaluator has written and
closed its metrics file. In assembly acceptance, native teardown otherwise exited
the process with code 0 before those metrics were written. The wrapper changes
resource-finalization order only: observations, actions, success predicates,
episode selection and the official metric calculations are unchanged. It records
`evaluator_lifecycle.json`, preserves evaluator failures as nonzero exits, and
then invokes the requested native close. Native process exit during that final
close can leave the lifecycle marker at `started`; the launcher still requires
complete official metrics and cleans up the separate inference process.

By default it copies simulator/challenge code into a private short `/tmp/xrs_*`
runtime, links the original assets, creates a fresh robot assembly directory, and
redirects the copied EmbodiChain simulator cache. This avoids DexSim's native
path-length assertion without changing meshes, task configs, evaluator, or HOME.
The single cache-path compatibility edit is hash-recorded in `launch_contract.json`.
EmbodiChain v0.2.4/DexSim 0.4.3 is the acceptance combination. A separately validated
organizer environment may use `--no-isolate-simulator`; this disables the compatibility
copy and must not be used to work around a failed environment check. Native DexSim
assertions cause immediate failure with the log preserved. Runtime copies are
removed only after the evaluator and inference child processes have stopped.

For each of the ten tasks below, substitute the task name and use a distinct output
directory. Use the organizer's prescribed episode count and seed/settings; the
two-episode command above is an integration acceptance run, not an official score.

| Task | Artifact | Execute steps |
|---|---|---:|
| click_bell | shared_160k | 29 |
| handle_basket | shared_160k | 29 |
| water_pouring | water_pouring_160k_ft5k | 29 |
| table_rearrangement | shared_160k | 29 |
| items_handover | shared_160k | 29 |
| drawer_open_place | shared_160k | 29 |
| mixer_operating | shared_160k | 29 |
| item_assembly | item_assembly_160k_ft10k | 29 |
| manipulate_pipette | shared_160k | 29 |
| sample_loading | shared_160k | 29 |

## Interface and evaluation behavior

- Inputs: three RGB views (`cam_high`, `cam_left_wrist`, `cam_right_wrist`), 14D
  robot joint/gripper state, and the task instruction supplied by the evaluator.
- Output: horizon 30, 14D absolute joint targets; execute up to 29 actions, stopping
  on success or termination. Prefix length is zero. The eight normalized-gripper
  tasks and the two raw-gripper tasks retain their distinct data contracts.
- Each episode clears client history and starts a fresh policy RNG stream at seed
  20260908. The policy does not read environment seed IDs, object poses, depth or
  camera intrinsics. Task identity selects only the declared gripper convention.
- Reported inference time includes RGB preprocessing, transport, model computation
  and action encoding. It excludes simulator stepping. The model and simulator run
  on the same selected GPU; CPU physics is selected by the supplied adapter config.
- Per-episode traces and request logs are best-effort diagnostics. The actual
  evaluation metrics and process exit are mandatory acceptance evidence.
- The package supports a caller-provided official evaluator and forwards hidden
  test overrides. No fixed public episode manifest is included.

## Acceptance result — 2026-10-09

The packaged SDPA runtime completed **50/50 successful click_bell episodes** on
NVIDIA L20: average 48.92 action steps (range 45–51), two model calls per episode.
The official expert filter checked 76 candidate scenes and skipped 26 before
policy execution. No policy failures were filtered out. All 50 videos, episode
RNG/chunk resets, metrics and process cleanup passed the integrity audit.

The official evaluator/config/assets were pinned to commit
`8818c82df5fa2522809b8105e7018950dcc9a9a8`; use that revision for reproduction.
The run explicitly used `eval_reset_sync_steps=0`. This publication makes **0 the
adapter default**, replacing the historical two-episode acceptance default of 10.
Policy/model inference code and the shared bell weights are unchanged; the build verifies that the published
configuration equals the effective configuration used in the 50-episode run.
The new teardown-order wrapper was not part of that historical bell run and is
validated separately by the assembly acceptance report.

See `verification/bell50_audit.json` and `verification/bell50_metrics.json` for
current evidence. `verification/acceptance.json` and `verification/RESULTS.md`
retain the earlier two-episode history, whose reset setting was 10.

This verifies the public click_bell pipeline on the existing L20 environments.
A fresh CUDA 12.8 install, RTX 5090, all ten tasks using this packaged SDPA runtime,
and the organizer's held-out scenes have not been validated here.

## Assembly checkpoint addition — 2026-10-11

`item_assembly_160k_ft10k` is exported from run
`xr1_single_item_assembly_160k_ft10k_20261009`, step 10000, initialized from the
shared step-160000 checkpoint. It retains its exact training normalization and
model configuration; normalization equals the shared checkpoint. Only model
tensors are exported, with source/export digests and exact tensor equality checks
in `verification/assembly_export.json`. This record does not establish a new
assembly success rate. Packaged runtime acceptance is reported separately in
`verification/assembly_acceptance.json` and `verification/assembly_metrics.json`;
the previous 50/50 bell result must not be attributed to this checkpoint.

The updated package completed **1/1 successful assembly episode** on the existing
L20 environments: 176 action steps, seven inference calls, H30/k29/p0 and SDPA.
All 351 pinned official source/config/asset files matched; official metrics,
finite action outputs, the three-view video and inference-process cleanup passed
the audit. The first attempt succeeded in the task but lost metrics during native
close; it was rejected as acceptance. The reported passed run uses the documented
teardown-order wrapper. This is not a fresh-install, RTX 5090 or 50-episode test.

## Water checkpoint addition — 2026-10-11

`water_pouring_160k_ft5k` comes from run
`xr1_single_water_pouring_160k_ft5k_20261009`, step 5000, initialized from shared
160k and fine-tuned on the official 1,000-episode water dataset. Exact tensor
export checks are in `verification/water_pouring_export.json`.

The selected checkpoint's historical five-shard run achieved **50/50 success**,
average 183 steps, H30/k29/p0. It used **reset synchronization 10**, plus the
recording/native compatibility changes preserved in `verification/water50/`.
The final submission uses the **official reset default 0**; the historical 50/50
must not be presented as a same-configuration 50-episode test of this release.

This package passed **1/1 water episode at reset 0**: 176 steps and seven model
calls on the existing L20 environments. All 351 pinned official files, finite
actions, three-view video, official metrics and process cleanup passed
`verification/water_pouring_acceptance.json`. The first attempt aborted during
DexSim/OptiX initialization before any policy action; its log is retained as
`verification/water_initial_native_failure.txt`. An unchanged retry with a fresh
private cache passed; the native initialization failure's root cause remains
unresolved. No simulator-thread override was added. This is not fresh-install,
RTX 5090 or all-ten-task acceptance.

## Submit

Sign in at https://robosyn-bench.net/#/login with the registration email and token.
Provide one code repository and one Hugging Face model-repository URL, method/data
description and run notes covering all ten tasks. Private GitHub repositories must
grant the official `robosynchallenge` account access. Do not commit tokens.

Repository names:

- GitHub: `87361/robosyn-xr1-submission`
- Hugging Face: `87361Yly/robosyn-xr1-sim10`

The website currently states that only the latest valid submission is evaluated
once, with deadline October 11, 2026 AoE. Verify the live FAQ before submission.
See `SUBMISSION_TEXT.md` for copy-ready field text and `verification/bell50_audit.json`
for the exact validation boundary. No official evaluation request is sent by publishing these repositories.

## Provenance

Model architecture and inference utilities derive from Xiaomi-Robotics-1 and retain
its Apache-2.0 license. `provenance.json` records original file hashes and deployment
modifications. A source archive can be generated with `python tools/build_release.py`;
large weights and caches are not included in that archive.
