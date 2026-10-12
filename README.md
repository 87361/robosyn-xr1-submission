# XR1 RoboSynChallenge submission

This repository contains a standalone RoboSynChallenge policy adapter, an isolated
inference runtime, and launch/validation tools. All ten task checkpoints are user-confirmed. `item_assembly` uses the user-selected checkpoint after
10,000 further training steps on 1,000 additional generated assembly expert
trajectories. `water_pouring` uses the shared 160k checkpoint fine-tuned for
5,000 steps on the official 1,000-episode water dataset. `task_mapping.json` is
the authoritative mapping: `handle_basket` selects H30 140k/k29 and
`items_handover` selects H30 180k/k20. The confirmed bell checkpoint is shared160k+3k; existing shared-bell acceptance is historical only. Table rearrangement selects shared160k+5k and drawer placement selects shared160k+10k. Mixer selects the latest terminal-press +4k (initialized from the original single-task 160k+10k); pipette and sample loading select their original single-task 160k+20k checkpoints. These replace the former shared160k fallbacks.

Code: https://github.com/87361/robosyn-xr1-submission

Weights: https://huggingface.co/87361Yly/robosyn-xr1-sim10

## Current selected-policy acceptance — 2026-10-12

All selected policies have 20 completed public-evaluator episodes at official
reset synchronization 0. The latest mixer +4k achieved **20/20**, original pipette
+20k **19/20**, and original sample loading +20k **11/20**. The other seven results
are retained from the exact same model/config/processor and runtime bytes in
previous N20 batches. Combined: **184/200 (92%)** across ten tasks; this is a
combination of batches, not one simultaneous ten-task rerun or an official
held-out leaderboard score.

| Task | Successes / episodes | Peak GPU memory (MiB) |
| --- | --- | --- |
| click_bell | 20/20 | 13405 |
| handle_basket | 19/20 | 13665 |
| water_pouring | 20/20 | 13434 |
| table_rearrangement | 20/20 | 13464 |
| items_handover | 19/20 | 13408 |
| drawer_open_place | 16/20 | 13559 |
| mixer_operating | 20/20 | 13434 |
| item_assembly | 20/20 | 13406 |
| manipulate_pipette | 19/20 | 13406 |
| sample_loading | 11/20 | 13407 |

The three new jobs completed with exit code 0, empty native-fault logs and no
cleanup warnings. Their peak GPU memory was 13,434 MiB or less on L4. Simulator
Python 3.11 used PyTorch 2.10.0/CUDA 12.8 and NumPy 1.26.4; inference Python 3.12
used PyTorch 2.8.0/CUDA 12.6 and NumPy 2.1.3. Real CUDA kernels passed in both
environments. This validates the existing cloud environments; a fresh install
and RTX 5090 remain unverified. Historical native initialization failures remain
documented and are not claimed to have a proven root-cause fix.

All tasks use H30/p0; handover uses k20, others k29. Evidence and exact checksums
are in `verification/original3_20261012/`. The older results below are historical
records with their original checkpoints and evaluation settings.

## Runtime diagnostics update — 2026-10-12

The launcher now checks real CUDA execution separately in the simulator and model
Python environments before model loading. It clears inherited Python package paths
for these checks and defaults `OMP_NUM_THREADS` to 4, while honoring an explicit
caller setting. The confirmed worker evaluations also used 4 threads.

Failed runs retain their private simulator runtime and original exit code.
`cleanup.json` records process cleanup, `sim_process.json` identifies the simulator
process and versions, and `native_fault.log` captures native faults even when the
official expert checker suppresses stderr. Request logs hash the three actual RGB
inputs, state, instruction and output actions. Set `XR1_CAPTURE_FIRST_INPUT=1` only
when a saved first input is needed for diagnostics. Optional diagnostic writes
are best effort. These changes do not alter inference, reset timing, task selection
or the official expert filter.
During slow model startup, periodic `Timeout (0:02:00)!` stack dumps are diagnostic
snapshots; they do not stop loading. The actual launcher readiness deadline remains
900 seconds unless the caller explicitly changes `--startup-timeout`.

The earlier mixer initialization signal 11 is **not resolved**. Three independent
fresh-cache component runs passed on L4, but that does not establish a native-engine
fix. The original failed attempt is retained. The build requires separate real
episode evidence for this runtime update and preserves all historical acceptance
records; see `verification/runtime_diagnostics_20261012/` for its scope and results.
Fresh installation and RTX 5090 validation remain pending.

The selected checkpoints also completed separate 50-episode public-setting runs
before this diagnostic update: bell 50/50, basket 47/50, water 50/50, table 50/50,
handover 48/50, drawer 45/50 and assembly 50/50. The total is **340/350 (97.14%)
across seven tasks**; it is not a ten-task score. Exact per-run code/HF revisions,
metrics and audits are retained in `verification/selected_official_n50/`.

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
   hf download 87361Yly/robosyn-xr1-sim10 --revision fdc1c2f53a0eeb61b02459ef4fca41c7e2aa0dc4 --local-dir /absolute/path/to/models
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
     handle_basket_140k/model.pt
     handle_basket_140k/config.yaml
     items_handover_180k/model.pt
     items_handover_180k/config.yaml
     table_rearrangement_160k_ft5k/{model.pt,config.yaml}
     drawer_open_place_160k_ft10k/{model.pt,config.yaml}
     click_bell_160k_ft3k/{model.pt,config.yaml}
     mixer_operating_terminal_press_ft4k/{model.pt,config.yaml}
     manipulate_pipette_160k_ft20k/{model.pt,config.yaml}
     sample_loading_160k_ft20k/{model.pt,config.yaml}
     processor/...
   ```

   The eleven retained model files occupy about 118.72 GB in total (ten selected plus the historical shared checkpoint). Each evaluation loads
   only the checkpoint selected for that task.

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
assertions cause immediate failure with the log preserved. Successful runtime copies
are removed after child processes have stopped; failed copies are retained for diagnosis.

For each of the ten tasks below, substitute the task name and use a distinct output
directory. Use the organizer's prescribed episode count and seed/settings; the
two-episode command above is an integration acceptance run, not an official score.

| Task | Artifact | Execute steps |
|---|---|---:|
| click_bell | click_bell_160k_ft3k | 29 |
| handle_basket | handle_basket_140k | 29 |
| water_pouring | water_pouring_160k_ft5k | 29 |
| table_rearrangement | table_rearrangement_160k_ft5k | 29 |
| items_handover | items_handover_180k | 20 |
| drawer_open_place | drawer_open_place_160k_ft10k | 29 |
| mixer_operating | mixer_operating_terminal_press_ft4k | 29 |
| item_assembly | item_assembly_160k_ft10k | 29 |
| manipulate_pipette | manipulate_pipette_160k_ft20k | 29 |
| sample_loading | sample_loading_160k_ft20k | 29 |

## Interface and evaluation behavior

- Inputs: three RGB views (`cam_high`, `cam_left_wrist`, `cam_right_wrist`), 14D
  robot joint/gripper state, and the task instruction supplied by the evaluator.
- Output: horizon 30, 14D absolute joint targets; execute up to 20 actions for handover
  and 29 for other tasks, stopping
  on success or termination. Prefix length is zero. The eight normalized-gripper
  tasks and the two raw-gripper tasks retain their distinct data contracts.
- Each episode clears client history and starts a fresh policy RNG stream at seed
  20260908. The policy does not read environment seed IDs, object poses, depth or
  camera intrinsics. Task identity selects the declared checkpoint, execute steps and gripper convention.
- Reported inference time includes RGB preprocessing, transport, model computation
  and action encoding. It excludes simulator stepping. The model and simulator run
  on the same selected GPU; CPU physics is selected by the supplied adapter config.
- Per-episode traces and request logs are best-effort diagnostics. The actual
  evaluation metrics and process exit are mandatory acceptance evidence.
- The package supports a caller-provided official evaluator and forwards hidden
  test overrides. No fixed public episode manifest is included.

## Historical shared-checkpoint acceptance — 2026-10-09

The packaged SDPA runtime completed **50/50 successful click_bell episodes** on
NVIDIA L20: average 48.92 action steps (range 45–51), two model calls per episode.
The official expert filter checked 76 candidate scenes and skipped 26 before
policy execution. No policy failures were filtered out. All 50 videos, episode
RNG/chunk resets, metrics and process cleanup passed the integrity audit.

The official evaluator/config/assets were pinned to commit
`8818c82df5fa2522809b8105e7018950dcc9a9a8`; use that revision for reproduction.
The run explicitly used `eval_reset_sync_steps=0`. This publication makes **0 the
adapter default**, replacing the historical two-episode acceptance default of 10.
Model generation logic is unchanged; the build verifies that the published
configuration equals the effective configuration used in the 50-episode run.
The new teardown-order wrapper was not part of that historical bell run and is
validated separately by the assembly acceptance report. This shared-bell result
does not establish a score for the newly selected bell+3k checkpoint.

See `verification/bell50_audit.json` and `verification/bell50_metrics.json` for
historical shared-checkpoint evidence. `verification/acceptance.json` and `verification/RESULTS.md`
retain the earlier two-episode history, whose reset setting was 10.

This historical record verifies the public shared-checkpoint click_bell pipeline on
the existing L20 environments. It does not cover all ten selected policies; current
selected-policy acceptance is recorded in `verification/original3_20261012/`. A fresh
CUDA 12.8 install, RTX 5090 and the organizer's held-out scenes remain unverified.

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

## Basket and handover selection — 2026-10-11

`handle_basket_140k` and `items_handover_180k` are original H30 checkpoints from
the same ten-task training run as shared 160k. Their exact source/export digests,
normalization and tensor equality checks are in the corresponding
`verification/*_export.json` files. Historical candidate evaluations on L4
completed **50/50 basket** and **49/50 handover**; the metrics, checkpoint audits
and runtime patches are preserved under `verification/selected_history/`.
Those runs used a separate evaluation runner, including failure-budget stopping
(which did not truncate either selected run), and are not 50-episode tests of
this standalone release.

`task_mapping.json` now selects both weights and execute steps. Handover executes
**20** of each 30 predicted actions; all other tasks execute **29**. Prefix remains
zero. The direct `get_model` entry point and launcher read the same mapping;
`xr1_execute_steps: null` means select from the mapping, and conflicting explicit
values fail. The launcher records and forwards the selected k. Observations,
action conversion, RNG resets and success/termination handling are unchanged.
Tests cover actual k20/k29 replanning boundaries and early success termination.
At official reset 0, the packaged runtime passed **1/1 basket** (332 steps,
12 model calls; 11 blocks of 29 steps plus 13) and **1/1 handover** (318 steps,
16 model calls; 15 blocks of 20 steps plus 18). Each passed the 351-file official
identity check, finite action/video/metrics validation and process cleanup.
A shared-checkpoint bell regression also passed **1/1**, 47 steps / two calls
(actual blocks 29 + 18). Current acceptance is recorded separately for all three
runs. Historical assembly/water reports retain the
older adapter hashes; they are not new runs of this adapter revision.

## Seven confirmed task selections — 2026-10-11

Table rearrangement adds the shared160k+5k checkpoint and drawer placement adds
shared160k+10k, each trained on its official 1,000-episode task dataset. Existing
selected weights are retained byte-for-byte. Bell now selects shared160k+3k;
assembly remains the explicitly selected shared160k+10k. The old shared artifact
is retained for historical provenance; no current task selects it.

The selected table and drawer checkpoints have historical results of 50/50 and
48/50; the newly selected bell+3k also has historical 50/50. Retained runner hashes prove those runs used reset synchronization 10,
plus recording and native-thread compatibility changes. The submission uses
reset 0 and the SDK's default thread mode. The historical scores are selection
evidence, not completed 50-episode evaluations of this release.
`verification/selected_history/` preserves the selection records, shard contracts,
metrics, runtime patches and exact historical launch script. Export and current
packaged-runtime acceptance reports are kept separately. Build-time checks bind
new task acceptance to the current mapping and exact runtime bytes; previous
unchanged tasks retain their original reports and exact prior mapping evidence.

The three added checkpoints passed one real episode each through the packaged
runtime at official reset 0 on the existing L20 environment:

| Checkpoint | Successes / episodes | Action steps / official limit | Model calls |
|---|---:|---:|---:|
| table160k+5k | 1/1 | 145/361 | 5 |
| drawer160k+10k | 1/1 | 406/900 | 14 |
| bell160k+3k | 1/1 | 47/361 | 2 |

Each passed official-file identity, candidate RNG, finite action/chunk checks,
video, metrics and process cleanup. See the task-specific acceptance and metrics
files in `verification/`. These are integration checks, not new 50-episode results.
The initial table staging attempt failed before policy execution because a copy
filter omitted the packaged model source; complete source-hash preflight and
module-origin checks were added before the successful run. No inference code,
weights or scoring settings changed during that repair. Shared-GPU memory was
checked before the drawer run; no other workloads were terminated.

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
See `SUBMISSION_TEXT.md` for copy-ready field text and `verification/README.md`
for the current and historical validation boundaries. No official evaluation request is sent by publishing these repositories.

## Provenance

Model architecture and inference utilities derive from Xiaomi-Robotics-1 and retain
its Apache-2.0 license. `provenance.json` records original file hashes and deployment
modifications. A source archive can be generated with `python tools/build_release.py`;
large weights and caches are not included in that archive.

## Latest mixer and original pipette/sample selections — 2026-10-12

The user confirmed the latest mixer terminal-press +4k, initialized from mixer160k+10k; pipette and sample loading retain their original shared160k+20k checkpoints. The original checkpoints are exported with exact tensor equality and their own training normalization. Mixer used the official 1,000 episodes plus 100 newly generated terminal-press expert episodes at a 1:1 frame-sampling ratio; pipette used additional generated expert trajectories (`expert_fulldata_20261007`), and sample loading used the original redesigned expert dataset (`expert_redesign_20261007_add800_v1`). No later observer or replay fine-tune is selected for pipette or sample loading.

Historical reset10 selection scores (45/50, 46/50 and 23/50 respectively) are not scores for the official-reset submission pipeline. The previous full N20 total 169/200 used shared160k for these three tasks and does not describe this new mapping. New reset0 acceptance is recorded separately in `verification/original3_20261012/`. Fresh installation and organizer-held-out evaluation remain unverified.
