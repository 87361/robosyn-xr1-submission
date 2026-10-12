---
license: apache-2.0
base_model: XiaomiRobotics/Xiaomi-Robotics-1-5B
tags:
- robotics
- robosynchallenge
- vision-language-action
---

# XR1 RoboSynChallenge Sim10: selected task checkpoints

The shared step-160000 checkpoint was trained from Xiaomi-Robotics-1 on the
official Sim release (1,000 episodes per task). It initializes the selected task fine-tunes; no current task uses the shared checkpoint directly.
Assembly uses a separate checkpoint fine-tuned for 10,000 additional steps on
1,000 generated assembly expert trajectories. Water pouring uses 5,000 further steps on the official
1,000-episode water dataset. Basket selects the original H30 140k checkpoint, and handover selects H30 180k.
Table uses shared160k+5k and drawer shared160k+10k, each fine-tuned on its official 1,000-episode task dataset. Bell selects the shared160k+3k fine-tune on the official 1,000-episode bell dataset. All selected checkpoints include their
exact normalization statistics and share the bundled offline processor.

Use the paired code repository `87361/robosyn-xr1-submission` and its pinned commit.
The code README documents installation and evaluation commands.

## Files and task mapping

- `shared_160k/model.pt`: model tensors only; no optimizer states or credentials.
- `shared_160k/config.yaml`: normalization statistics and model parameters.
- `item_assembly_160k_ft10k/model.pt`: assembly fine-tune tensors only.
- `item_assembly_160k_ft10k/config.yaml`: exact assembly training statistics and model parameters.
- `water_pouring_160k_ft5k/model.pt`: selected water fine-tune tensors only.
- `water_pouring_160k_ft5k/config.yaml`: exact water training statistics and model parameters.
- `handle_basket_140k/{model.pt,config.yaml}`: original H30 140k for basket.
- `items_handover_180k/{model.pt,config.yaml}`: original H30 180k for handover.
- `table_rearrangement_160k_ft5k/{model.pt,config.yaml}`: selected table fine-tune.
- `drawer_open_place_160k_ft10k/{model.pt,config.yaml}`: selected drawer fine-tune.
- `click_bell_160k_ft3k/{model.pt,config.yaml}`: selected bell fine-tune.
- `mixer_operating_terminal_press_ft4k/{model.pt,config.yaml}`: latest mixer terminal-press +4k after single-task +10k.
- `manipulate_pipette_160k_ft20k/{model.pt,config.yaml}`: original single-task pipette +20k.
- `sample_loading_160k_ft20k/{model.pt,config.yaml}`: original single-task sample loading +20k.
- `processor/`: tokenizer, chat template, image processor and Qwen model config.
- `manifest.json`: file sizes, SHA256 digests, source-checkpoint digest and processor revision.

Mixer maps to `mixer_operating_terminal_press_ft4k`, pipette to `manipulate_pipette_160k_ft20k`, and sample loading to `sample_loading_160k_ft20k`. Mixer is the latest user-selected terminal-press fine-tune; pipette and sample loading are their original single-task runs. Only `item_assembly` maps to
`item_assembly_160k_ft10k`, from run `xr1_single_item_assembly_160k_ft10k_20261009`.
`water_pouring` maps to `water_pouring_160k_ft5k`, from run
`xr1_single_water_pouring_160k_ft5k_20261009`, step 5000.

The model takes three RGB images, robot joint/gripper state and task text. It
predicts 30 absolute joint actions. The submitted adapter executes up to 20 for
`items_handover`, and 29 for all other tasks, before
replanning, uses zero action prefix, and resets the policy RNG each episode.

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
are in the paired code repository under `verification/original3_20261012/`. The older results below are historical
records with their original checkpoints and evaluation settings.

## Historical evaluation boundary

The original 160k checkpoint was evaluated on 20 public-scene episodes per task.
The paired code release's integration acceptance is recorded in its
`verification/bell50_audit.json`: 50/50 click_bell success on L20, with official
expert feasibility filtering and reset synchronization set to 0. Neither result is an official held-out competition
score. This release does not claim real-robot or RTX 5090 validation.
The bell result is not an evaluation of the assembly fine-tune. Its export
checks and any component inference checks are documented separately in the code
repository's `verification/` directory; no new assembly success rate is claimed
by packaging the checkpoint.

The selected water checkpoint achieved **50/50 public-scene successes** in the
previous recorded evaluation (H30/k29/p0, `eval_reset_sync_steps=10`). This is historical checkpoint evidence,
not a 50-episode evaluation of this release, which uses the official reset default 0. That runner included recording and
native-runtime compatibility patches; their hashes and the five shard metrics
are preserved in the paired code repository under `verification/water50/`.
Packaged-runtime acceptance at reset 0 passed one episode: 176 steps, seven
model calls on the existing L20 environment. See `verification/water_pouring_acceptance.json`.
An initial native initialization failure and the unchanged successful retry are
documented in the code README; its root cause remains unresolved.

The selected original H30 checkpoints achieved 50/50 basket at 140k/k29 and
49/50 handover at 180k/k20 in the historical L4 candidate evaluation. Their
source hashes and raw metrics are retained in `verification/selected_history/`.
These are separate historical runs; current packaged-runtime acceptance is
reported in the paired code's basket/handover and shared-regression reports.
The packaged runtime passed one episode each at official reset 0 on L20:
basket 332 steps / 12 calls, handover 318 steps / 16 calls. Actual trace blocks
verified k29 and k20 respectively, including early success termination.
The final adapter selects k from `task_mapping.json` at both launch and direct
policy construction and rejects conflicting execution settings.

The selected table/drawer checkpoints achieved historical 50/50 and 48/50 at
reset=10 with recording/native compatibility changes. Current submission reset=0
acceptance is recorded separately; those historical scores are not new official
pipeline results. Mixer uses the latest terminal-press +4k; pipette/sample loading use their original single-task weights; prior shared-fallback results are historical only. Bell now selects shared160k+3k (historical reset10 50/50); previous shared-bell reset0 50/50 is not a score for this new checkpoint.

The new packaged-runtime integration checks at official reset 0 passed one
real episode each: table 145/361 steps (5 calls), drawer 406/900 (14 calls),
bell+3k 47/361 (2 calls). Official source identity, candidate RNG, actual k29
blocks, finite actions, metrics/video and cleanup passed. These are existing-L20
single-episode checks, not new 50-episode scores or fresh-install verification.

## Attribution

Xiaomi-Robotics-1 model architecture and code: Xiaomi Corporation, Apache-2.0.
RoboSynChallenge tasks and datasets: the RoboSynChallenge organizers.
Processor/model-configuration assets originate from Qwen/Qwen3-VL-4B-Instruct;
see the upstream model's license and notices for those assets.

The selected mixer fine-tune adds 4,000 steps to the original single-task +10k, using the official 1,000 episodes plus 100 generated terminal-press expert episodes, sampled 1:1 by frames. Pipette uses additional generated expert trajectories (`expert_fulldata_20261007`); sample loading uses the original redesigned expert dataset (`expert_redesign_20261007_add800_v1`). Historical reset10 scores 45/50, 46/50 and 23/50 are selection evidence only. See the paired code release for new reset0 acceptance.
