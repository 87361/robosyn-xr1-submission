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
official Sim release (1,000 episodes per task). Six tasks retain this checkpoint.
Assembly uses a separate checkpoint fine-tuned for 10,000 additional steps on
1,000 generated assembly expert trajectories. Water pouring uses 5,000 further steps on the official
1,000-episode water dataset. Basket selects the original H30 140k checkpoint, and handover selects H30 180k.
All five checkpoints include their
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
- `processor/`: tokenizer, chat template, image processor and Qwen model config.
- `manifest.json`: file sizes, SHA256 digests, source-checkpoint digest and processor revision.

Six tasks map to `shared_160k`: click_bell,
table_rearrangement, drawer_open_place, mixer_operating,
manipulate_pipette, sample_loading. Only `item_assembly` maps to
`item_assembly_160k_ft10k`, from run `xr1_single_item_assembly_160k_ft10k_20261009`.
`water_pouring` maps to `water_pouring_160k_ft5k`, from run
`xr1_single_water_pouring_160k_ft5k_20261009`, step 5000.

The model takes three RGB images, robot joint/gripper state and task text. It
predicts 30 absolute joint actions. The submitted adapter executes up to 20 for
`items_handover`, and 29 for all other tasks, before
replanning, uses zero action prefix, and resets the policy RNG each episode.

## Evaluation boundary

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

## Attribution

Xiaomi-Robotics-1 model architecture and code: Xiaomi Corporation, Apache-2.0.
RoboSynChallenge tasks and datasets: the RoboSynChallenge organizers.
Processor/model-configuration assets originate from Qwen/Qwen3-VL-4B-Instruct;
see the upstream model's license and notices for those assets.
