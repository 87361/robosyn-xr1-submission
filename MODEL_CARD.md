---
license: apache-2.0
base_model: XiaomiRobotics/Xiaomi-Robotics-1-5B
tags:
- robotics
- robosynchallenge
- vision-language-action
---

# XR1 RoboSynChallenge Sim10: shared 160k + assembly 10k fine-tune

The shared step-160000 checkpoint was trained from Xiaomi-Robotics-1 on the
official Sim release (1,000 episodes per task). Nine tasks retain this checkpoint.
Assembly uses a separate checkpoint fine-tuned for 10,000 additional steps on
1,000 generated assembly expert trajectories. Both checkpoints include their
exact normalization statistics and share the bundled offline processor.

Use the paired code repository `87361/robosyn-xr1-submission` and its pinned commit.
The code README documents installation and evaluation commands.

## Files and task mapping

- `shared_160k/model.pt`: model tensors only; no optimizer states or credentials.
- `shared_160k/config.yaml`: normalization statistics and model parameters.
- `item_assembly_160k_ft10k/model.pt`: assembly fine-tune tensors only.
- `item_assembly_160k_ft10k/config.yaml`: exact assembly training statistics and model parameters.
- `processor/`: tokenizer, chat template, image processor and Qwen model config.
- `manifest.json`: file sizes, SHA256 digests, source-checkpoint digest and processor revision.

Nine tasks map to `shared_160k`: click_bell, handle_basket, water_pouring,
table_rearrangement, items_handover, drawer_open_place, mixer_operating,
manipulate_pipette, sample_loading. Only `item_assembly` maps to
`item_assembly_160k_ft10k`, from run `xr1_single_item_assembly_160k_ft10k_20261009`.

The model takes three RGB images, robot joint/gripper state and task text. It
predicts 30 absolute joint actions. The submitted adapter executes up to 29 before
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

## Attribution

Xiaomi-Robotics-1 model architecture and code: Xiaomi Corporation, Apache-2.0.
RoboSynChallenge tasks and datasets: the RoboSynChallenge organizers.
Processor/model-configuration assets originate from Qwen/Qwen3-VL-4B-Instruct;
see the upstream model's license and notices for those assets.
