---
license: apache-2.0
base_model: XiaomiRobotics/Xiaomi-Robotics-1-5B
tags:
- robotics
- robosynchallenge
- vision-language-action
---

# XR1 RoboSynChallenge Sim10 shared 160k

An acceptance-release checkpoint for all ten official RoboSynChallenge simulation
tasks. The model was fine-tuned from Xiaomi-Robotics-1 on the official Sim release
(1,000 episodes per task). This model repository contains the shared step-160000
tensor checkpoint, exact normalization statistics, and offline processor assets.

Use the paired code repository `87361/robosyn-xr1-submission` and its pinned commit.
The code README documents installation and evaluation commands.

## Files and task mapping

- `shared_160k/model.pt`: model tensors only; no optimizer states or credentials.
- `shared_160k/config.yaml`: normalization statistics and model parameters.
- `processor/`: tokenizer, chat template, image processor and Qwen model config.
- `manifest.json`: file sizes, SHA256 digests, source-checkpoint digest and processor revision.

All ten tasks map to `shared_160k`: click_bell, handle_basket, water_pouring,
table_rearrangement, items_handover, drawer_open_place, mixer_operating,
item_assembly, manipulate_pipette, sample_loading.

The model takes three RGB images, robot joint/gripper state and task text. It
predicts 30 absolute joint actions. The submitted adapter executes up to 29 before
replanning, uses zero action prefix, and resets the policy RNG each episode.

## Evaluation boundary

The original 160k checkpoint was evaluated on 20 public-scene episodes per task.
The paired code release's integration acceptance is recorded in its
`verification/bell50_audit.json`: 50/50 click_bell success on L20, with official
expert feasibility filtering and reset synchronization set to 0. Neither result is an official held-out competition
score. This release does not claim real-robot or RTX 5090 validation.

## Attribution

Xiaomi-Robotics-1 model architecture and code: Xiaomi Corporation, Apache-2.0.
RoboSynChallenge tasks and datasets: the RoboSynChallenge organizers.
Processor/model-configuration assets originate from Qwen/Qwen3-VL-4B-Instruct;
see the upstream model's license and notices for those assets.
