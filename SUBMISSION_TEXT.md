# Prepared submission fields

Status: publication release; no official evaluation request has been sent.

**Artifact name:** XR1-Sim10-Selected-H30-K20K29

**Experiment name:** XR1-Sim10-ConfirmedSeven-v5

**Short description:** Xiaomi-Robotics-1 fine-tuned on the ten official
RoboSynChallenge simulation tasks, with task-specific bell, assembly, water-pouring, table-rearrangement and drawer-placement fine-tunes. The policy predicts a 30-step sequence
of absolute joint actions from three RGB observations, proprioception and the task
instruction. The policy executes up to 20 steps for item handover and 29 for other tasks before replanning.

**Code URL:** `https://github.com/87361/robosyn-xr1-submission`

**Hugging Face checkpoint URL:** `https://huggingface.co/87361Yly/robosyn-xr1-sim10`

**Data source:** Official RoboSynChallenge pre-collected Sim release: ten task
datasets with 1,000 episodes each for the shared step-160000 checkpoint, initialized
from Xiaomi-Robotics-1 pretrained weights. Three unconfirmed tasks use that checkpoint.
Basket and handover use the original 140k and 180k checkpoints from that same
ten-task training run, respectively. Assembly uses 10,000 additional training steps on 1,000 generated assembly expert
trajectories, run `xr1_single_item_assembly_160k_ft10k_20261009`.
Water pouring uses 5,000 additional steps on the official 1,000-episode water
dataset, run `xr1_single_water_pouring_160k_ft5k_20261009`. Bell uses another 3,000 steps on its official 1,000-episode dataset. Table and drawer use another 5,000 and 10,000 steps on their respective official 1,000-episode datasets. The upstream
foundation model's pretraining is inherited from Xiaomi-Robotics-1.

**Run and dependency notes:** See the pinned code revision's README and
`task_mapping.json`. Three unconfirmed tasks map to `shared_160k/model.pt`; `item_assembly` maps
to `item_assembly_160k_ft10k/model.pt`; `water_pouring` maps to
`water_pouring_160k_ft5k/model.pt`; basket maps to `handle_basket_140k/model.pt`
and handover to `items_handover_180k/model.pt`; table maps to `table_rearrangement_160k_ft5k/model.pt` and drawer to `drawer_open_place_160k_ft10k/model.pt`. Mixer, pipette and sample loading remain unconfirmed shared160k fallbacks. Bell maps to `click_bell_160k_ft3k/model.pt`. Each uses the config.yaml in its own folder.
Run `policy/xr1_submission/eval.sh TASK` with the
documented simulator Python, inference Python, model directory and output path.
Use an independent Python 3.12/PyTorch 2.8 inference environment. SDPA is the default
attention backend. Tokenizer/config files are bundled for offline inference. The
policy has horizon 30, execute steps 20 for handover / 29 otherwise and no action prefix; model RNG resets to
20260908 each episode. Reset synchronization defaults to 0, matching the
official evaluator. Historical shared-bell 50/50 results do not describe the newly selected bell+3k checkpoint. Official seeds, task settings and episode counts must come
from the organizers. The code exposes `get_model`, `eval` and `reset_model`.
The supplied launcher defers the evaluator's final native `env.close()` until
its unchanged metric calculations and file write complete; it does not change
the official evaluator file or any task/episode/inference behavior.

**Leaderboard option:** Choose at publication; no evaluation request has been sent.
