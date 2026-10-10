# Prepared submission fields

Status: publication release; no official evaluation request has been sent.

**Artifact name:** XR1-Sim10-160k-AssemblyFT10k-H30-K29

**Experiment name:** XR1-Sim10-Shared160k-AssemblyFT10k-v2

**Short description:** Xiaomi-Robotics-1 fine-tuned on the ten official
RoboSynChallenge simulation tasks, with a task-specific assembly fine-tune. The policy predicts a 30-step sequence
of absolute joint actions from three RGB observations, proprioception and the task
instruction. The policy executes up to 29 steps before replanning.

**Code URL:** `https://github.com/87361/robosyn-xr1-submission`

**Hugging Face checkpoint URL:** `https://huggingface.co/87361Yly/robosyn-xr1-sim10`

**Data source:** Official RoboSynChallenge pre-collected Sim release: ten task
datasets with 1,000 episodes each for the shared step-160000 checkpoint, initialized
from Xiaomi-Robotics-1 pretrained weights. Nine tasks use that checkpoint.
Assembly uses 10,000 additional training steps on 1,000 generated assembly expert
trajectories, run `xr1_single_item_assembly_160k_ft10k_20261009`. The upstream
foundation model's pretraining is inherited from Xiaomi-Robotics-1.

**Run and dependency notes:** See the pinned code revision's README and
`task_mapping.json`. Nine tasks map to `shared_160k/model.pt`; `item_assembly` maps
to `item_assembly_160k_ft10k/model.pt`. Each uses the config.yaml in its own folder.
Run `policy/xr1_submission/eval.sh TASK` with the
documented simulator Python, inference Python, model directory and output path.
Use an independent Python 3.12/PyTorch 2.8 inference environment. SDPA is the default
attention backend. Tokenizer/config files are bundled for offline inference. The
policy has horizon 30, execute steps 29 and no action prefix; model RNG resets to
20260908 each episode. Reset synchronization defaults to 0, matching the
official evaluator and the 50/50 successful public click_bell evaluation. Official seeds, task settings and episode counts must come
from the organizers. The code exposes `get_model`, `eval` and `reset_model`.
The supplied launcher defers the evaluator's final native `env.close()` until
its unchanged metric calculations and file write complete; it does not change
the official evaluator file or any task/episode/inference behavior.

**Leaderboard option:** Choose at publication; no evaluation request has been sent.
