# Prepared submission fields

Status: publication release; no official evaluation request has been sent.

**Artifact name:** XR1-Sim10-160k-H30-K29

**Experiment name:** XR1-Sim10-Shared160k-Submission-v1

**Short description:** Xiaomi-Robotics-1 fine-tuned on the ten official
RoboSynChallenge simulation tasks. A shared checkpoint predicts a 30-step sequence
of absolute joint actions from three RGB observations, proprioception and the task
instruction. The policy executes up to 29 steps before replanning.

**Code URL:** `https://github.com/87361/robosyn-xr1-submission`

**Hugging Face checkpoint URL:** `https://huggingface.co/87361Yly/robosyn-xr1-sim10`

**Data source:** Official RoboSynChallenge pre-collected Sim release: ten task
datasets with 1,000 episodes each. The release uses the original shared step-160000
checkpoint, initialized from Xiaomi-Robotics-1 pretrained weights. No additional
expert-generated fine-tuning data is used by this selected checkpoint. The upstream
foundation model's pretraining is inherited from Xiaomi-Robotics-1.

**Run and dependency notes:** See the pinned code revision's README and
`task_mapping.json`. All ten tasks map to `shared_160k/model.pt` and its matching
`shared_160k/config.yaml`. Run `policy/xr1_submission/eval.sh TASK` with the
documented simulator Python, inference Python, model directory and output path.
Use an independent Python 3.12/PyTorch 2.8 inference environment. SDPA is the default
attention backend. Tokenizer/config files are bundled for offline inference. The
policy has horizon 30, execute steps 29 and no action prefix; model RNG resets to
20260908 each episode. Reset synchronization defaults to 0, matching the
official evaluator and the 50/50 successful public click_bell evaluation. Official seeds, task settings and episode counts must come
from the organizers. The code exposes `get_model`, `eval` and `reset_model`.

**Leaderboard option:** Choose at publication; no evaluation request has been sent.
