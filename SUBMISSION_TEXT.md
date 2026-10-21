# Submission fields

**Artifact name:** XR1-Sim10-Selected-H30-K20K29

**Experiment name:** XR1-Sim10-AllConfirmed-MixerTerminal4k-v6

**Short description:** Xiaomi-Robotics-1 with task-specific checkpoints for all ten RoboSynChallenge simulation tasks. The policy predicts 30 absolute joint actions from three RGB views, the 14D joint/gripper state and the task instruction, and executes up to 20 actions for items handover and 29 for the other tasks before replanning, with no action prefix.

**Code URL:** `https://github.com/87361/robosyn-xr1-submission`; the portal entry uses the full commit URL (`.../commit/<SHA>`) of the evaluated revision.

**Hugging Face model repository:** `https://huggingface.co/87361Yly/robosyn-xr1-sim10`, revision `fdc1c2f53a0eeb61b02459ef4fca41c7e2aa0dc4`

**Per-task checkpoint URLs** (each folder contains `model.pt` and `config.yaml`):

| Task | Checkpoint URL |
| --- | --- |
| click_bell | https://huggingface.co/87361Yly/robosyn-xr1-sim10/tree/fdc1c2f53a0eeb61b02459ef4fca41c7e2aa0dc4/click_bell_160k_ft3k |
| handle_basket | https://huggingface.co/87361Yly/robosyn-xr1-sim10/tree/fdc1c2f53a0eeb61b02459ef4fca41c7e2aa0dc4/handle_basket_140k |
| water_pouring | https://huggingface.co/87361Yly/robosyn-xr1-sim10/tree/fdc1c2f53a0eeb61b02459ef4fca41c7e2aa0dc4/water_pouring_160k_ft5k |
| table_rearrangement | https://huggingface.co/87361Yly/robosyn-xr1-sim10/tree/fdc1c2f53a0eeb61b02459ef4fca41c7e2aa0dc4/table_rearrangement_160k_ft5k |
| items_handover | https://huggingface.co/87361Yly/robosyn-xr1-sim10/tree/fdc1c2f53a0eeb61b02459ef4fca41c7e2aa0dc4/items_handover_180k |
| drawer_open_place | https://huggingface.co/87361Yly/robosyn-xr1-sim10/tree/fdc1c2f53a0eeb61b02459ef4fca41c7e2aa0dc4/drawer_open_place_160k_ft10k |
| mixer_operating | https://huggingface.co/87361Yly/robosyn-xr1-sim10/tree/fdc1c2f53a0eeb61b02459ef4fca41c7e2aa0dc4/mixer_operating_terminal_press_ft4k |
| item_assembly | https://huggingface.co/87361Yly/robosyn-xr1-sim10/tree/fdc1c2f53a0eeb61b02459ef4fca41c7e2aa0dc4/item_assembly_160k_ft10k |
| manipulate_pipette | https://huggingface.co/87361Yly/robosyn-xr1-sim10/tree/fdc1c2f53a0eeb61b02459ef4fca41c7e2aa0dc4/manipulate_pipette_160k_ft20k |
| sample_loading | https://huggingface.co/87361Yly/robosyn-xr1-sim10/tree/fdc1c2f53a0eeb61b02459ef4fca41c7e2aa0dc4/sample_loading_160k_ft20k |

**Data source:** All checkpoints start from Xiaomi-Robotics-1 pretrained weights and descend from one ten-task run on the official RoboSynChallenge pre-collected Sim release (ten tasks × 1,000 episodes). Basket and handover use steps 140,000 and 180,000 of that run. Bell, water pouring, table rearrangement and drawer placement are fine-tuned from step 160,000 for 3,000, 5,000, 5,000 and 10,000 steps on their official 1,000-episode datasets. Assembly is fine-tuned from step 160,000 for 10,000 steps on 1,000 generated assembly expert trajectories. Pipette and sample loading are fine-tuned from step 160,000 for 20,000 steps on generated expert trajectories (`expert_fulldata_20261007` and `expert_redesign_20261007_add800_v1`). Mixer starts from a single-task mixer model (step 160,000 + 10,000 steps) and is fine-tuned 4,000 more steps on the official 1,000 episodes plus 100 generated terminal-press episodes, sampled 1:1 by frames.

**Changes in this revision (correction for evaluation `eval_d75b898cb35548`):** All submission runtime files are contained in `policy/xr1_submission/`. The entry point directly runs the unchanged official evaluator with `EMBODICHAIN_SIM_EXIT_PROCESS=0`; the adapter starts its separate Python inference worker following the official SmolVLA pattern. The former launcher, simulator-copy/cache patch and evaluator wrapper are removed. An exit hook stops the worker, releases registered rigid constraints using the official API while the world is alive, then calls the existing official SimulationManager.flush_cleanup_queue() to finish deferred cleanup. Checkpoints, task mapping, observation/action logic, execution steps and policy RNG are unchanged.

**Run and dependency notes:**

1. Copy the complete `policy/xr1_submission/` into the official checkout's `policy/` directory.
2. Use an existing compatible Python 3.12 model environment, or create a new one with `bash policy/xr1_submission/tools/install_inference.sh [NEW_ENV_DIR]`. The default is policy-local `.xr1_env`; the fresh installer selects PyTorch 2.8.0 CUDA 12.8 wheels and SDPA. It has not been tested as a fresh installation in this acceptance.
3. Download the complete model repository: `hf download 87361Yly/robosyn-xr1-sim10 --revision fdc1c2f53a0eeb61b02459ef4fca41c7e2aa0dc4 --local-dir MODEL_DIR`. It includes all checkpoints/configs, `manifest.json` and the shared `processor/`. The adapter hashes only the selected task's model and config at startup.
4. With the official simulator environment and its data roots configured, run:
   `PYTHON_BIN=SIM_PYTHON XR1_PYTHON=MODEL_PYTHON bash policy/xr1_submission/eval.sh TASK SETTING MODEL_DIR GPU --max_episodes N --seed SEED --eval_result_dir NEW_OUTPUT_DIR --xr1_log_root NEW_OUTPUT_DIR/server`
   `EMBODICHAIN_ROOT` defaults to the checkout's sibling `EmbodiChain/`. Use actual executable paths for both Python variables. The policy directory may be renamed; the entry point derives its name.

The official evaluator controls task configuration, episode count, seed, expert feasibility filtering and scoring. Failed episodes stay in the denominator. Policy RNG restarts at seed 20260908 each episode; reset synchronization is 0. Official `evaluation_metrics.json` is written in the evaluator's nested directory below `NEW_OUTPUT_DIR`. The adapter records the selected checkpoint, worker port and request logs, and cleans up its own inference process.

On Baidu, environment-only import registration and a short data-root alias to the same PFS assets resolved the startup blockers, without upgrading CUDA or modifying simulator/evaluator sources. Original robot meshes were retained.

All 10 tasks completed one episode and passed runtime/cleanup acceptance; 9/10 episodes succeeded. Every evaluator exited 0, wrote official metrics, matched request counts and execution contracts, and released the worker, port and GPU memory; native-fault logs were empty. All ten episodes ran afresh with this exact policy revision.
Historical public-evaluator results remain 184/200 from the original separate N20 batches; they are not hidden-test scores or a new score for this adapter. Fresh installation and RTX 5090 hardware have not been verified.
