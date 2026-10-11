# Native initialization issue — reproduction record

Status: unresolved native crash; diagnostic improvements are not a native-engine fix.

The first `mixer_operating` attempt in the ten-task acceptance exited with signal 11 while creating the simulator. The inference server was already READY, but received zero model requests. The evaluator log ended immediately after CobotMagic URDF assembly. The failing launcher recorded `-11` at 2026-10-12 04:02:35 +0800.

- Official challenge: `8818c82df5fa2522809b8105e7018950dcc9a9a8`.
- Published policy: `3935166d34285543e2c4af236e31710821dd939a`.
- HF artifacts: `87361Yly/robosyn-xr1-sim10` at `d001dcf7f2a4b3fae9419a38f95d2954afc38cb4`.
- EmbodiChain source: frozen v0.2.4 snapshot; DexSim 0.4.3; NumPy 1.26.4; simulator PyTorch 2.7.1+cu126, Python 3.11; L20, driver 535.261.03.
- Task: random, seed0, one environment, CPU physics, hybrid renderer, default RENDER_SHARE_ENGINE, reset_sync_steps=0, live official feasibility filter.
- Model: shared160k, H30/k29/p0, SDPA. No policy action occurred before the crash.
- Only SDK source modification: isolated simulator-cache directory. No HOME override, task patch or thread-mode override.

Original cloud evidence: `/mnt/pfs/public/models/xr1/submission_e2e_20261012/mixer_operating/`. The failed attempt is preserved separately from the successful continuation. Kernel history contains VulkanCore/librender-hybrid.so null-pointer crashes, but there is no precise original PID mapping, so that library is a lead rather than a proven fault location for this attempt.

The diagnostic candidate enables a dedicated `native_fault.log` descriptor before environment creation, unaffected by the official expert-check stderr suppression. It records process/version metadata and retains failed simulator runtime paths. These changes improve the next failure report without changing evaluation semantics.

Investigation scripts: `scripts/validation/submission_diagnostics_20261012/env_probe.py`, `replay.py`, `accept_candidate.py`. Three fresh-cache default-thread mixer component runs and the candidate's real policy episode are recorded separately in the accompanying diagnostic report. A component pass is not proof that the sporadic crash is fixed.
