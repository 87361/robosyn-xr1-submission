# Submission acceptance — 2026-10-09

**PASS: two complete click_bell episodes, 2/2 success.** This is an integration
acceptance result, not an official competition score.

| Episode | Scene seed | Result | Action steps | Model calls |
|---|---:|---|---:|---:|
| 1 | 209652396 | Success | 47 / 361 | 2 |
| 2 | 398764591 | Success | 50 / 361 | 2 |

Official expert feasibility filtering accepted both candidates. Average action
steps: 48.5. Average inference latency including first-call overhead: 0.946798 s
over four calls; 1.893596 s per episode. The first call took 2.776 s; subsequent
calls took approximately 0.328–0.351 s. These are L20 measurements.

## Verified chain

- Offline model/processor load from the exported tensor-only checkpoint.
- Unmodified official evaluator at commit
  `8818c82df5fa2522809b8105e7018950dcc9a9a8`, with all 210 official asset blobs verified.
- Actual three RGB observations and 14-dimensional state through the packaged
  adapter, socket transport, XR1 inference and action conversion to the simulator.
- Per-episode policy RNG reset to 20260908, chunk indices starting at zero,
  horizon 30 / execute 29 / prefix 0.
- Two valid 1920×480 videos, 48 and 51 frames, showing the three camera views.
  The second episode's final frame was visually inspected.
- Official success metrics, evaluator exit 0, and inference listener cleanup.

The GPU was NVIDIA L20, driver 535.261.03. Inference used PyTorch 2.8.0+cu126,
Transformers 4.57.1 and SDPA. Simulator used EmbodiChain v0.2.4, DexSim 0.4.3 and
PyTorch 2.7.1+cu126. Exact inference versions and source hashes are in
`acceptance.json`; model hashes are in `artifact_manifest.json`.

## Integration issues resolved

The initial sparse challenge checkout lacked assets. The acceptance setup now
links only official assets after Git-blob verification; the launcher checks task
asset references before GPU startup.

DexSim failed on long material/cache paths. The launcher now creates an isolated
short-path simulator copy, retaining original meshes, task definitions and
evaluator. Its sole simulator-source change redirects the cache directory and is
hash-recorded. An actual environment step passed before the full model run.
Original failure logs remain in cloud storage.

## Boundary

This release maps all ten tasks but only `click_bell` received a full packaged
end-to-end run here. The existing original 160k checkpoint evaluation is separate
from this acceptance. The fresh CUDA 12.8 installer, RTX 5090, held-out scenes and
all-ten-task packaged SDPA evaluation have not been validated. Repositories have
not been published, and no official evaluation request has been submitted.
