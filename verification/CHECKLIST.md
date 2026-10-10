# Acceptance criteria

- [x] All ten official tasks have explicit checkpoint mappings.
- [x] Checkpoint export contains only the 1,135 model tensors.
- [x] Exported mean/std/q01/q99 are numerically identical to the training config.
- [x] Checksum manifest covers model/config and offline processor files.
- [x] Inference starts offline with only the packaged runtime source.
- [x] Official challenge commit is `8818c82df5fa2522809b8105e7018950dcc9a9a8`.
- [x] Cloud-resident official task assets match all 210 corresponding Git blob hashes.
- [x] Initial missing-asset failure preserved; child server cleaned up afterwards.
- [x] All ten task configurations pass local asset-reference checks.
- [x] Short-path runtime resolves DexSim native material-path assertion with original assets.
- [x] Two complete successful click_bell episodes using the official evaluator.
- [x] Each episode starts its own policy RNG stream and chunk index zero.
- [x] Valid three-camera videos and consistent raw episode metrics.
- [x] Inference listener removed after successful evaluator exit.

Only `acceptance.json` with `passed: true` establishes that the end-to-end criteria
have passed. Static checks and server initialization alone do not do so.

Outside this acceptance run: ten-task reevaluation of the packaged SDPA runtime,
RTX 5090 execution, private held-out testing, repository publication and submission.
