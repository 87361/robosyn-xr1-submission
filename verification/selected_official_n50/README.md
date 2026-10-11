# Selected checkpoint evaluation, public official settings

Seven selected task checkpoints completed 50 episodes each in two independently
recorded batches before the runtime diagnostics update: 340/350 successes (97.14%).
Per-task code/HF revisions are retained in summary.json and preflight.json; these
are historical results for the listed selected checkpoints, not a new 350-episode
run of the diagnostic release. Mixer, pipette and sample loading are not covered
by this batch, so it is not a ten-task total.

All use H30, p0, SDPA, per-episode model RNG 20260908, random settings, candidate
RandomState(0), reset0 and the unchanged live official expert filter. Handover uses
k20; the other six use k29. Original task success predicates and step limits apply.
Policy failures remain in the denominator. Official revision:
8818c82df5fa2522809b8105e7018950dcc9a9a8. This is not a held-out official score,
fresh-install verification or a cross-machine deterministic replay guarantee.
