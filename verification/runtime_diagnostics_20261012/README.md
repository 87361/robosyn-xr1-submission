# Runtime diagnostic acceptance, 2026-10-12

These records cover three actual official episodes on an existing L4 queue
environment, one each for basket, sample loading and mixer. Each task's audit
checks official source/config hashes, live feasibility filtering and the complete
candidate seed sequence, finite actions, H30/k29/p0, SDPA, reset0, unchanged step
limits, original success predicates, metrics and process cleanup. A completed
runtime can still have a failed policy episode: consult `proof.json` for outcomes.

Simulator: Python 3.11, PyTorch 2.10.0+cu128, NumPy 1.26.4. Model: Python 3.12,
PyTorch 2.8.0+cu126, NumPy 2.1.3. Both real CUDA kernel checks passed. This is
neither a fresh installation nor an RTX 5090 or held-out test verification.

The first candidate preflight failed because the model Python package path leaked
into simulator Python. That attempt is preserved in `first_preflight_failure.json`;
the successful real episodes used the corrected launcher. The earlier native
mixer crash remains unresolved; see `NATIVE_INIT_ISSUE.md`. Three separate
fresh-cache component starts passed without changing renderer thread mode.

`candidate_source_sha256.json` records the exact runtime used in these episodes.
`baseline_source_sha256.json` identifies the previous public release. The release
builder checks current executable runtime bytes against the real episode snapshot,
then checks historical acceptance against its historical source bytes. Historical
results do not become new results merely because the build passed.

`replay_result.json` compares seven controlled real basket inputs: baseline and
intermediate diagnostic server actions are bitwise identical. The intermediate
server's exact source is retained in `replay_server_source.py`; later best-effort
logging/startup diagnostics were accepted in the real episodes separately. Same
input and RNG produced identical actions, while changing reset RGB changed the
actions. This does not prove why the original closed-loop basket episode failed.
