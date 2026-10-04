# Canonical GR00T LIBERO evaluation runbook

Use `python -m gr00t.eval.evaluate_checkpoint` from minimal-groot for all saved
checkpoint benchmarks. See README.md, “LIBERO checkpoint evaluation — one canonical
path”. This supersedes the former ReferenceLiberoPolicy/post_train_vla procedure.

## Setup

Read the GPU provider's instance guide. Verify available GPUs, rendering, network
and storage before downloading weights. Use HTTPS authentication without exposing
tokens. On the multiversion branch, install `uv sync --extra performance --extra eval`
and run `bash gr00t/eval/sim/LIBERO/setup_libero.sh`. The idempotent setup pins
LIBERO and simulator dependencies and smoke-tests the native LiberoEnv. Do not
replace host NVIDIA drivers. No post_train_vla or openpi installation is required.

## One inference and environment contract

Saved checkpoint and live-training evaluation use Gr00tPolicy,
Gr00tSimPolicyWrapper, and LiberoEnv. Keep checkpoint-specific native model and
processor classes, attention masks, action horizons and statistics. Reject missing,
unexpected or mismatched weights. The policy returns decoded RLDS gripper values;
only LiberoEnv.step converts them to controller commands. Never add a conversion
in a policy adapter, transport or launcher.

The old ReferenceLiberoPolicy and reference_simulator entry points are retired.
The legacy evaluate_live_model import delegates to the canonical implementation.
Custom low-level rollout APIs use the same native policy/environment, but their
default budgets are not the benchmark protocol.

## Protocol and correctness

Defaults: all ten tasks, 40 initial states per task, seed 7, five-step replanning,
ten settling steps, 256px images, NVIDIA EGL, no videos. Policy-step limits:
Spatial 220, Object 280, Goal 300, LIBERO-10 520. Native BF16 checkpoint inference
is recorded in run.json. Synchronous inference batches are bounded by workers,
max-batch-size and episode count. For paired eager/compile checks, use one worker
to avoid batch-order differences; small subsets are not full 400-rollout results.

Compilation is optional via --compile for N1.5/N1.6/N1.7. Eager and compiled runs
use the same native policy and simulator path, differing only in transformer
compilation and warmup. Warmup covers selected prompt/batch shapes and preserves
RNG. Report warmup separately from rollouts. Record graph counts and fail on
compiler errors or skipped capture. Numeric differences and success-rate parity
must be tested, not assumed.

Use separate output directories, GPU indices and ports for independent jobs.
A complete full-suite result requires 400 unique task/initial-state pairs, no
errors, and summary agreement. Summary JSON must include per-subtask successes
and SR. Preserve run settings, strict-load report, raw episodes and logs.
Historical reference-adapter SRs before the gripper fix are not trustworthy
checkpoint-quality evidence; do not mix them with canonical results.

## Provenance and long jobs

Download only the requested checkpoint folder at a pinned Hub revision and verify
its weights/assets. Check suite, generation, normalization and embodiment mapping.
Do not silently bypass archival runtime-integrity requirements. Keep required
processor assets available locally for offline evaluation.

Run long queues under Supervisor, bound disk usage, and retain failed checkpoints
for diagnosis. Never delete downloaded weights or upload results without an
explicit queue policy. Save artifacts off ephemeral instances. Do not commit
credentials, downloaded weights or machine-specific assignments to the repository.
