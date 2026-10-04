# Compiled GR00T LIBERO evaluation: authoritative agent runbook

This is the single source of truth for saved-checkpoint LIBERO benchmarks in
minimal-groot. The user selected **compiled evaluation** for the bulk campaign.
Use the command in section 4 with `--compile`; omitting it runs eager inference.
Do not start a bulk queue unless the user supplies/authorizes its checkpoint list.

## 1. Non-negotiable execution path

- Repository: `https://github.com/RahulSChand/minimal-groot`.
- Branch: `feat/gr00t-multiversion-libero-training`.
- Entry point: `.venv/bin/python -m gr00t.eval.evaluate_checkpoint`.
- Native path: `Gr00tPolicy -> Gr00tSimPolicyWrapper -> LiberoEnv`.
- Compilation changes only `model.action_head.model.forward`. Do not reimplement
  inference, image preprocessing, action decoding, attention masks, or diffusion.
- The native policy returns decoded RLDS gripper values. **Only LiberoEnv.step**
  converts them to simulator commands using `-sign(2*g - 1)`. Never add gripper
  conversion to a policy, server, client, queue runner, or experiment script.

Do NOT use `ReferenceLiberoPolicy`, `reference_simulator`, `post_train_vla`,
an archived checkpoint's `runtime/`, or an ad-hoc HF evaluation adapter.
Do NOT assemble a benchmark from `run_gr00t_server.py` plus
`rollout_policy.py`: those remain low-level APIs with different defaults.
The canonical launcher starts its own loopback server and simulator; do not
start a second server. No post_train_vla/openpi/PaliGemma setup is needed for
this GR00T-only workflow, even if a general cluster-bootstrap prompt lists them.

| Checkpoint config model_type | Generation | Backend chosen by --compile |
|---|---|---|
| gr00t_n1_5 | N1.5 | Inductor, reduce-overhead, static shapes, fullgraph |
| Gr00tN1d6 | N1.6 | Inductor, reduce-overhead, static shapes, fullgraph |
| Gr00tN1d7 | N1.7 | cudagraphs, static shapes, fullgraph |

The launcher reads the checkpoint config; no manual version/backend flag is
needed. N1 is NOT supported by `--compile`. Do not silently run it eagerly or
change the N1.7 backend to Inductor. Report unsupported models or compiler errors.

## 2. Prepare each machine once

Read the provider's instance guide. Check free GPUs/VRAM, CPU quota, disk space,
download speed, and NVIDIA EGL rendering. Do not replace host NVIDIA drivers.
Preserve unrelated processes and dirty worktrees. Use a new checkout/worktree
rather than resetting an existing user's checkout.

For a fresh checkout, run in Bash from a chosen parent directory:

```bash
git clone --branch feat/gr00t-multiversion-libero-training --single-branch \
  https://github.com/RahulSChand/minimal-groot.git
cd minimal-groot
git merge-base --is-ancestor aec1ff3223c78d4d043bad61a1fd682ec381c12c HEAD
git rev-parse HEAD
uv sync --frozen --extra performance --extra eval
bash gr00t/eval/sim/LIBERO/setup_libero.sh
.venv/bin/python -m gr00t.eval.evaluate_checkpoint --help
```

Use one agreed, exact Git SHA on every machine. The ancestor check above rejects
code predating the native-path consolidation; it does not pin a fleet by itself.
Record the chosen SHA in the campaign manifest and check it on every machine.
Do not pull, upgrade packages, or edit inference code halfway through a queue.

The setup uses Python 3.12, the locked model dependencies (PyTorch 2.9.0/CUDA
12.8 on Linux), and the separately pinned LIBERO simulator environment.
The model launcher uses `.venv/bin/python`; its simulator uses
`gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python`. Do not swap these interpreters.
The simulator setup performs a short native environment smoke check.

## 3. Download the assigned checkpoint, pinned to a Hub revision

The input queue must explicitly identify HF repo, immutable revision, checkpoint
subfolder, suite, output ID, and owning machine/GPU. Do not guess suite from
model generation. Goal checkpoints use `libero_goal`; Object checkpoints use
`libero_object`. Other supported suites are `libero_spatial` and `libero_10`.

Example for the N1.6 checkpoint used in validation (replace the assignments for
other queue entries, keeping the folder/revision pairing correct):

```bash
EVAL_HF_REPO=Chand0320/groot-libero-goal-object-trajectory-efficiency
EVAL_HF_REV=47c9dd8aa90e0f994177ed32735e3b53e2a408bb
EVAL_HF_SUBDIR=n1d6/goal/seed-043/trajectories-050/epoch-007
EVAL_DOWNLOAD_DIR="$PWD/checkpoints/groot-libero"

.venv/bin/hf download "$EVAL_HF_REPO" \
  --revision "$EVAL_HF_REV" --include "$EVAL_HF_SUBDIR/*" \
  --local-dir "$EVAL_DOWNLOAD_DIR"

EVAL_CHECKPOINT="$EVAL_DOWNLOAD_DIR/$EVAL_HF_SUBDIR"
test -f "$EVAL_CHECKPOINT/config.json"
test -f "$EVAL_CHECKPOINT/embodiment_id.json"
```

Pass the actual epoch directory to `--checkpoint`, not the download root.
Retain all weights/shards, config, processor assets, statistics and embodiment
mapping. Processor files may be at the root or its native `processor/` directory.
Verify available checksums. Strict loading rejects missing/unexpected/mismatched
weights: do not bypass it, edit the checkpoint, or substitute a base model.

N1.7 may require gated Cosmos/Qwen processor assets in its bundled `vlm_assets`
or the local HF cache. Resolve access/assets using its native processor before
queuing; never substitute an Eagle/N1.6 processor. Do not enable HF offline mode
until all required assets are local. Authenticate without displaying tokens.

Download assigned folders only; use bounded caching instead of downloading all
250 checkpoints everywhere. Do not delete weights without an approved retention
policy. Never upload results/checkpoints unless requested.

## 4. Exact compiled benchmark command

Run from the repository root. The following Bash command is the production
baseline: **all ten subtasks x 40 initial states = 400 episodes per checkpoint**.
Replace the machine-specific assignments; keep the protocol flags unchanged.

```bash
set -euo pipefail
# EVAL_CHECKPOINT is the absolute epoch-directory path from section 3.
EVAL_SUITE=libero_goal
EVAL_RUN_ID=n1d6_goal_s043_t050_e007_compiled_attempt01
EVAL_GPU=0
EVAL_PORT=8765
EVAL_OUTPUT="$PWD/outputs/libero-eval/$EVAL_RUN_ID"
EVAL_SIM_PYTHON="$PWD/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python"

test -f "$EVAL_CHECKPOINT/config.json"
test -x "$EVAL_SIM_PYTHON"
test ! -e "$EVAL_OUTPUT"
mkdir -p "$(dirname "$EVAL_OUTPUT")" "$PWD/.cache/torchinductor"

env -u CUDA_VISIBLE_DEVICES -u GR00T_PAIRED_NOISE \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  TORCHINDUCTOR_COMPILE_THREADS=4 \
  TORCHINDUCTOR_CACHE_DIR="$PWD/.cache/torchinductor" \
  .venv/bin/python -u -m gr00t.eval.evaluate_checkpoint \
  --checkpoint "$EVAL_CHECKPOINT" \
  --suite "$EVAL_SUITE" \
  --eval-python "$EVAL_SIM_PYTHON" \
  --gpu "$EVAL_GPU" --port "$EVAL_PORT" \
  --workers 4 --max-batch-size 4 \
  --episodes-per-task 40 --seed 7 \
  --timeout 14400 \
  --output-dir "$EVAL_OUTPUT" \
  --compile
```

Important meanings, not optional interpretation:

- **Do not add `--task-id` for a full evaluation.** Its absence selects all tasks.
- No video flag is necessary: the canonical simulator does not save videos.
  Do not use a video-recording wrapper or invent `--save-video` arguments.
- GPU EGL rendering is configured by the launcher and checked for NVIDIA.
  Do not switch to CPU/OSMesa/software rendering or override the simulator.
- `--gpu` is the physical host GPU index for both model inference and rendering.
  Do not remap it with an outer `CUDA_VISIBLE_DEVICES`.
- Four workers/max batch four is an explicit campaign starting configuration,
  not a claim of optimal throughput on every machine. Keep it consistent.
  If it fails/OOMs, report it; do not silently change batch size or protocol.
- Fixed protocol: source images 256px, five-action replanning, ten settling
  steps, initial states 0–39, environment/run seed 7. Step limits are Spatial
  220, Object 280, Goal 300, LIBERO-10 520. Native checkpoint inference uses BF16.
- `--timeout 14400` bounds the rollout phase only (four hours). Loading/warmup
  are additional. Give the managed job a separate total-runtime bound if needed.
- Output directories must not already exist. The launcher creates them.
  A failed run is NOT resumable by passing its directory again. Retain it and
  create a distinct retry ID; never overwrite or label it complete.

For a second GPU, use another checkpoint/run ID, `EVAL_GPU=1`,
`EVAL_PORT=8766`, and another output directory. One active checkpoint job per
physical GPU; both GPUs may infer AND render independently. Do not split one
checkpoint across an inference GPU and a separate rendering GPU for this protocol.

Before the bulk queue, a short smoke command may replace only
`--episodes-per-task 40` with `--episodes-per-task 1 --task-id 7` and use a
different output ID. Label it SMOKE, not a benchmark; restore the full command
afterward. Do not extrapolate success rate or fleet completion time from it.

## 5. Multi-machine queues and process ownership

Run the exact command under the provider's managed-job mechanism (Supervisor on
the validated Vast host), from the pinned repository root. Store the resolved
absolute command and logs. In Supervisor use `autostart=false`,
`autorestart=false`, `stopasgroup=true`, `killasgroup=true`, and a sufficient
grace period (at least 20s) for owned simulator cleanup. Do not use an untracked
background shell, repeatedly restart into the same output directory, or expose
the loopback inference port publicly.

Each GPU executes its assigned checkpoints sequentially; GPUs/machines may run
in parallel. Partition the queue so exactly one worker owns each
(repo, HF revision, checkpoint folder, suite, protocol) entry. Use unique run
IDs and per-host ports for concurrent jobs. A directory's existence is not a
completion receipt. Run section 6 before marking an item complete or skipping it.

Record Git SHA, HF revision/subfolder/checksums, command, host/GPU identity,
library versions, start/end times, exit status and artifact location alongside
each queue entry. Persist artifacts off ephemeral machines. Shared CPU resources,
batch size and warmup matter: the small A100 timing test does not establish an
eight-machine deadline or production throughput. Do not change settings to meet
a deadline without recording/agreeing the new campaign configuration.

## 6. Completion gate: reject eager or incomplete results

After the command exits successfully, run this from the same repo/root shell.
It is read-only and verifies the full 400-episode compiled protocol.

```bash
.venv/bin/python - "$EVAL_OUTPUT" "$EVAL_SUITE" "$EVAL_CHECKPOINT" <<'PY'
import json
import sys
from pathlib import Path
from gr00t.eval.evaluate_checkpoint import validate_results

out = Path(sys.argv[1])
suite, checkpoint = sys.argv[2], str(Path(sys.argv[3]).absolute())
read = lambda name: json.loads((out / name).read_text())
run, summary, manifest = read("run.json"), read("summary.json"), read("suite.json")
assert run["status"] == summary["status"] == "complete"
assert run["compile_model"] is True and summary["compile"] is True
assert run["checkpoint"] == summary["checkpoint"] == checkpoint
assert run["suite"] == summary["suite"] == manifest["suite"] == suite
assert run["protocol"] == summary["protocol"] == "native_libero_v1"
assert run["task_ids"] is None
assert run["episodes_per_task"] == manifest["episodes_per_task"] == 40
assert {t["task_id"] for t in manifest["tasks"]} == set(range(10))
assert len(manifest["tasks"]) == 10
assert run["workers"] == run["max_batch_size"] == run["effective_max_batch_size"] == 4
assert run["seed"] == 7 and run["model_dtype"] == "torch.bfloat16"
assert summary["save_video"] is False
assert summary["replan_steps"] == 5 and summary["wait_steps"] == 10
assert summary["render_resolution"] == 256
limits = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
assert summary["max_steps"] == limits[suite]
strict = read("strict_load.json")
assert not any(strict.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"))
meta = run["policy_metadata"]
assert meta["policy_path"] == "Gr00tPolicy/Gr00tSimPolicyWrapper/LiberoEnv"
backends = {"gr00t_n1_5": "inductor", "Gr00tN1d6": "inductor", "Gr00tN1d7": "cudagraphs"}
assert meta["compile"]["backend"] == backends[meta["model_type"]]
assert meta["compile"]["target"] == "action_head.model.forward"
assert run["compile_graphs_after_warmup"] > 0
assert run["compile_graphs_after_rollouts"] == run["compile_graphs_after_warmup"]
checked = validate_results(out, manifest)
assert checked["episodes"] == 400
assert summary["per_task"] == checked["per_task"]
assert len(summary["per_task"]) == 10
assert all(t["episodes"] == 40 for t in summary["per_task"])
print(json.dumps({"status": "verified", "successes": summary["successes"],
                  "episodes": 400, "success_rate": summary["success_rate"],
                  "per_task": summary["per_task"]}, indent=2))
PY
```

Keep `run.json`, `strict_load.json`, `suite.json`, `warmup.json`,
`summary.json`, `episodes.jsonl`, `prepare.log`, `rollout.log` and the outer
process log. Summary JSON includes each subtask's successes, episode count and
SR. Warmup and rollout times are separate. A failed gate requires investigation:
do not remove `--compile`, bypass strict loading, fabricate completion, or
silently classify a timeout/partial result as a zero-SR completed checkpoint.

## 7. Compilation and the diagnostic experiment are different things

Compilation warms the selected prompt/batch shapes once per process, not once
per episode. Each checkpoint process still loads and warms up. Persistent
Inductor caches may reduce repeated compilation; do not promise zero startup
cost. The launcher warms batch sizes 1 through the effective cap. Graph counts
should stay unchanged during rollouts; investigate unexpected new graphs.

The published evaluator seeds model RNG once per run, after warmup. The
short matched-noise diagnostic used a separate, uncommitted experiment patch
that reseeded each episode and logged RNG/action fingerprints. **That patch is
not the bulk protocol.** Do not run from the temporary paired-test worktree,
set `GR00T_PAIRED_NOISE`, copy a probe script into the queue, or claim
`--seed 7` provides per-episode paired noise in the published code.

The completed N1.6 matched-noise check used four Goal tasks x four initial states:
eager and compiled both succeeded on 6/16, with all 16 success flags matching.
The eager repeat was stopped at the user's request after six completed episodes;
it was not a completed 16-episode control. Numeric action sequences were not
bitwise identical. This is encouraging evidence, not proof of SR equivalence for
all 250 checkpoints, other generations, or production batch sizes. Do not use
the old reference-adapter results as checkpoint-quality evidence.

Compiled is the user's selected bulk mode. Preserve its identity in result
metadata and use the same mode/protocol across comparisons. Eager is only for
an explicitly requested diagnostic/recheck, with separate outputs.
