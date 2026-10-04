# Minimal GR00T N1, N1.5, N1.6, and N1.7

This is a focused extraction of NVIDIA Isaac GR00T for fine-tuning and
LIBERO evaluation. The launcher detects N1, N1.5, N1.6, or N1.7 from a local or
Hugging Face checkpoint's `config.json`. Each version retains its own model
architecture and vision/language processor.

See [EXTRACTION_RATIONALE.md](EXTRACTION_RATIONALE.md) for how the extraction
boundary was selected, why each major subsystem remains, and what was excluded.

## Included

- GR00T N1, N1.5, and N1.6 with their native Eagle backbones, and N1.7 with Qwen3-VL
- LeRobot dataset loading and video decoding
- State/action processing and normalization
- Single- and multi-dataset fine-tuning
- Single-GPU, DDP, and DeepSpeed training paths
- Standalone Hugging Face checkpoint creation
- The custom-embodiment modality configuration example
- Local and ZMQ-served policy inference
- Open-loop evaluation
- Shared closed-loop rollout and video wrappers
- LIBERO simulation evaluation

## Excluded

- SimplerEnv, RoboCasa, and real-robot evaluation
- TensorRT and ONNX deployment
- Platform Dockerfiles and installation scripts
- Dataset conversion and repair utilities

## Requirements

- Python 3.12
- Linux x86_64
- A CUDA environment compatible with PyTorch 2.9 / CUDA 12.8
- FFmpeg 4 through 7 for TorchCodec 0.8
- For N1.7, access to the gated `nvidia/Cosmos-Reason2-2B` Hugging Face model
- For N1/N1.5/N1.6, install `--extra performance` for their Flash Attention backbones

Authenticate before loading a GR00T checkpoint:

```bash
hf auth login
```

## Install

For single-GPU training with PyTorch SDPA attention:

```bash
uv sync
```

For the normal optimized multi-GPU setup:

```bash
uv sync --extra distributed --extra performance
```

Policy inference and evaluation dependencies are optional so training-only
installs remain small:

```bash
uv sync --extra eval
```

## Fine-tune

Run from this repository's root:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python \
    gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path /path/to/lerobot-dataset \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/SO100/so100_config.py \
    --num-gpus 1 \
    --output-dir ./outputs/run-1 \
    --max-steps 2000 \
    --global-batch-size 32 \
    --dataloader-num-workers 4
```

The convenience wrapper exposes the same path:

```bash
USE_WANDB=0 CUDA_VISIBLE_DEVICES=0 uv run bash examples/finetune.sh \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path /path/to/lerobot-dataset \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/SO100/so100_config.py \
    --output-dir ./outputs/run-1
```

For multiple GPUs, install the `distributed` extra and launch with `torchrun`,
or set `NUM_GPUS` when using `examples/finetune.sh`.

Use `nvidia/GR00T-N1-2B`, `nvidia/GR00T-N1.5-3B`, or `nvidia/GR00T-N1.6-3B`
as `--base-model-path` to train the older models. `--model-version N1` (or N1.5/N1.6/N1.7) optionally
checks that the supplied checkpoint matches your intended version. The loader
checks for missing, unexpected, or mismatched weights before training.

### LIBERO Long dataset

The long-horizon LIBERO suite is named `libero_10` in LIBERO and in the IPEC
LeRobot release. Use this external-video dataset; there is no separate
`libero_long` repository:

```text
Hugging Face: IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot
revision: e1a223d30b896c1613f270a2bfc63d382b3de7e1
local path: datasets/libero_10_no_noops_1.0.0_lerobot
```

Download the pinned revision from the repository root:

```bash
.venv/bin/hf download \
  IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot \
  --repo-type dataset \
  --revision e1a223d30b896c1613f270a2bfc63d382b3de7e1 \
  --local-dir datasets/libero_10_no_noops_1.0.0_lerobot
```

Keep `meta/modality.json` matched to the checked-out GR00T runtime and generate
`meta/stats.json` from this exact complete dataset before creating trajectory
subsets. Do not reuse normalization statistics from Spatial, Goal, or Object.

### LIBERO Long seed-43 trajectory campaign

The fixed campaign runs independent full-model fine-tunes for N1.5, N1.6, and
N1.7 on nested 50- and 100-trajectory subsets. Every run starts from its own
base checkpoint and trains for exactly six epochs. Prepare and validate it with:

```bash
.venv/bin/python -m gr00t.data.stats \
  --dataset-path datasets/libero_10_no_noops_1.0.0_lerobot \
  --embodiment-tag LIBERO_PANDA
.venv/bin/python experiments/prepare_libero_trajectory_views.py \
  --suite long --seed 43 --budgets 50 100
.venv/bin/python experiments/run_long_seed43.py --validate-only
```

Launch the detached campaign and monitor its persistent log:

```bash
setsid .venv/bin/python experiments/run_long_seed43.py \
  > outputs/groot-long-seed43.log 2>&1 < /dev/null &
tail -f outputs/groot-long-seed43.log
```

The six runs produce 36 inference checkpoints: two trajectory budgets times
three model versions times six epochs. Each completed epoch is uploaded beneath
`Chand0320/groot-libero-long-trajectory-efficiency`, verified against the remote
bytes, recorded in `publication_receipts/`, and then removed locally. A failed
upload leaves the checkpoint on disk and stops the campaign rather than deleting
an unverified artifact.

The H100 campaign uses microbatch 16 with three-way gradient accumulation
(effective batch 48), eight persistent data-loader workers, pinned memory,
prefetch factor 4, and fused AdamW. A bounded N1.7 benchmark measured 23.52
frames/s with 71.16 GiB peak reserved memory, versus 6.11 frames/s for the old
microbatch-8, accumulation-6, single-process loader. Microbatch 24 was only 5%
faster but reserved 78.03 GiB, so it is intentionally not used for the campaign.

### LIBERO Spatial trajectory subsets

Run independent full-model fine-tunes on 5, 10, 15, 25, and 50 trajectories:

```bash
uv sync --extra performance --extra eval
bash gr00t/eval/sim/LIBERO/setup_libero.sh
bash examples/LIBERO/sample_efficiency.sh 1.5
```

To run all 20 experiments sequentially on one GPU:

```bash
for version in 1 1.5 1.6 1.7; do
  bash examples/LIBERO/sample_efficiency.sh "$version"
done
```

The launcher uses `/root/libero_spatial_post` and the canonical native LIBERO
evaluator below. Set `BASE_MODEL_PATH`, `DATASET_ROOT`, `OUTPUT_DIR`, or
`EVAL_PYTHON` to override those locations. Pass `1`, `1.6`, or
`1.7` to select another generation. Weights can be a local directory or the
corresponding `nvidia/GR00T-N1-2B` or `nvidia/GR00T-N1.x-3B` Hugging Face repository.

The defaults follow `post_train_vla/scripts/run_sample_efficiency.sh` and its
trainer: seed 42, globally sampled nested trajectory subsets, microbatch 8,
accumulation 6 (effective batch 48), full-model AdamW, constant learning rate
`5e-5`, weight decay `0.01` on all parameters, betas `(0.9, 0.999)`, epsilon
`1e-8`, and gradient clipping at `1.0`. BF16 autocast is used with state dropout
disabled. Each epoch visits every selected frame exactly once, including short
final batches; action chunks repeat the final frame within each episode.
Every budget starts again from the base checkpoint.

Each epoch is evaluated on LIBERO Spatial task 0, initial states 0–19, seed 7,
with up to eight active simulator workers (the inference batch cap), five-step replanning,
ten settling steps, and a 220-step limit. Training stops after at least three
epochs when two consecutive epochs fail to improve the best success count,
without a fixed epoch limit. These executable defaults supersede the older
3-patience/20-epoch prose in the reference README.

GR00T retains its own image processing, action normalization, and native action
horizons (N1/N1.5: 16, N1.6: 50, N1.7: 40, read from the checkpoint). N1 also
retains its native 16 diffusion inference steps. The evaluator
sends 256px source images for GR00T to resize. Actions already use LIBERO delta
commands with no extra state subtraction; the native environment converts
RLDS gripper values to simulator commands exactly once. Shared
dataset normalization statistics remain fixed across budgets. N1 and N1.5 use the
new-embodiment projector at index 31; N1.6 and N1.7 use their LIBERO projector
at index 2.

The canonical simulator environment must be available through `EVAL_PYTHON`.
The setup script pins LIBERO/MuJoCo/robosuite and shares PyTorch with the model
environment. Both training and checkpoint evaluation use this same setup.
The live training model serves evaluation, avoiding a second GPU model copy.

```bash
# Write/validate the exact manifest and run plan without training:
bash examples/LIBERO/sample_efficiency.sh 1.5 --prepare-only

# Standard fine-tuning remains available independently of the reference project:
uv run --extra performance python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.5-3B \
  --dataset-path /root/libero_spatial_post --embodiment-tag LIBERO_PANDA \
  --modality-config-path examples/LIBERO/libero_spatial_config.py \
  --output-dir outputs/n15-custom
```

Each run records the shared manifest, selected episode IDs, effective
hyperparameters, per-epoch training losses and simulation outcomes. The best
epoch checkpoint contains weights, model config, processor, normalization
statistics, and embodiment IDs. Metrics for every epoch are retained. Use
`--checkpoint-retention best-and-last` to also keep the final epoch weights.
Checkpoints stay local and omit optimizer state; restart an
interrupted budget from base weights in a new output directory. Rerunning an
unchanged completed plan skips completed budgets. The standard fine-tuning
launcher continues to support resumable optimizer checkpoints.

### Verified Spatial campaign with every epoch on the Hub

`examples/LIBERO/run_spatial_campaign.py` runs the 20 requested experiments:
N1, N1.5, N1.6 and N1.7, each with 5, 10, 15, 25 and 50 trajectories total.
Completed runs are skipped when adding a budget; their checkpoints are retained.
It pins the four official NVIDIA base revisions, uses one seed-42 nested
manifest, and reads `/root/liber_spatial_post`. Each budget starts from its
version's base. Stopping uses two consecutive non-improvements, with ties
counting as non-improvements and no epoch cap.

Before training, `examples/LIBERO/prepare_spatial.py` validates every episode,
task and embedded image against the pinned LIBERO Spatial task definitions,
then computes normalization over the verified Spatial corpus. Its provenance
and file hashes are saved in the dataset's `meta/` directory.

Each epoch is saved, evaluated on 20 task-0 rollouts, and published under
`n<version>/trajectories-<NNN>/epoch-<EEE>/`. The upload includes native
processor assets, statistics, embodiment IDs, manifests, and compatible runtime
source. A fresh subprocess loads the downloaded checkpoint with an empty Hub
cache and networking disabled by the Hub offline settings, checks strict
weight loading, and runs action inference. Only after this succeeds is the
local epoch directory deleted. Optimizer state stays in memory during a run
and is never saved or uploaded.

Logs, evaluation results, verification receipts and run summaries are retained
under `outputs/spatial-trajectory-efficiency-20260915/`. A failed upload or load
check stops the campaign and preserves the local checkpoint for repair.

## Dataset contract

The input must be a GR00T-compatible LeRobot dataset. At minimum its `meta/`
directory must contain `info.json`, `modality.json`, `stats.json`, and task and
episode metadata. Video datasets also require their referenced video files.

The Python modality configuration is separate from the dataset's
`meta/modality.json`. It defines the model-facing video, state, action, and
language keys and their temporal delta indices. Copy and edit
`examples/SO100/so100_config.py` for a new robot embodiment.

## Checkpoint contract

Periodic `checkpoint-*` directories are standalone. In addition to model
weights and Hugging Face configuration, the training callback copies:

- `processor_config.json`
- `statistics.json`
- `embodiment_id.json`
- `experiment_cfg/`

Do not discard these files; they are required to interpret model inputs and
decode normalized actions later.

## LIBERO checkpoint evaluation — compiled bulk workflow

**Agents: read and follow the [authoritative compiled-evaluation runbook](.config/groot_libero_eval_runbook.md)
before running any checkpoint.** It contains pinned setup/download commands,
the full launch command, multi-GPU queue rules, and a copy-paste completion gate.
Root [AGENTS.md](AGENTS.md) points to the same instructions.

Use branch `feat/gr00t-multiversion-libero-training`, pinned to the same exact
Git SHA on every machine. The only supported benchmark entry point is:

```bash
# Run from minimal-groot; replace paths and choose a NEW output directory.
env -u CUDA_VISIBLE_DEVICES -u GR00T_PAIRED_NOISE \
  .venv/bin/python -u -m gr00t.eval.evaluate_checkpoint \
  --checkpoint /absolute/path/to/epoch-directory \
  --suite libero_goal \
  --eval-python "$PWD/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python" \
  --gpu 0 --port 8765 \
  --workers 4 --max-batch-size 4 \
  --episodes-per-task 40 --seed 7 --timeout 14400 \
  --output-dir /absolute/path/to/new-compiled-result-directory \
  --compile
```

This runs all ten subtasks x 40 episodes (400 total), uses native NVIDIA EGL GPU
rendering, saves no videos, and records per-subtask SR in `summary.json`.
Do not pass `--task-id` for a full benchmark. Use the suite assigned to the
checkpoint; `libero_goal` above is an example, not a default for Object models.
The launcher owns both server and simulator.

The user-selected bulk workflow **requires `--compile`**, although the CLI still
defaults to eager when that flag is omitted. N1.5/N1.6 use Inductor
reduce-overhead; N1.7 uses the cudagraphs backend automatically. N1 compilation
is unsupported. Do not silently fall back to eager.

Both modes share `Gr00tPolicy -> Gr00tSimPolicyWrapper -> LiberoEnv`.
Only `LiberoEnv.step` converts the decoded RLDS gripper values to simulator
commands. Do not add any conversion in a policy/client/server or use the retired
reference adapter, `post_train_vla`, archived checkpoint runtime code, or a
separate hand-assembled server/rollout benchmark.

For GPU 1, use `--gpu 1 --port 8766`, another assigned checkpoint and another
output directory. Keep one active checkpoint job per physical GPU. Run queues
under managed processes and validate the full result before marking it complete.
See the runbook for exact environment/cache settings, provenance, error handling,
and the distinction between bulk evaluation and temporary diagnostic experiments.

## Scope note

This extraction intentionally includes only LIBERO simulation support. Other
simulators can be added later as separate adapters without changing the
fine-tuning, policy, or shared rollout layers.
