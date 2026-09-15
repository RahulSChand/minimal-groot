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

### LIBERO Spatial trajectory subsets

Run independent full-model fine-tunes on 5, 10, 15, 25, and 50 trajectories:

```bash
uv sync --extra performance --extra eval
bash examples/LIBERO/setup_reference_eval.sh
bash examples/LIBERO/sample_efficiency.sh 1.5
```

To run all 20 experiments sequentially on one GPU:

```bash
for version in 1 1.5 1.6 1.7; do
  bash examples/LIBERO/sample_efficiency.sh "$version"
done
```

The launcher uses `/root/libero_spatial_post` and the policy-agnostic evaluator
in `/root/post_train_vla`. Set `BASE_MODEL_PATH`, `DATASET_ROOT`, `OUTPUT_DIR`,
`REFERENCE_PROJECT`, or `EVAL_PYTHON` to override those locations. Pass `1`, `1.6`, or
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
with 20 simulator workers, inference batches up to 8, five-step replanning,
ten settling steps, and a 220-step limit. Training stops after at least three
epochs when two consecutive epochs fail to improve the best success count,
with a limit of 15 epochs. These executable defaults supersede the older
3-patience/20-epoch prose in the reference README.

GR00T retains its own image processing, action normalization, and native action
horizons (N1/N1.5: 16, N1.6: 50, N1.7: 40, read from the checkpoint). N1 also
retains its native 16 diffusion inference steps. The evaluator
sends 256px source images for GR00T to resize. Actions already use LIBERO delta
commands and receive no extra state subtraction or gripper inversion. Shared
dataset normalization statistics remain fixed across budgets. N1 and N1.5 use the
new-embodiment projector at index 31; N1.6 and N1.7 use their LIBERO projector
at index 2.

The reference simulator environment must be available through `EVAL_PYTHON`.
The setup script pins LIBERO/MuJoCo/robosuite and shares PyTorch with the
model environment. The existing `setup_libero.sh` provides a separate,
standalone environment for the general GR00T evaluator.
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

## LIBERO simulation evaluation

LIBERO runs in a dedicated virtual environment because its Gymnasium, MuJoCo,
robosuite, and NumPy requirements differ from the model environment. Set it up
once from the repository root. On a fresh Ubuntu host, install the shared EGL
libraries first:

```bash
sudo apt update
sudo apt install libegl1-mesa-dev libglu1-mesa
```

Then create the LIBERO environment:

```bash
bash gr00t/eval/sim/LIBERO/setup_libero.sh
```

The setup script clones the upstream-pinned LIBERO commit into
`external_dependencies/LIBERO`, creates
`gr00t/eval/sim/LIBERO/libero_uv/.venv`, and performs a headless environment
smoke test. It leaves an existing LIBERO checkout untouched and fails if that
checkout is on a different commit.

The published checkpoint is stored in a nested Hugging Face repository folder,
so download that folder into a local checkpoint directory:

```bash
uv run hf download nvidia/GR00T-N1.7-LIBERO \
    --include "libero_10/config.json" \
              "libero_10/embodiment_id.json" \
              "libero_10/model-*.safetensors" \
              "libero_10/model.safetensors.index.json" \
              "libero_10/processor_config.json" \
              "libero_10/statistics.json" \
    --local-dir checkpoints/GR00T-N1.7-LIBERO
```

Start the policy server in the primary project environment:

```bash
uv run --extra eval python gr00t/eval/run_gr00t_server.py \
    --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
    --embodiment-tag LIBERO_PANDA \
    --use-sim-policy-wrapper
```

Then start a short rollout in a second terminal using the LIBERO environment:

```bash
gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python \
    gr00t/eval/rollout_policy.py \
    --n-episodes 1 \
    --policy-client-host 127.0.0.1 \
    --policy-client-port 5555 \
    --max-episode-steps 40 \
    --env-name libero_sim/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it \
    --n-action-steps 8 \
    --n-envs 1
```

For a benchmark-length evaluation, use `--max-episode-steps 720` and increase
the episode and environment counts. The checkpoint must be LIBERO-finetuned
and contain `embodiment_id.json`, `processor_config.json`, and
`statistics.json`; the generic SO100 or base checkpoint is not a substitute.

## Scope note

This extraction intentionally includes only LIBERO simulation support. Other
simulators can be added later as separate adapters without changing the
fine-tuning, policy, or shared rollout layers.
