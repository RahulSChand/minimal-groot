# Minimal GR00T N1.7

This is a focused extraction of NVIDIA Isaac GR00T N1.7 for fine-tuning and
LIBERO evaluation. The GR00T source files were copied without rewriting their
model, processor, dataset, training, inference, or rollout behavior.

See [EXTRACTION_RATIONALE.md](EXTRACTION_RATIONALE.md) for how the extraction
boundary was selected, why each major subsystem remains, and what was excluded.

## Included

- GR00T N1.7 model and Qwen3-VL backbone integration
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
- Access to the gated `nvidia/Cosmos-Reason2-2B` Hugging Face model

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
