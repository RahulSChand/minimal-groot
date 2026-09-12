# Minimal GR00T N1.7

This is a fine-tuning-only extraction of NVIDIA Isaac GR00T N1.7. The GR00T
source files were copied without rewriting their model, processor, dataset,
training, distributed-training, or checkpoint behavior.

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

## Excluded

- Inference policies and policy server/client
- ZMQ, msgpack, and network serving
- Offline open-loop evaluation and plotting
- Simulator and real-robot evaluation
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

Development tools are available with:

```bash
uv sync --extra dev
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

## Scope note

This extraction intentionally does not provide evaluation or inference entry
points. Those can be added later as a separate package without changing the
fine-tuning core.
