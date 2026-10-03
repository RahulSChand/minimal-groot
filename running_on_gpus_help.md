# Running GR00T experiments on GPU machines

This document records the setup used for the GR00T N1.7 LIBERO Spatial
trajectory experiments on a single NVIDIA H100 80 GB GPU.

## Machine and environment

- Repository: `/workspace/minimal-groot`
- Environment: `/workspace/minimal-groot/.venv`
- Python 3.12
- PyTorch 2.9.0 with CUDA 12.8
- Flash Attention 2.8.3
- Hugging Face and W&B authentication are persisted for `root`.

Create the environment with:

```bash
cd /workspace/minimal-groot
uv sync --python 3.12 --locked --extra performance --extra eval
```

The finite-epoch campaign also uses `post_train_vla` from the sibling checkout:

```bash
uv pip install --python .venv/bin/python --no-deps -e /workspace/post_train_vla
```

## Models

The base checkpoints are stored under `checkpoints/`:

```text
checkpoints/GR00T-N1.5-3B
checkpoints/GR00T-N1.6-3B
checkpoints/GR00T-N1.7-3B
```

They were downloaded from the corresponding official `nvidia/GR00T-N*-3B`
Hugging Face repositories.

## LIBERO dataset metadata

The N1.7 Spatial campaign uses the NVIDIA-documented dataset at the pinned
revision below:

```text
IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot
revision: bf14d6258218d12c2e3c1a3b9922e163cdf6455d
local path: datasets/libero_spatial_no_noops_1.0.0_lerobot
```

### Why this campaign does not use `Chand0320/libero_spatial_post`

These repositories contain the same general LIBERO Spatial task family but use
different storage schemas:

```text
IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot
  observation.images.image        external video
  observation.images.wrist_image  external video
  observation.state
  action

Chand0320/libero_spatial_post
  image                            embedded in Parquet
  wrist_image                      embedded in Parquet
  state
  actions
```

The current N1.7 `LeRobotEpisodeLoader` expects the newer external-video
schema, so the campaign uses the pinned IPEC dataset. The `modality.json` and
`stats.json` stored in `Chand0320/libero_spatial_post` are correct for that
dataset, but metadata alone does not convert its embedded images into the
format expected by the current N1.7 loader.

For a fresh N1.7 run, use the pinned IPEC repository above. Alternatively,
publish a converted external-video version under the user's Hugging Face
account and pin that new dataset revision. Do not silently substitute
`Chand0320/libero_spatial_post` in the N1.7 command.

GR00T needs two metadata files beyond the downloaded data:

### `modality.json`

This is a schema mapping. It tells GR00T which dataset fields represent the
front image, wrist image, state dimensions, action dimensions, and language
instruction. Use NVIDIA's file from the GR00T source revision that matches the
runtime. For this checkout it came from:

```text
NVIDIA/Isaac-GR00T commit 51d4c89f72fda44cbf77285c6a8114b52676b8a1
examples/LIBERO/modality.json
```

It is installed as:

```text
datasets/libero_spatial_no_noops_1.0.0_lerobot/meta/modality.json
```

The runtime-authoritative copy belongs in the compatible dataset's `meta/`
directory. A matching template may also live in the code repository, but it
must not override dataset-specific field names. A copy is embedded in every
published checkpoint under `dataset_metadata/modality.json`.

### `stats.json`

This file is generated from the exact training dataset. It contains mean,
standard deviation, min, max, q01, and q99 values for `observation.state`,
`action`, and `timestamp`. GR00T uses these values to normalize states and
actions. Do not hand-edit or reuse it with a different dataset revision.

Generate it with:

```bash
cd /workspace/minimal-groot
.venv/bin/python -m gr00t.data.stats \
  --dataset-path datasets/libero_spatial_no_noops_1.0.0_lerobot \
  --embodiment-tag LIBERO_PANDA
```

Generated statistics belong with the pinned dataset artifact, not as general
source code. For this campaign they are preserved in every uploaded checkpoint
under `dataset_metadata/stats.json`, so no separate model-repository upload is
required. If publishing a reusable derived training dataset, include
`meta/stats.json` in that Hugging Face dataset repository.

### Trajectory manifests

The shared seed-43 ordering is stored at:

```text
outputs/n1d7-spatial-seed43/trajectory_manifest_seed43.json
```

The 10, 25, and 50 trajectory datasets use nested prefixes of this same
ordering. Each checkpoint includes its run-specific `trajectory_manifest.json`.
Manifests belong with experiment outputs/checkpoints because they record exactly
which episodes were used.

## N1.7 Spatial campaign

The campaign runner is:

```text
experiments/run_n1d7_spatial_seed43.py
```

It runs, sequentially:

- 10 trajectories, 7 epochs
- 25 trajectories, 7 epochs
- 50 trajectories, 7 epochs

All runs use seed 43, batch size 8, gradient accumulation 6, learning rate
`1e-5`, and fully unfrozen vision, language, projector, and action-head
parameters.

The finite-epoch trainer saves standalone BF16 checkpoints without optimizer,
scheduler, or RNG state. They are intentionally not resumable.

At every epoch boundary the campaign:

1. Saves a standalone checkpoint.
2. Uploads it to
   `Chand0320/groot-n1d7-libero-spatial-trajectory-efficiency`.
3. Verifies remote file sizes and content hashes at the resulting commit.
4. Saves a compact local publication receipt.
5. Deletes the verified local checkpoint to conserve disk.

Remote prefixes follow this structure:

```text
n1d7/spatial/seed-043/trajectories-010/epoch-001
```

## Supervisor

Long-running training is managed by Supervisor so it continues after the SSH
session disconnects and receives process-group shutdown signals correctly.

```text
service: groot_n1d7_spatial_seed43
wrapper: /opt/supervisor-scripts/groot_n1d7_spatial_seed43.sh
config:  /etc/supervisor/conf.d/groot_n1d7_spatial_seed43.conf
log:     /var/log/portal/groot_n1d7_spatial_seed43.log
```

Useful commands:

```bash
supervisorctl status groot_n1d7_spatial_seed43
tail -f /var/log/portal/groot_n1d7_spatial_seed43.log
supervisorctl stop groot_n1d7_spatial_seed43
```

Run status and metrics are also written beneath:

```text
outputs/n1d7-spatial-seed43/trajectories-NNN/
```

## Validation performed before launch

A real unfrozen optimizer update was run on the 10-trajectory dataset. Vision,
language, and action-head gradients were finite and nonzero, parameters in all
three groups changed, and peak GPU allocation was approximately 58.6 GiB.

The separate native `launch_finetune.py` entrypoint currently has a missing
`MODEL_REGISTRY` export in `gr00t/model/__init__.py`. This campaign uses the
finite-epoch adapter and is not affected by that native-launcher defect.
