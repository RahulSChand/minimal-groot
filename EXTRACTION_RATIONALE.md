# Why this subset exists

## Goal

`minimal-groot` is a focused fine-tuning and LIBERO-evaluation extraction of
Isaac GR00T N1.7. Its goal is to preserve the existing, tested training and
evaluation behavior without carrying every simulator, deployment target, and
real-robot integration from the upstream repository.

This is deliberately an extraction rather than a rewrite. GR00T fine-tuning
depends on details that are easy to miss when recreating the pipeline: modality
registration, temporal indexing, dataset statistics, state and action
normalization, processor serialization, Hugging Face model registration,
distributed synchronization, and standalone checkpoint formatting. Keeping
the relevant upstream implementation intact retains those behaviors and their
edge-case handling.

## How the boundary was chosen

The training extraction starts at `gr00t/experiment/launch_finetune.py`. The
evaluation extension starts at `gr00t/eval/run_gr00t_server.py` and
`gr00t/eval/rollout_policy.py`, then follows their policy, transport, temporal
horizon, wrapper, and LIBERO-adapter dependencies. Simulator packages remain
outside the project environment and are installed into a benchmark-specific
virtual environment.

The copied implementation is organized into seven responsibilities:

- **Configuration** defines the N1.7 model, datasets, training arguments,
  embodiment modalities, and DeepSpeed settings. These files also validate
  incompatible precision, batching, modality, and resume configurations.
- **Data loading and processing** reads GR00T-compatible LeRobot datasets,
  selects temporally indexed observations and actions, decodes video, computes
  or merges statistics, and converts physical states and actions into the
  representations expected by the model.
- **Model code** contains the N1.7 action head, Qwen3-VL backbone adapter,
  diffusion transformer modules, image augmentation, processor, and Hugging
  Face registration. Model and processor registration are import-time side
  effects required when loading existing checkpoints.
- **Training orchestration** assembles the model, processor, dataset, collator,
  Hugging Face trainer, distributed runtime, and checkpoint callbacks. It
  preserves single-GPU, DDP, and DeepSpeed paths.
- **Small utilities** provide distributed rank coordination, safe serialization
  of initial actions, and TorchCodec-backed video access used by the dataset
  loader.
- **Policy inference** loads a checkpoint and processor, validates model-facing
  observations, and decodes normalized predictions into physical actions.
- **Evaluation and simulation** provide ZMQ policy serving, open-loop metrics,
  temporal observation/action horizon handling, closed-loop rollout, video
  recording, and a Gymnasium adapter for LIBERO.

The shell launcher and one custom-embodiment modality example are included so
the extracted pipeline has a concrete invocation and configuration template.
License and attribution files remain with the copied source.

## What is intentionally excluded

The following concerns are outside the current goal and were therefore not
included:

- SimplerEnv, RoboCasa, RoboCasa365, and GR1 simulator integrations
- Real-robot evaluation
- Vendored simulator repositories and assets; LIBERO is fetched on demand at a
  pinned commit into an ignored external-dependency directory
- ONNX export, TensorRT engines, and inference benchmarks
- Jetson, Spark, and dGPU deployment containers and installation scripts
- Dataset conversion, repair, download, and platform activation utilities
- Benchmark-specific examples, media, notebooks, and documentation tests
- The general pretraining launcher

These areas do not participate in producing a fine-tuned checkpoint. Excluding
them avoids large external repositories, mutually incompatible simulator
environments, hardware-specific packages, and serving dependencies.

## Why some apparently optional pieces remain

Some code is retained even when it is not exercised by every run:

- DeepSpeed configuration is needed by the upstream default multi-GPU path.
- TorchCodec is needed when training data stores observations as video.
- Distributed helpers protect rank-zero file generation and propagate failures
  so multi-GPU workers do not deadlock.
- Checkpoint callbacks copy processor and normalization artifacts into periodic
  checkpoints, making those checkpoints self-contained.
- The generic model registry and configuration structure are preserved because
  the training entry point uses their registration side effects.
- W&B remains a base dependency because the unchanged upstream training module
  imports it at module load time, even when logging is disabled.

FlashAttention and DeepSpeed are exposed as optional dependency groups because
they improve or enable particular training configurations but are not required
to import the package or use the single-GPU SDPA path. Gymnasium, plotting, and
ZMQ serialization packages are similarly grouped under the `eval` extra.

## Compatibility contract

The extraction preserves the `gr00t` Python namespace and the original class
and module locations. This is important for Hugging Face registration and
checkpoint compatibility.

A usable periodic checkpoint must continue to contain model weights and config
along with:

- `processor_config.json`
- `statistics.json`
- `embodiment_id.json`
- `experiment_cfg/`

Removing or reconstructing those artifacts would risk changing normalization,
modality interpretation, or embodiment routing during later inference.

## Validation performed

The training extraction was validated with:

- All copied implementation and example files matched the upstream files
  byte-for-byte.
- The fine-tuning CLI imported and rendered its help output.
- The dependency lockfile resolved successfully.
- A source distribution and wheel built successfully, including the DeepSpeed
  JSON resources.
- 183 tests passed (with one optional test skipped), covering configuration
  safety, batch-size and resume invariants, dataset construction, policy
  transport, embodiment mappings, action-horizon validation, rollout wrappers,
  gated-backbone errors, initial-action serialization, and the extraction
  boundary.
- A three-step GPU fine-tuning smoke run and a parity comparison with the full
  Isaac-GR00T checkout.

The evaluation extension retains the upstream Python implementation
byte-for-byte and adds its focused transport, horizon, wrapper, and rollout
tests. The LIBERO setup script has two extraction-owned safety changes: it
clones only the pinned LIBERO repository rather than relying on the full
upstream submodule tree, and it does not delete an existing `~/.libero`
configuration. A model-in-the-loop rollout with the LIBERO Spatial checkpoint
completed successfully, and the same task and seed also succeeded through the
full Isaac-GR00T source tree.

## Adding capabilities later

Additional simulator support should be added as a separate adapter and
dependency island. Each addition should begin from its upstream entry point,
follow the same dependency-closure process, and add parity tests before any
cleanup or refactoring. The shared policy and rollout layers should not be
reimplemented per simulator.
