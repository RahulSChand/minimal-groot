# Extraction provenance

The GR00T Python sources, DeepSpeed configuration files, fine-tuning shell
launcher, and SO100 modality configuration in this directory were copied from
Isaac-GR00T commit:

```text
51d4c89f72fda44cbf77285c6a8114b52676b8a1
```

Extraction date: 2026-09-12.

The original N1 model, SmolLM2/SigLIP Eagle backbone, tokenizer assets, and
native image/prompt processing were extracted from NVIDIA Isaac-GR00T
`n1-release` commit `755876a9afdb41ca6eb6383b36f4a0adb085c73f` on 2026-09-15.
They reside under `gr00t/model/gr00t_n1`, with local package paths, shared
state/action processing, strict checkpoint loading, and Transformers 4.57
configuration compatibility. N1 retains its original action head and 16-step
diffusion sampler.
The published N1 base checkpoint also contains an unused two-class auxiliary
`action_head.decode_layer` (weight and bias). These two obsolete tensors are
ignored, matching the released architecture; all policy weights are checked.

The N1.5 model and Eagle assets were extracted from NVIDIA Isaac-GR00T
`n1d5` commit `4af2b622892f7dcb5aae5a3fb70bcb02dc217b96`; the N1.6 model,
processor, modules and Eagle assets came from `n1d6` commit
`9b37aa1ce69c73c6d165233fa88128283bba4508`. Their source headers and licenses
are retained. Model code is versioned under `gr00t/model/gr00t_n1d5` and
`gr00t/model/gr00t_n1d6` to preserve checkpoint architectures.

Local adaptations include version dispatch, native checkpoint loading,
Transformers 4.57 compatibility, N1.5 prompting and processor integration,
Parquet image decoding, configurable optimizer options, and a finite
trajectory-subset trainer. The adapters were recovered from the local
`minimal_groot-0.1.0` build artifact and checked against the pinned sources.

The sample-efficiency protocol matches `post_train_vla` commit
`a48f359cd0c1d789e407bc4d0082ef74ccb07c23`, specifically
`scripts/run_sample_efficiency.sh` and `src/post_train_vla/sample_efficiency.py`.
The comparison runner imports its policy-agnostic WebSocket server/evaluator
from the configured reference-project path; it does not import its pi0 models.
