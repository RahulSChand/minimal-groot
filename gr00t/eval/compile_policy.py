"""Opt-in compilation of the native GR00T policy's diffusion transformer.

Keep this module free of eager torch imports: the CLI validates model types
before setting CUDA_VISIBLE_DEVICES and initializing CUDA.
"""

COMPILE_MODEL_TYPES = ("gr00t_n1_5", "Gr00tN1d6", "Gr00tN1d7")


def validate_compile_model_type(model_type):
    if model_type not in COMPILE_MODEL_TYPES:
        raise ValueError(f"--compile supports GR00T N1.5, N1.6 and N1.7; got {model_type!r}")


def compile_action_transformer(model):
    """Compile DiT/AlternateVLDiT, preserving the native masks and denoising loop."""
    import torch

    validate_compile_model_type(model.config.model_type)
    if model.training or next(model.parameters()).device.type != "cuda":
        raise ValueError("--compile requires an eval-mode model on CUDA")
    # All three generations call this block through get_action, bypassing the
    # top-level model.forward. N1.6/N1.7 also pass their image/text attention masks;
    # compile the actual bound forward rather than reimplementing those paths.
    metadata = {"target": "action_head.model.forward", "backend": "inductor", "mode": "reduce-overhead"}
    compile_kwargs = {"mode": "reduce-overhead"}
    if model.config.model_type == "Gr00tN1d7":
        # N1.7 exceeded our numerical tolerance with Inductor fusion, including
        # emulate_precision_casts. Capture/replay native ATen operators instead:
        # reduce launch overhead without Inductor's fused-kernel rewrites.
        compile_kwargs = {"backend": "cudagraphs"}
        metadata.update(backend="cudagraphs", mode="default")
    model.action_head.model.forward = torch.compile(
        model.action_head.model.forward, dynamic=False, fullgraph=True, **compile_kwargs
    )
    return metadata
