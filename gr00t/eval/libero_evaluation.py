"""Training-time entry point for the canonical native LIBERO evaluator."""

import os
import random
import warnings
from pathlib import Path

import numpy as np
import torch

from gr00t.eval.evaluate_checkpoint import evaluate_loaded_policy, parse_args, prepare_output
from gr00t.policy.gr00t_policy import Gr00tPolicy


def evaluate_live_model(
    model,
    processor,
    *,
    eval_python,
    output_dir,
    reference_project=None,
    port=8765,
    workers=20,
    max_batch_size=8,
    episodes=20,
    seed=7,
    timeout=1800,
    suite="libero_spatial",
    task_ids=(0,),
    gpu=None,
):
    """Same native inference/environment as checkpoints; restore training state/RNG."""
    if reference_project is not None:
        warnings.warn(
            "reference_project is obsolete: evaluation is self-contained in minimal-groot",
            DeprecationWarning,
            stacklevel=2,
        )
    if gpu is None:
        gpu = torch.cuda.current_device()
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible:
            gpu = int(visible.split(",")[gpu])
    argv = [
        "--checkpoint",
        str(Path.cwd()),
        "--suite",
        suite,
        "--output-dir",
        str(output_dir),
        "--eval-python",
        str(eval_python),
        "--gpu",
        str(gpu),
        "--port",
        str(port),
        "--workers",
        str(workers),
        "--max-batch-size",
        str(max_batch_size),
        "--episodes-per-task",
        str(episodes),
        "--seed",
        str(seed),
        "--timeout",
        str(timeout),
    ]
    for task in task_ids or ():
        argv.extend(("--task-id", str(task)))
    args = parse_args(argv)
    manifest = prepare_output(args)
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    training = model.training
    processor_training = getattr(processor, "training", training)
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            model.eval()
            processor.eval()
            native = Gr00tPolicy.from_model(model, processor)
            return evaluate_loaded_policy(
                native, args, manifest, {**vars(args), "protocol": "native_libero_v1", "live_model": True}
            )
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        model.train(training)
        processor.train() if processor_training else processor.eval()
