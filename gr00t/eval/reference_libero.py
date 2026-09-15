"""Evaluate the live GR00T model with post_train_vla's unchanged LIBERO evaluator."""

import asyncio
import json
import os
import random
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

import numpy as np
import torch

from gr00t.data.types import EmbodimentTag, MessageType, VLAStepData


class ReferenceLiberoPolicy:
    """Adapt OpenPI wire observations to GR00T without loading another model."""

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.tag = EmbodimentTag.LIBERO_PANDA
        self.modalities = processor.get_modality_configs()[self.tag.value]
        self.metadata = {
            "model_type": model.config.model_type,
            "action_horizon": model.config.action_horizon,
        }

    def infer(self, observation):
        return self.infer_batch([observation])[0]

    @torch.inference_mode()
    def infer_batch(self, observations):
        started = time.monotonic()
        features, states = [], []
        keys = self.modalities["state"].modality_keys
        for observation in observations:
            state_array = np.asarray(observation["observation/state"], dtype=np.float32)
            if state_array.shape != (8,):
                raise ValueError(f"Expected 8D LIBERO state, got {state_array.shape}")
            state = {key: state_array[None, i : i + 1] for i, key in enumerate(keys[:-1])}
            state["gripper"] = state_array[None, 6:8]
            states.append(state)
            step = VLAStepData(
                images={key: [observation[f"observation/{key}"]] for key in ("image", "wrist_image")},
                states=state,
                actions={},
                text=observation["prompt"],
                embodiment=self.tag,
            )
            features.append(self.processor([{"type": MessageType.EPISODE_STEP.value, "content": step}]))
        batch = self.processor.collator(features)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = self.model.get_action(**batch)["action_pred"]
        decoded = self.processor.decode_action(
            prediction.float().cpu().numpy(),
            self.tag,
            {key: np.stack([state[key] for state in states]) for key in keys},
        )
        actions = np.concatenate([decoded[key] for key in self.modalities["action"].modality_keys], axis=-1)
        if actions.shape[-1] != 7 or not np.isfinite(actions).all():
            raise ValueError("GR00T produced invalid LIBERO actions")
        # The parquet actions already use LIBERO's controller convention. No
        # gripper inversion or extra state-relative subtraction is appropriate.
        timing = {"infer_ms": (time.monotonic() - started) * 1000, "batch_size": len(observations)}
        return [{"actions": action.astype(np.float32), "policy_timing": timing} for action in actions]


async def _evaluate(
    model,
    processor,
    *,
    reference_project,
    eval_python,
    output_dir,
    port,
    workers,
    max_batch_size,
    episodes,
    seed,
    timeout,
):
    from post_train_vla.policy_server import PolicyServer

    server = PolicyServer(
        ReferenceLiberoPolicy(model, processor),
        host="127.0.0.1",
        port=port,
        max_batch_size=max_batch_size,
        batch_wait_ms=10,
    )
    server_task = asyncio.create_task(server.run())
    env = os.environ.copy()
    env.update(
        MUJOCO_GL="egl",
        PYOPENGL_PLATFORM="egl",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        TOKENIZERS_PARALLELISM="false",
        PYTHONUNBUFFERED="1",
        # The pinned LIBERO initial states are NumPy pickles. Restore the
        # serializer default used by its original simulator environment.
        TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1",
    )
    env["PYTHONPATH"] = str(reference_project / "src")
    command = [
        str(eval_python),
        "-m",
        "post_train_vla.eval_libero",
        "--policy-url",
        f"ws://127.0.0.1:{port}",
        "--suite",
        "libero_spatial",
        "--task-id",
        "0",
        "--episodes-per-task",
        str(episodes),
        "--episode-offset",
        "0",
        "--seed",
        str(seed),
        "--workers",
        str(workers),
        "--replan-steps",
        "5",
        "--wait-steps",
        "10",
        "--render-resolution",
        "256",
        "--resize-size",
        "256",
        "--output-dir",
        str(output_dir),
    ]
    # GR00T performs its own checkpoint-specific crop/resize from the 256px
    # source cameras; every other rollout option matches the reference.
    output_dir.mkdir(parents=True, exist_ok=False)
    process = None
    try:
        await asyncio.sleep(0)
        if server_task.done():
            await server_task
        with (output_dir / "rollout.log").open("w") as log:
            process = await asyncio.create_subprocess_exec(
                *command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True
            )
            wait_task = asyncio.create_task(process.wait())
            done, _ = await asyncio.wait({wait_task, server_task}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
            if server_task in done:
                await server_task
                raise RuntimeError("Policy server stopped during evaluation")
            if wait_task not in done:
                raise TimeoutError(f"LIBERO evaluation exceeded {timeout} seconds")
            if process.returncode:
                raise RuntimeError(f"LIBERO evaluator exited {process.returncode}; see {output_dir / 'rollout.log'}")
        summary = json.loads((output_dir / "summary.json").read_text())
        records = [json.loads(line) for line in (output_dir / "episodes.jsonl").read_text().splitlines()]
        if len(records) != episodes or summary["episodes"] != episodes or any(r.get("error") for r in records):
            raise RuntimeError(f"LIBERO evaluation incomplete or contains errors: {output_dir}")
        return summary
    finally:
        if process is not None and process.returncode is None:
            import signal

            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(process.wait(), 10)
            if process.returncode is None:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                await process.wait()
        server_task.cancel()
        with suppress(asyncio.CancelledError):
            await server_task


def evaluate_live_model(
    model,
    processor,
    *,
    reference_project: Path,
    eval_python: Path,
    output_dir: Path,
    port=8765,
    workers=20,
    max_batch_size=8,
    episodes=20,
    seed=7,
    timeout=1800,
):
    """Keep evaluation RNG draws separate from training and restore train mode."""
    source = str(reference_project / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    training = model.training
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            model.eval()
            processor.eval()
            return asyncio.run(
                _evaluate(
                    model,
                    processor,
                    reference_project=reference_project,
                    eval_python=eval_python,
                    output_dir=output_dir,
                    port=port,
                    workers=workers,
                    max_batch_size=max_batch_size,
                    episodes=episodes,
                    seed=seed,
                    timeout=timeout,
                )
            )
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        model.train(training)
        processor.train() if training else processor.eval()
