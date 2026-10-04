"""Canonical checkpoint evaluation using native Gr00tPolicy and LiberoEnv."""

import argparse
import json
import logging
import os
import signal
import subprocess
import time
from contextlib import nullcontext, suppress
from pathlib import Path

from gr00t.eval.compile_policy import validate_compile_model_type

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]


def handle_termination(signum, frame):
    raise KeyboardInterrupt(f"Received signal {signum}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--suite", choices=("libero_spatial", "libero_goal", "libero_object", "libero_10"), required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-python", type=Path, default=ROOT / "gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python")
    parser.add_argument("--gpu", type=int, default=0, help="Physical CUDA/EGL GPU index")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--episodes-per-task", type=int, default=40)
    parser.add_argument("--task-id", type=int, action="append", dest="task_ids")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timeout", type=float, default=14400, help="Rollout timeout, excluding model load/warmup")
    parser.add_argument(
        "--compile", action="store_true", dest="compile_model", help="Compile native N1.5/N1.6/N1.7 action transformer"
    )
    args = parser.parse_args(argv)
    if min(args.workers, args.max_batch_size, args.episodes_per_task, args.timeout) <= 0:
        parser.error("Worker, batch, episode counts and timeout must be positive")
    if args.gpu < 0 or not 0 < args.port < 65536:
        parser.error("GPU must be nonnegative and port in 1..65535")
    if args.task_ids is not None and (len(set(args.task_ids)) != len(args.task_ids) or min(args.task_ids) < 0):
        parser.error("Task IDs must be unique and nonnegative")
    for name in ("checkpoint", "output_dir", "eval_python"):
        setattr(args, name, getattr(args, name).absolute())
    return args


def simulator_environment(args):
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.update(
        MUJOCO_GL="egl",
        PYOPENGL_PLATFORM="egl",
        MUJOCO_EGL_DEVICE_ID=str(args.gpu),
        PYTHONPATH=str(ROOT),
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        TOKENIZERS_PARALLELISM="false",
        PYTHONUNBUFFERED="1",
        TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1",
    )
    return env


def simulator_command(args):
    command = [
        str(args.eval_python),
        "-m",
        "gr00t.eval.libero_simulator",
        "--suite",
        args.suite,
        "--episodes-per-task",
        str(args.episodes_per_task),
        "--output-dir",
        str(args.output_dir),
        "--workers",
        str(args.workers),
        "--max-batch-size",
        str(args.max_batch_size),
        "--seed",
        str(args.seed),
        "--port",
        str(args.port),
    ]
    for task in args.task_ids or []:
        command.extend(("--task-id", str(task)))
    return command


def validate_results(output_dir, manifest):
    rows = [json.loads(line) for line in (output_dir / "episodes.jsonl").read_text().splitlines() if line]
    summary = json.loads((output_dir / "summary.json").read_text())
    expected = {(task["task_id"], i) for task in manifest["tasks"] for i in range(manifest["episodes_per_task"])}
    actual = [(r["task_id"], r["episode_index"]) for r in rows]
    if len(actual) != len(expected) or set(actual) != expected or any(r.get("error") for r in rows):
        raise RuntimeError("Evaluation has missing/duplicate/unexpected episodes or errors")
    successes = sum(r["success"] for r in rows)
    if (
        summary["episodes"] != len(rows)
        or summary["successes"] != successes
        or abs(summary["success_rate"] - successes / len(rows)) > 1e-12
    ):
        raise RuntimeError("Summary disagrees with episode records")
    if summary["suite"] != manifest["suite"] or any(r["suite"] != manifest["suite"] for r in rows):
        raise RuntimeError("Evaluation suite disagrees with requested suite")
    summary["per_task"] = []
    for task in manifest["tasks"]:
        records = [r for r in rows if r["task_id"] == task["task_id"]]
        count = sum(r["success"] for r in records)
        summary["per_task"].append(
            dict(
                task_id=task["task_id"],
                task=task["prompt"],
                episodes=len(records),
                successes=count,
                success_rate=count / len(records),
            )
        )
    summary["status"] = "complete"
    return summary


def warmup_policy(policy, manifest, max_batch_size):
    import random

    import numpy as np
    import torch

    from gr00t.eval.libero_simulator import batch_observations

    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    timings = []
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            for task in manifest["tasks"]:
                observation = {
                    f"video.{key}": np.zeros((256, 256, 3), dtype=np.uint8) for key in ("image", "wrist_image")
                }
                observation.update(
                    {
                        f"state.{key}": np.zeros(2 if key == "gripper" else 1, dtype=np.float32)
                        for key in ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
                    }
                )
                observation["annotation.human.action.task_description"] = task["prompt"]
                for batch in range(1, max_batch_size + 1):
                    started = time.monotonic()
                    for _ in range(3):
                        policy.get_action(batch_observations([observation] * batch))
                    torch.cuda.synchronize()
                    seconds = time.monotonic() - started
                    timings.append(dict(task_id=task["task_id"], batch=batch, seconds=seconds))
                    LOGGER.info("Warmup task=%s batch=%s: %.2fs", task["task_id"], batch, seconds)
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
    return timings


def run_rollouts(policy, args):
    from gr00t.policy.server_client import PolicyServer

    process = None
    try:
        with PolicyServer(policy, host="127.0.0.1", port=args.port) as server:
            with (args.output_dir / "rollout.log").open("w") as log:
                process = subprocess.Popen(
                    simulator_command(args),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=simulator_environment(args),
                    start_new_session=True,
                )
                deadline = time.monotonic() + args.timeout
                while process.poll() is None:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Rollouts exceeded {args.timeout}s; see rollout.log")
                    server.serve_once(timeout_ms=100)
                if process.returncode:
                    raise RuntimeError(f"Simulator exited {process.returncode}; see rollout.log")
    finally:
        if process is not None and process.poll() is None:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def prepare_output(args):
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "prepare.log").open("w") as log:
        subprocess.run(
            [*simulator_command(args), "--describe"],
            env=simulator_environment(args),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=120,
        )
    return json.loads((args.output_dir / "suite.json").read_text())


def evaluate_loaded_policy(native, args, manifest, run):
    import torch

    from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper
    from gr00t.utils.determinism import seed_everything

    policy = Gr00tSimPolicyWrapper(native)
    batch_size = min(args.max_batch_size, args.workers, args.episodes_per_task)
    run.update(
        effective_max_batch_size=batch_size,
        torch=torch.__version__,
        policy_metadata=native.metadata,
        model_dtype=str(next(native.model.parameters()).dtype),
    )
    cache_limit = max(8, len(manifest["tasks"]) * batch_size + 8)
    context = (
        torch._dynamo.config.patch(
            cache_size_limit=cache_limit, accumulated_cache_size_limit=max(256, cache_limit), suppress_errors=False
        )
        if args.compile_model
        else nullcontext()
    )

    def save_run():
        (args.output_dir / "run.json").write_text(json.dumps(run, indent=2, default=str) + "\n")

    try:
        with context:
            run["status"] = "warming_up"
            save_run()
            started = time.monotonic()
            warmup = warmup_policy(policy, manifest, batch_size) if args.compile_model else []
            run["warmup_seconds"] = time.monotonic() - started
            (args.output_dir / "warmup.json").write_text(json.dumps(warmup, indent=2) + "\n")
            if args.compile_model:
                from torch._dynamo.utils import counters

                run["compile_graphs_after_warmup"] = counters["stats"]["unique_graphs"]
            seed_everything(args.seed)
            run["status"] = "running"
            save_run()
            started = time.monotonic()
            run_rollouts(policy, args)
            run["rollout_seconds"] = time.monotonic() - started
            if args.compile_model:
                run["compile_graphs_after_rollouts"] = counters["stats"]["unique_graphs"]
        summary = validate_results(args.output_dir, manifest)
        summary.update(
            compile=args.compile_model,
            warmup_seconds=run["warmup_seconds"],
            rollout_seconds=run["rollout_seconds"],
            gpu=args.gpu,
            checkpoint=str(args.checkpoint),
            policy_metadata=native.metadata,
        )
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        run["status"] = "complete"
        LOGGER.info("COMPLETE: %s/%s successes", summary["successes"], summary["episodes"])
        return summary
    except BaseException as exc:
        run.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save_run()


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not (args.checkpoint / "config.json").is_file() or not args.eval_python.is_file():
        raise FileNotFoundError("Checkpoint config.json and pinned simulator Python must exist")
    if args.compile_model:
        validate_compile_model_type(json.loads((args.checkpoint / "config.json").read_text())["model_type"])
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "4")
    previous = signal.signal(signal.SIGTERM, handle_termination)
    run = {**vars(args), "status": "preparing", "protocol": "native_libero_v1"}
    try:
        manifest = prepare_output(args)
        import torch

        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.policy.gr00t_policy import Gr00tPolicy
        from gr00t.utils.determinism import seed_everything

        torch.set_num_threads(1)
        seed_everything(args.seed)
        native = Gr00tPolicy(
            EmbodimentTag.LIBERO_PANDA, str(args.checkpoint), device="cuda", compile_model=args.compile_model
        )
        (args.output_dir / "strict_load.json").write_text(json.dumps(native.loading_info, indent=2) + "\n")
        evaluate_loaded_policy(native, args, manifest, run)
    except BaseException as exc:
        run.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        if args.output_dir.is_dir():
            (args.output_dir / "run.json").write_text(json.dumps(run, indent=2, default=str) + "\n")
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
