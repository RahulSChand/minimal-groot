"""Canonical LIBERO suite runner: native LiberoEnv + native PolicyClient.

Run in the pinned simulator environment. No model imports or post_train_vla.
Each environment has a dedicated process; the coordinator sends explicit batches.
"""

import argparse
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
ACTION_KEYS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
_env = _task = _suite = _lock = None


def describe_suite(suite_name, task_ids, episodes):
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[suite_name]()
    ids = list(range(suite.n_tasks)) if task_ids is None else task_ids
    if not ids or len(set(ids)) != len(ids) or any(i < 0 or i >= suite.n_tasks for i in ids):
        raise ValueError("Invalid or duplicate task IDs")
    tasks = []
    for i in ids:
        if not 0 < episodes <= len(suite.get_task_init_states(i)):
            raise ValueError(f"Invalid initial-state count for task {i}")
        tasks.append({"task_id": i, "prompt": str(suite.get_task(i).language)})
    return {"suite": suite_name, "tasks": tasks, "episodes_per_task": episodes}


def init_worker(suite_name, lock):
    global _suite, _lock
    from libero.libero import benchmark

    _suite = benchmark.get_benchmark_dict()[suite_name]()
    _lock = lock


def reset_worker(task_id, episode, seed):
    global _env, _task
    from libero.libero import get_libero_path
    from OpenGL import GL

    from gr00t.eval.sim.LIBERO.libero_env import LiberoEnv

    with _lock:
        if _task != task_id:
            if _env is not None:
                _env.close()
            task = _suite.get_task(task_id)
            path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            _env = LiberoEnv(str(path), str(task.language))
            _task = task_id
        observation, info = _env.reset(
            seed=seed,
            options={
                "initial_state": _suite.get_task_init_states(task_id)[episode],
                "wait_steps": 10,
            },
        )
        if GL.glGetString(GL.GL_VENDOR) != b"NVIDIA Corporation":
            raise RuntimeError("Canonical GPU evaluation requires NVIDIA EGL rendering")
    return observation, bool(info["success"])


def step_worker(actions):
    success = False
    for index in range(len(actions["action.x"])):
        observation, _, done, _, info = _env.step({k: v[index] for k, v in actions.items()})
        success = bool(done or info.get("success"))
        if success:
            break
    return observation, success, index + 1


def batch_observations(observations):
    result = {}
    for key in observations[0]:
        values = [o[key] for o in observations]
        if key.startswith("video."):
            result[key] = np.stack(values).astype(np.uint8)[:, None]
        elif key.startswith("state."):
            result[key] = np.stack(values).astype(np.float32)[:, None]
        else:
            result[key] = [str(v) for v in values]
    return result


def evaluate(args, manifest):
    from gr00t.policy.server_client import PolicyClient

    context = mp.get_context("spawn")
    lock = context.Lock()
    workers = min(args.workers, args.max_batch_size, args.episodes_per_task)
    pools = [
        ProcessPoolExecutor(max_workers=1, mp_context=context, initializer=init_worker, initargs=(args.suite, lock))
        for _ in range(workers)
    ]
    rows = []
    client = PolicyClient(host="127.0.0.1", port=args.port, timeout_ms=120000)
    try:
        with (args.output_dir / "episodes.jsonl").open("x") as output:
            for task in manifest["tasks"]:
                for offset in range(0, args.episodes_per_task, workers):
                    episodes = list(range(offset, min(offset + workers, args.episodes_per_task)))
                    started = time.monotonic()
                    futures = [
                        pools[i].submit(reset_worker, task["task_id"], ep, args.seed) for i, ep in enumerate(episodes)
                    ]
                    active = {}
                    for i, future in enumerate(futures):
                        observation, success = future.result()
                        active[i] = dict(
                            observation=observation,
                            success=success,
                            steps=0,
                            calls=0,
                            server_ms=[],
                            client_ms=[],
                            batch_sizes=[],
                        )
                    while active:
                        finished = [
                            i
                            for i, state in active.items()
                            if state["success"] or state["steps"] >= MAX_STEPS[args.suite]
                        ]
                        for i in finished:
                            state = active.pop(i)
                            row = dict(
                                suite=args.suite,
                                task_id=task["task_id"],
                                task=task["prompt"],
                                episode_index=episodes[i],
                                seed=args.seed,
                                success=state["success"],
                                error=None,
                                policy_steps=state["steps"],
                                inference_calls=state["calls"],
                                elapsed_seconds=time.monotonic() - started,
                                mean_server_inference_ms=float(np.mean(state["server_ms"])) if state["calls"] else 0,
                                mean_inference_ms=float(np.mean(state["client_ms"])) if state["calls"] else 0,
                                mean_inference_batch_size=float(np.mean(state["batch_sizes"])) if state["calls"] else 0,
                            )
                            rows.append(row)
                            output.write(json.dumps(row) + "\n")
                            output.flush()
                            print(
                                f"task={row['task_id']} episode={row['episode_index']} success={row['success']}",
                                flush=True,
                            )
                        if not active:
                            break
                        ids = list(active)
                        before = time.monotonic()
                        actions, info = client.get_action(batch_observations([active[i]["observation"] for i in ids]))
                        elapsed = (time.monotonic() - before) * 1000
                        futures = {}
                        for batch_index, i in enumerate(ids):
                            state = active[i]
                            steps = min(5, MAX_STEPS[args.suite] - state["steps"])
                            chunk = {
                                f"action.{k}": np.asarray(actions[f"action.{k}"][batch_index, :steps])
                                for k in ACTION_KEYS
                            }
                            if any(v.shape != (steps, 1) or not np.isfinite(v).all() for v in chunk.values()):
                                raise ValueError("Invalid native policy action chunk")
                            futures[i] = pools[i].submit(step_worker, chunk)
                            state["calls"] += 1
                            state["server_ms"].append(info.get("infer_ms", 0))
                            state["client_ms"].append(elapsed)
                            state["batch_sizes"].append(len(ids))
                        for i, future in futures.items():
                            observation, success, steps = future.result()
                            active[i].update(observation=observation, success=success)
                            active[i]["steps"] += steps
        successes = sum(r["success"] for r in rows)
        summary = dict(
            suite=args.suite,
            episodes=len(rows),
            successes=successes,
            success_rate=successes / len(rows),
            workers=workers,
            protocol="native_libero_v1",
            save_video=False,
            replan_steps=5,
            wait_steps=10,
            render_resolution=256,
            max_steps=MAX_STEPS[args.suite],
        )
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    finally:
        client.close()
        for pool in pools:
            pool.shutdown(wait=True, cancel_futures=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, choices=MAX_STEPS)
    parser.add_argument("--task-id", type=int, action="append", dest="task_ids")
    parser.add_argument("--episodes-per-task", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if min(args.workers, args.max_batch_size, args.episodes_per_task) <= 0:
        parser.error("Worker, batch and episode counts must be positive")
    manifest = describe_suite(args.suite, args.task_ids, args.episodes_per_task)
    if args.describe:
        (args.output_dir / "suite.json").write_text(json.dumps(manifest, indent=2) + "\n")
    else:
        evaluate(args, manifest)


if __name__ == "__main__":
    main()
