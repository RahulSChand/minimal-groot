"""Deterministic trajectory selection for sample-efficiency experiments."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

MANIFEST_VERSION = 1


def _json_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def create_or_validate_manifest(dataset_root: Path | str, manifest_path: Path | str, seed: int) -> dict:
    """Create one stable episode ordering, or validate the existing manifest.

    Taking prefixes of the ordering makes trajectory budgets nested: a 5-trajectory
    run sees the same first five episodes as a 10-trajectory run.
    """
    dataset_root = Path(dataset_root).expanduser().resolve()
    manifest_path = Path(manifest_path).expanduser().resolve()
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    tasks_path = dataset_root / "meta" / "tasks.jsonl"
    info_path = dataset_root / "meta" / "info.json"
    for required in (episodes_path, tasks_path, info_path):
        if not required.is_file():
            raise FileNotFoundError(f"Dataset metadata is missing: {required}")

    episodes = sorted(_json_lines(episodes_path), key=lambda item: int(item["episode_index"]))
    task_by_prompt = {item["task"]: int(item["task_index"]) for item in _json_lines(tasks_path)}
    episode_records = []
    for episode in episodes:
        prompts = episode.get("tasks", [])
        if len(prompts) != 1 or prompts[0] not in task_by_prompt:
            raise ValueError(f"Episode {episode.get('episode_index')} does not map to exactly one known task")
        episode_records.append(
            {
                "episode_index": int(episode["episode_index"]),
                "task_index": task_by_prompt[prompts[0]],
                "length": int(episode["length"]),
            }
        )

    metadata_digest = hashlib.sha256(episodes_path.read_bytes() + tasks_path.read_bytes()).hexdigest()
    ordered_indices = [item["episode_index"] for item in episode_records]
    random.Random(seed).shuffle(ordered_indices)
    expected = {
        "version": MANIFEST_VERSION,
        "seed": seed,
        "selection": "global_without_replacement_python_random_v1",
        "dataset_metadata_sha256": metadata_digest,
        "total_episodes": len(episode_records),
        "ordered_episode_indices": ordered_indices,
        "episodes": episode_records,
    }

    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing != expected:
            raise ValueError(
                f"Existing trajectory manifest does not match seed={seed} and dataset metadata: {manifest_path}"
            )
        return existing

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
    temporary.replace(manifest_path)
    return expected


def selected_episode_indices(manifest: dict, trajectory_count: int) -> list[int]:
    """Return the selected prefix after validating the requested budget."""
    if trajectory_count < 1:
        raise ValueError(f"trajectory_count must be positive, got {trajectory_count}")
    ordered = [int(value) for value in manifest["ordered_episode_indices"]]
    if trajectory_count > len(ordered):
        raise ValueError(f"Requested {trajectory_count} trajectories, but the dataset contains only {len(ordered)}")
    return ordered[:trajectory_count]
