"""Deterministic whole-trajectory subsets with finite, frame-exact epochs."""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.types import EmbodimentTag, MessageType


def create_or_validate_manifest(dataset_root: Path, manifest_path: Path, seed: int) -> dict:
    """Match post_train_vla.sample_efficiency's version-1 ordering exactly."""
    episodes_path = dataset_root / "meta/episodes.jsonl"
    tasks_path = dataset_root / "meta/tasks.jsonl"
    episodes = sorted(
        [json.loads(line) for line in episodes_path.read_text().splitlines() if line.strip()],
        key=lambda item: int(item["episode_index"]),
    )
    tasks = [json.loads(line) for line in tasks_path.read_text().splitlines() if line.strip()]
    task_by_prompt = {item["task"]: int(item["task_index"]) for item in tasks}
    records = []
    for episode in episodes:
        prompts = episode.get("tasks", [])
        if len(prompts) != 1 or prompts[0] not in task_by_prompt:
            raise ValueError(f"Episode {episode['episode_index']} must have one known task")
        records.append(
            dict(
                episode_index=int(episode["episode_index"]),
                task_index=task_by_prompt[prompts[0]],
                length=int(episode["length"]),
            )
        )
    indices = [item["episode_index"] for item in records]
    if len(set(indices)) != len(indices) or any(item["length"] < 1 for item in records):
        raise ValueError("Episode IDs must be unique and lengths positive")
    random.Random(seed).shuffle(indices)
    manifest = {
        "version": 1,
        "seed": seed,
        "selection": "global_without_replacement_python_random_v1",
        "dataset_metadata_sha256": hashlib.sha256(episodes_path.read_bytes() + tasks_path.read_bytes()).hexdigest(),
        "total_episodes": len(records),
        "ordered_episode_indices": indices,
        "episodes": records,
    }
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError(f"Manifest does not match dataset and seed: {manifest_path}")
    else:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        temporary.replace(manifest_path)
    return manifest


class TrajectorySubsetDataset(Dataset):
    """Preload selected episodes only; expose every frame once and pad within episodes."""

    def __init__(self, dataset_root, episode_indices, modality_configs, processor=None):
        selected = list(episode_indices)
        if not selected or len(selected) != len(set(selected)):
            raise ValueError("Select at least one episode, without duplicates")
        self.loader = LeRobotEpisodeLoader(dataset_root, modality_configs)
        positions = {int(ep["episode_index"]): i for i, ep in enumerate(self.loader.episodes_metadata)}
        missing = set(selected).difference(positions)
        if missing:
            raise ValueError(f"Unknown episode IDs: {sorted(missing)}")
        self.episode_indices = selected
        self.modality_configs = modality_configs
        self.processor = processor
        self.episodes = []
        for episode_id in selected:
            position = positions[episode_id]
            episode = self.loader[position]
            if len(episode) != self.loader.get_episode_length(position):
                raise ValueError(f"Episode {episode_id} length differs from metadata")
            self.episodes.append(episode)
        self.ends = np.cumsum([len(episode) for episode in self.episodes])

    def __len__(self):
        return int(self.ends[-1])

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode_position = int(np.searchsorted(self.ends, index, side="right"))
        start = 0 if episode_position == 0 else int(self.ends[episode_position - 1])
        step = extract_step_data(
            self.episodes[episode_position],
            index - start,
            self.modality_configs,
            EmbodimentTag.LIBERO_PANDA,
            allow_padding=True,
        )
        messages = [{"type": MessageType.EPISODE_STEP.value, "content": step}]
        return self.processor(messages) if self.processor else messages

    def get_dataset_statistics(self):
        # Match the reference protocol: identical normalization across all budgets.
        return self.loader.get_dataset_statistics()
