import json
import random

import pytest

from gr00t.configs.data.data_config import DataConfig
from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.data.trajectory_selection import create_or_validate_manifest, selected_episode_indices


def _write_metadata(root, count=8):
    meta = root / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(json.dumps({"total_episodes": count}))
    (meta / "tasks.jsonl").write_text(
        "\n".join(json.dumps({"task_index": index, "task": f"task {index}"}) for index in range(2)) + "\n"
    )
    (meta / "episodes.jsonl").write_text(
        "\n".join(
            json.dumps({"episode_index": index, "tasks": [f"task {index % 2}"], "length": 10 + index})
            for index in range(count)
        )
        + "\n"
    )


def test_default_seed_is_43():
    assert DataConfig().seed == 43
    assert FinetuneConfig("model", "dataset", "robot").seed == 43


def test_manifest_is_deterministic_and_budgets_are_nested(tmp_path):
    _write_metadata(tmp_path)
    manifest_path = tmp_path / "manifest.json"

    manifest = create_or_validate_manifest(tmp_path, manifest_path, seed=43)

    expected = list(range(8))
    random.Random(43).shuffle(expected)
    assert manifest["ordered_episode_indices"] == expected
    assert selected_episode_indices(manifest, 5) == selected_episode_indices(manifest, 7)[:5]
    assert create_or_validate_manifest(tmp_path, manifest_path, seed=43) == manifest


def test_manifest_rejects_different_seed(tmp_path):
    _write_metadata(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    create_or_validate_manifest(tmp_path, manifest_path, seed=43)

    with pytest.raises(ValueError, match="does not match"):
        create_or_validate_manifest(tmp_path, manifest_path, seed=42)


def test_budget_cannot_exceed_available_episodes(tmp_path):
    _write_metadata(tmp_path, count=3)
    manifest = create_or_validate_manifest(tmp_path, tmp_path / "manifest.json", seed=43)

    with pytest.raises(ValueError, match="contains only 3"):
        selected_episode_indices(manifest, 4)
