import json
import random
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from gr00t.data.dataset.trajectory_subset import (
    TrajectorySubsetDataset,
    create_or_validate_manifest,
)
from gr00t.data.types import ModalityConfig
from gr00t.experiment.launch_finetune import select_model_config
from gr00t.experiment.sample_efficiency import stopping_state, train_one_epoch


def test_manifest_nested_global_selection_and_metadata_validation(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    episodes = [{"episode_index": i, "length": 3, "tasks": [f"task-{i % 2}"]} for i in range(60)]
    (meta / "episodes.jsonl").write_text("\n".join(map(json.dumps, reversed(episodes))))
    (meta / "tasks.jsonl").write_text("\n".join(json.dumps({"task": f"task-{i}", "task_index": i}) for i in range(2)))
    path = tmp_path / "manifest.json"
    manifest = create_or_validate_manifest(tmp_path, path, 42)
    expected = list(range(60))
    random.Random(42).shuffle(expected)
    assert manifest["ordered_episode_indices"] == expected
    assert create_or_validate_manifest(tmp_path, path, 42) == manifest
    for small, large in zip([5, 10, 15, 25], [10, 15, 25, 50]):
        assert set(expected[:small]) < set(expected[:large])
    with pytest.raises(ValueError, match="Manifest does not match"):
        create_or_validate_manifest(tmp_path, path, 43)
    (meta / "episodes.jsonl").write_text((meta / "episodes.jsonl").read_text() + "\n")
    with pytest.raises(ValueError, match="Manifest does not match"):
        create_or_validate_manifest(tmp_path, path, 42)


def test_exact_frames_and_padding_never_cross_episode_boundaries(monkeypatch):
    class Loader:
        episodes_metadata: ClassVar[list[dict]] = [{"episode_index": 19}, {"episode_index": 3}]

        def __init__(self, *args):
            self.loaded = []

        def get_episode_length(self, position):
            return [3, 2][position]

        def __getitem__(self, position):
            self.loaded.append(position)
            values = [100, 101, 102] if position == 0 else [200, 201]
            return pd.DataFrame({"state.x": values, "action.x": values, "language.task": ["task"] * len(values)})

    monkeypatch.setattr("gr00t.data.dataset.trajectory_subset.LeRobotEpisodeLoader", Loader)
    modalities = {
        "state": ModalityConfig(delta_indices=[0], modality_keys=["x"]),
        "action": ModalityConfig(delta_indices=[0, 1, 2], modality_keys=["x"]),
        "language": ModalityConfig(delta_indices=[0], modality_keys=["task"]),
    }
    dataset = TrajectorySubsetDataset("unused", [3, 19], modalities)
    assert dataset.loader.loaded == [1, 0]
    assert len(dataset) == 5
    assert [float(dataset[i][0]["content"].states["x"].item()) for i in range(5)] == [
        200,
        201,
        100,
        101,
        102,
    ]
    np.testing.assert_array_equal(dataset[1][0]["content"].actions["x"].ravel(), [201, 201, 201])
    np.testing.assert_array_equal(dataset[4][0]["content"].actions["x"].ravel(), [102, 102, 102])
    with pytest.raises(IndexError):
        dataset[5]
    with pytest.raises(ValueError, match="duplicates"):
        TrajectorySubsetDataset("unused", [3, 3], modalities)


@pytest.mark.parametrize(
    "model_type,version",
    [("gr00t_n1", "1"), ("gr00t_n1_5", "1.5"), ("Gr00tN1d6", "1.6"), ("Gr00tN1d7", "1.7")],
)
def test_model_selection_uses_checkpoint_metadata(tmp_path, model_type, version):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
    assert select_model_config(str(tmp_path)).model_type == model_type
    assert select_model_config(str(tmp_path), "N" + version).model_type == model_type
    with pytest.raises(ValueError, match="Requested"):
        select_model_config(str(tmp_path), "1.4")


def test_partial_accumulation_group_gets_full_weight(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.5))

        def forward(self, x):
            return {"loss": (x * self.weight).square().mean()}

    model = Toy()
    loader = DataLoader([{"x": torch.tensor(x)} for x in [0.1, 0.2, 0.3, 0.4, 0.5]], batch_size=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    step, loss, count = train_one_epoch(model, list(model.parameters()), optimizer, loader, 3, 4)
    expected = 0.5
    for xs in ([0.1, 0.2, 0.3], [0.4, 0.5]):
        expected -= 0.1 * 2 * expected * np.mean(np.square(xs))
    assert (step, count) == (6, 2)
    assert loss > 0
    assert model.weight.item() == pytest.approx(expected)


def test_early_stopping_requires_strict_success_improvement():
    state = None
    for epoch, successes in enumerate([1, 2, 2, 2], 1):
        state = stopping_state(state, epoch, successes)
    assert state == {"best_epoch": 2, "best_successes": 2, "epochs_without_improvement": 2}
    assert stopping_state(state, 5, 3)["epochs_without_improvement"] == 0
