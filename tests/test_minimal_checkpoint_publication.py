import importlib.util
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gr00t.experiment.minimal_checkpoint_publication import (
    publication_files,
    publish,
    retry_hub,
    transient_retry_delay,
)


def load_finite_trainer():
    path = Path(__file__).resolve().parents[1] / "experiments/finetune_groot.py"
    spec = importlib.util.spec_from_file_location("finite_groot_trainer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_saver_emits_only_inference_checkpoint_files(tmp_path):
    trainer = load_finite_trainer()

    class Model:
        def state_dict(self):
            return {}

        def save_pretrained(self, folder, state_dict):
            assert state_dict == {}
            (folder / "config.json").write_text("{}")
            (folder / "model.safetensors").write_bytes(b"weights")

    class Processor:
        def save_pretrained(self, folder):
            (folder / "processor_config.json").write_text("{}")
            (folder / "statistics.json").write_text("{}")
            (folder / "embodiment_id.json").write_text("{}")

    args = SimpleNamespace(output=tmp_path, epochs=7, version="n1d7")
    trainer.save_epoch(args, Model(), Processor(), object(), epoch=1, step=30)
    checkpoint = tmp_path / "epoch-001"
    names = {path.name for path in checkpoint.iterdir()}

    assert names == {
        "checkpoint_manifest.json",
        "config.json",
        "embodiment_id.json",
        "epoch.json",
        "model.safetensors",
        "processor_config.json",
        "statistics.json",
    }
    manifest = json.loads((checkpoint / "checkpoint_manifest.json").read_text())
    assert manifest["optimizer_saved"] is False
    assert manifest["scheduler_saved"] is False
    assert "runtime" not in names
    assert "campaign_code" not in names


def test_publication_files_uses_manifest_allowlist(tmp_path):
    checkpoint = tmp_path / "epoch-001"
    checkpoint.mkdir()
    for name in (
        "config.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
        "processor_config.json",
        "statistics.json",
        "embodiment_id.json",
        "epoch.json",
    ):
        (checkpoint / name).write_text(name)
    (checkpoint / "runtime").mkdir()
    (checkpoint / "runtime/gr00t.py").write_text("must not upload")
    (checkpoint / "campaign_code").mkdir()
    (checkpoint / "campaign_code/train.py").write_text("must not upload")
    allowed = sorted(
        path.name for path in checkpoint.iterdir() if path.is_file()
    )
    (checkpoint / "checkpoint_manifest.json").write_text(
        json.dumps({"format": "groot-inference-checkpoint-v1", "files": allowed})
    )

    selected = {str(path.relative_to(checkpoint)) for path in publication_files(checkpoint)}

    assert "checkpoint_manifest.json" in selected
    assert "runtime/gr00t.py" not in selected
    assert "campaign_code/train.py" not in selected
    assert len(selected) == 9


def test_publication_files_rejects_path_traversal(tmp_path):
    checkpoint = tmp_path / "epoch-001"
    checkpoint.mkdir()
    (checkpoint / "checkpoint_manifest.json").write_text(
        json.dumps({"format": "groot-inference-checkpoint-v1", "files": ["../secret"]})
    )

    with pytest.raises(ValueError, match="Unsafe checkpoint path"):
        publication_files(checkpoint)


def test_publish_passes_exact_allowlist_to_hub_api(tmp_path, monkeypatch):
    checkpoint = tmp_path / "epoch-001"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    (checkpoint / "epoch.json").write_text("{}")
    (checkpoint / "runtime").mkdir()
    (checkpoint / "runtime/source.py").write_text("must not upload")
    (checkpoint / "checkpoint_manifest.json").write_text(
        json.dumps(
            {
                "format": "groot-inference-checkpoint-v1",
                "files": ["config.json", "epoch.json", "model.safetensors"],
            }
        )
    )
    prefix = "n1d7/goal/seed-043/trajectories-010/epoch-001"

    class Api:
        def upload_folder(self, **kwargs):
            assert kwargs["allow_patterns"] == [
                "artifact_checksums.json",
                "checkpoint_manifest.json",
                "config.json",
                "epoch.json",
                "model.safetensors",
            ]
            assert "runtime/source.py" not in kwargs["allow_patterns"]
            return SimpleNamespace(oid="test-revision")

        def get_paths_info(self, _repo, paths, revision):
            assert revision == "test-revision"
            rows = []
            for remote_path in paths:
                relative = remote_path.removeprefix(prefix + "/")
                local = checkpoint / relative
                rows.append(
                    SimpleNamespace(
                        path=remote_path,
                        size=local.stat().st_size,
                        lfs=SimpleNamespace(sha256=hashlib.sha256(local.read_bytes()).hexdigest()),
                    )
                )
            return rows

    monkeypatch.setattr("huggingface_hub.HfApi", lambda: Api())
    monkeypatch.setattr(
        "huggingface_hub.get_hf_file_metadata",
        lambda _url: SimpleNamespace(size=(checkpoint / "model.safetensors").stat().st_size),
    )

    receipt = publish(checkpoint, repo="test/repo", prefix=prefix)

    assert receipt["revision"] == "test-revision"
    assert receipt["verified_files"] == 5


def test_rate_limit_waits_for_full_window_and_retries(monkeypatch):
    class RateLimited(Exception):
        response = SimpleNamespace(status_code=429, headers={})

    attempts = []
    sleeps = []

    def operation():
        attempts.append(1)
        if len(attempts) == 1:
            raise RateLimited("slow down")
        return "uploaded"

    monkeypatch.setattr("gr00t.experiment.minimal_checkpoint_publication.time.sleep", sleeps.append)

    assert transient_retry_delay(RateLimited(), 0) == 310
    assert retry_hub("upload", operation) == "uploaded"
    assert len(attempts) == 2
    assert sleeps == [310]
