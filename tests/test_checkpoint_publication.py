import hashlib
import json
import shutil
from types import SimpleNamespace

import pytest

from gr00t.experiment.checkpoint_publication import publish_verify_delete


@pytest.mark.parametrize("load_succeeds", [True, False])
def test_local_weights_survive_until_fresh_download_load_succeeds(tmp_path, monkeypatch, load_succeeds):
    checkpoint = tmp_path / "epoch-001"
    (checkpoint / "evaluation").mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"test weights")
    (checkpoint / "metadata.json").write_text("{}")
    (checkpoint / "runtime").mkdir()
    events = []

    class Api:
        def upload_folder(self, **kwargs):
            events.append("upload")
            assert checkpoint.exists()
            manifest = json.loads((checkpoint / "artifact_checksums.json").read_text())
            for name, digest in manifest.items():
                assert hashlib.sha256((checkpoint / name).read_bytes()).hexdigest() == digest
            return SimpleNamespace(oid="content-sha")

        def upload_file(self, **kwargs):
            events.append("receipt")
            assert checkpoint.exists()
            return SimpleNamespace(oid="receipt-sha")

    def download(repo_id, *, revision, allow_patterns, local_dir, cache_dir):
        events.append("download")
        assert revision == "content-sha"
        shutil.copytree(checkpoint, local_dir / "n1/trajectories-005/epoch-001")

    def load(command, *, cwd, env, **kwargs):
        events.append("load")
        assert checkpoint.exists()
        assert cwd != checkpoint
        assert env["HF_HUB_OFFLINE"] == "1"
        assert not (tmp_path / env["HF_HOME"]).exists()
        if not load_succeeds:
            raise RuntimeError("Load failed")
        from pathlib import Path

        Path(command[-1]).write_text(json.dumps({"inference_passed": True}))

    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    monkeypatch.setattr("subprocess.run", load)
    if load_succeeds:
        result = publish_verify_delete(
            checkpoint, "test/repo", "n1/trajectories-005/epoch-001", tmp_path / "archive", api=Api()
        )
        assert result["local_checkpoint_deleted"]
        assert not checkpoint.exists()
        assert events == ["upload", "download", "load", "receipt"]
    else:
        with pytest.raises(RuntimeError, match="Load failed"):
            publish_verify_delete(
                checkpoint, "test/repo", "n1/trajectories-005/epoch-001", tmp_path / "archive", api=Api()
            )
        assert (checkpoint / "model.safetensors").exists()
        assert events == ["upload", "download", "load"]
        load_succeeds = True
        result = publish_verify_delete(
            checkpoint, "test/repo", "n1/trajectories-005/epoch-001", tmp_path / "archive", api=Api()
        )
        assert result["local_checkpoint_deleted"]
        assert not checkpoint.exists()
        assert events == ["upload", "download", "load", "upload", "download", "load", "receipt"]
