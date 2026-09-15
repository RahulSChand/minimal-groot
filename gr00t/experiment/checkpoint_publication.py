"""Publish each epoch, reload its downloaded bytes offline, then remove local weights."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_checkpoint_assets(checkpoint, processor, dataset, plan, repository_root):
    import numpy as np

    from gr00t.data.types import EmbodimentTag

    checkpoint = Path(checkpoint)
    write_json(checkpoint / "training_manifest.json", plan)
    shutil.copytree(
        repository_root / "gr00t",
        checkpoint / "runtime/gr00t",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "libero_uv"),
        dirs_exist_ok=True,
    )
    for name in ("pyproject.toml", "uv.lock", "LICENSE", "ATTRIBUTIONS.md", "UPSTREAM.md"):
        shutil.copy2(repository_root / name, checkpoint / "runtime" / name)
    base_license = Path(plan["base_model_path"]) / "LICENSE"
    if base_license.is_file():
        shutil.copy2(base_license, checkpoint / "LICENSE")
    if processor.model_type == "qwen":
        processor.processor.save_pretrained(checkpoint / "vlm_assets")
    shutil.copytree(Path(plan["dataset_root"]) / "meta", checkpoint / "dataset_metadata", dirs_exist_ok=True)
    row = dataset.episodes[0].iloc[0]
    state_keys = processor.get_modality_configs()[EmbodimentTag.LIBERO_PANDA.value]["state"].modality_keys
    np.savez_compressed(
        checkpoint / "load_fixture.npz",
        image=row["video.image"],
        wrist_image=row["video.wrist_image"],
        state=np.concatenate([np.asarray(row[f"state.{key}"]).reshape(-1) for key in state_keys]),
        prompt=str(row["language.annotation.human.action.task_description"]),
    )
    (checkpoint / "README.md").write_text(
        "# GR00T LIBERO Spatial epoch checkpoint\n\n"
        "Weights, native processor assets, Spatial statistics, embodiment IDs, "
        "training manifests, and compatible runtime source are included. No optimizer state is saved.\n\n"
        "Install the dependencies in `runtime/pyproject.toml`, then set "
        "`PYTHONPATH=/absolute/path/to/checkpoint/runtime`. Load using "
        "`Gr00tPolicy('LIBERO_PANDA', '/absolute/path/to/checkpoint', device='cuda')` "
        "from `gr00t.policy.gr00t_policy`. The checkpoint can be loaded offline.\n\n"
        "`metadata.json` records the epoch results; `training_manifest.json` records "
        "the pinned base model, complete recipe and shared nested subsets. "
        "`upload_verification.json` records the fresh-download load check.\n"
    )


def verify_offline(checkpoint, output):
    import numpy as np
    import torch
    from transformers import AutoModel, AutoProcessor

    import gr00t.model  # noqa: F401
    from gr00t.eval.reference_libero import ReferenceLiberoPolicy

    torch.set_num_threads(4)
    model, info = AutoModel.from_pretrained(str(checkpoint), output_loading_info=True, local_files_only=True)
    if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"Uploaded weights did not load strictly: {info}")
    if any(p.is_meta or not torch.isfinite(p).all() for p in model.parameters()):
        raise RuntimeError("Checkpoint contains meta or nonfinite parameters")
    processor = AutoProcessor.from_pretrained(str(checkpoint), local_files_only=True)
    processor.eval()
    model.eval().to("cuda")
    torch.manual_seed(42)
    with np.load(checkpoint / "load_fixture.npz", allow_pickle=False) as fixture:
        observation = {"observation/" + key: fixture[key] for key in ("image", "wrist_image", "state")}
        observation["prompt"] = str(fixture["prompt"])
    actions = ReferenceLiberoPolicy(model, processor).infer(observation)["actions"]
    if actions.shape != (model.config.action_horizon, 7) or not np.isfinite(actions).all():
        raise RuntimeError(f"Invalid action output: {actions.shape}")
    write_json(
        output,
        {
            "strict_weight_load": True,
            "fresh_process": True,
            "empty_hub_cache": True,
            "offline": True,
            "inference_passed": True,
            "action_shape": list(actions.shape),
            "model_type": model.config.model_type,
            "parameter_count": sum(p.numel() for p in model.parameters()),
        },
    )


def publish_verify_delete(checkpoint, repo_id, folder, archive, *, api=None):
    from huggingface_hub import HfApi, snapshot_download

    checkpoint, archive = Path(checkpoint), Path(archive)
    archive.mkdir(parents=True, exist_ok=True)
    api = api or HfApi()
    # A retained checkpoint may already have a manifest after a failed attempt.
    # Rebuild it from the artifacts without including its own previous digest.
    checksum_manifest = checkpoint / "artifact_checksums.json"
    files = {
        p.relative_to(checkpoint).as_posix(): sha256(p)
        for p in sorted(checkpoint.rglob("*"))
        if p.is_file() and p != checksum_manifest
    }
    if any("optimizer" in name.lower() for name in files):
        raise ValueError("Optimizer files must never be published")
    write_json(checkpoint / "artifact_checksums.json", files)
    files["artifact_checksums.json"] = sha256(checkpoint / "artifact_checksums.json")
    print(f"Uploading {repo_id}/{folder} ({len(files)} files)", flush=True)
    commit = api.upload_folder(
        repo_id=repo_id,
        folder_path=str(checkpoint),
        path_in_repo=folder,
        commit_message=f"Verified Spatial training: {folder}",
    )
    revision = commit.oid
    with tempfile.TemporaryDirectory(prefix="verify-epoch-", dir=checkpoint.parent) as temporary:
        temporary = Path(temporary)
        snapshot_download(
            repo_id,
            revision=revision,
            allow_patterns=[folder + "/*"],
            local_dir=temporary / "download",
            cache_dir=temporary / "download-cache",
        )
        downloaded = temporary / "download" / folder
        for name, digest in files.items():
            if sha256(downloaded / name) != digest:
                raise RuntimeError(f"Upload checksum mismatch: {name}")
        env = os.environ.copy()
        env.update(
            HF_HOME=str(temporary / "empty-hub-cache"),
            HF_HUB_CACHE=str(temporary / "empty-hub-cache/hub"),
            HF_MODULES_CACHE=str(temporary / "empty-hub-cache/modules"),
            TRANSFORMERS_CACHE=str(temporary / "empty-hub-cache/transformers"),
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            HF_DATASETS_OFFLINE="1",
            PYTHONPATH=str(downloaded / "runtime"),
        )
        result_path = temporary / "verification.json"
        with (archive / "load_verification.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "gr00t.experiment.checkpoint_publication",
                    "--checkpoint",
                    str(downloaded),
                    "--output",
                    str(result_path),
                ],
                cwd=downloaded,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        receipt = json.loads(result_path.read_text())
        receipt.update(
            repo_id=repo_id, folder=folder, content_revision=revision, files_verified=len(files), checksums_match=True
        )
        write_json(archive / "upload_verification.json", receipt)
        receipt_commit = api.upload_file(
            repo_id=repo_id,
            path_or_fileobj=str(archive / "upload_verification.json"),
            path_in_repo=folder + "/upload_verification.json",
            commit_message=f"Offline reload verified: {folder}",
        )
        receipt["verification_revision"] = receipt_commit.oid
    # Reached only after a successful round-trip and strict offline inference.
    shutil.copytree(checkpoint / "evaluation", archive / "evaluation", dirs_exist_ok=True)
    for name in ("metadata.json", "artifact_checksums.json"):
        shutil.copy2(checkpoint / name, archive / name)
    shutil.rmtree(checkpoint)
    receipt["local_checkpoint_deleted"] = True
    write_json(archive / "upload_verification.json", receipt)
    print(f"Verified and removed local checkpoint: {folder} @ {revision}", flush=True)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify_offline(args.checkpoint, args.output)
