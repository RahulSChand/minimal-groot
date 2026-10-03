#!/usr/bin/env python3
"""Run the N1.7 LIBERO Spatial trajectory-budget campaign and publish each epoch."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from gr00t.data.trajectory_selection import create_or_validate_manifest, selected_episode_indices


ROOT = Path("/workspace/minimal-groot")
SOURCE_DATASET = ROOT / "datasets/libero_spatial_no_noops_1.0.0_lerobot"
MODEL = ROOT / "checkpoints/GR00T-N1.7-3B"
OUTPUT_ROOT = ROOT / "outputs/n1d7-spatial-seed43"
SHARED_MANIFEST = OUTPUT_ROOT / "trajectory_manifest_seed43.json"
TRAIN_SCRIPT = Path("/workspace/post_train_vla/scripts/finetune_groot.py")
PYTHON = ROOT / ".venv/bin/python"
HF_REPO = "Chand0320/groot-n1d7-libero-spatial-trajectory-efficiency"
SEED = 43
BUDGETS = (10, 25, 50)
EPOCHS = 7
BATCH_SIZE = 8
ACCUMULATION = 6
LEARNING_RATE = 1e-5

active_child: subprocess.Popen | None = None


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def read_json_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json_lines(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def prepare_view(count: int, shared: dict) -> Path:
    view = ROOT / "datasets" / f"libero_spatial_seed43_trajectories_{count:03d}"
    selected = selected_episode_indices(shared, count)
    selected_set = set(selected)
    source_meta = SOURCE_DATASET / "meta"
    view_meta = view / "meta"

    if view.exists():
        existing = json.loads((view / "trajectory_manifest.json").read_text())
        if existing["ordered_episode_indices"] != selected:
            raise RuntimeError(f"Existing dataset view does not match seed {SEED}: {view}")
        return view

    view.mkdir(parents=True)
    shutil.copytree(source_meta, view_meta)
    (view / "data").symlink_to(SOURCE_DATASET / "data", target_is_directory=True)
    (view / "videos").symlink_to(SOURCE_DATASET / "videos", target_is_directory=True)

    episodes_by_id = {int(row["episode_index"]): row for row in read_json_lines(source_meta / "episodes.jsonl")}
    episodes = [episodes_by_id[index] for index in selected]
    write_json_lines(view_meta / "episodes.jsonl", episodes)

    stats_path = source_meta / "episodes_stats.jsonl"
    if stats_path.exists():
        stats_by_id = {int(row["episode_index"]): row for row in read_json_lines(stats_path)}
        write_json_lines(view_meta / "episodes_stats.jsonl", [stats_by_id[index] for index in selected])

    total_frames = sum(int(row["length"]) for row in episodes)
    info = json.loads((source_meta / "info.json").read_text())
    info["total_episodes"] = count
    info["total_frames"] = total_frames
    info["total_videos"] = count * 2
    info["splits"] = {"train": f"0:{count}"}
    write_json(view_meta / "info.json", info)

    selected_records = [row for row in shared["episodes"] if int(row["episode_index"]) in selected_set]
    selected_records.sort(key=lambda row: selected.index(int(row["episode_index"])))
    manifest = {
        "suite": "libero_spatial",
        "trajectory_count": count,
        "total_frames": total_frames,
        "seed": SEED,
        "selection": shared["selection"],
        "dataset_metadata_sha256": shared["dataset_metadata_sha256"],
        "ordered_episode_indices": selected,
        "episodes": selected_records,
    }
    write_json(view / "trajectory_manifest.json", manifest)
    return view


def training_command(count: int, dataset: Path, output: Path, smoke: bool = False) -> list[str]:
    command = [
        str(PYTHON),
        str(TRAIN_SCRIPT),
        "--version", "n1d7",
        "--checkpoint", str(MODEL),
        "--dataset", str(dataset),
        "--output", str(output),
        "--epochs", str(EPOCHS),
        "--batch-size", str(BATCH_SIZE),
        "--accumulation", str(ACCUMULATION),
        "--learning-rate", str(LEARNING_RATE),
        "--seed", str(SEED),
    ]
    if smoke:
        command.append("--smoke")
    return command


def publish_checkpoint(count: int, output: Path, epoch: int) -> None:
    checkpoint = output / f"epoch-{epoch:03d}"
    receipt_dir = output / "publication_receipts"
    receipt = receipt_dir / f"epoch-{epoch:03d}.json"
    if receipt.exists():
        return
    if not (checkpoint / "epoch.json").is_file():
        return
    prefix = f"n1d7/spatial/seed-{SEED:03d}/trajectories-{count:03d}/epoch-{epoch:03d}"
    subprocess.run(
        [
            str(PYTHON), "-m", "post_train_vla.checkpoint_publication",
            str(checkpoint), "--repo", HF_REPO, "--prefix", prefix,
        ],
        check=True,
    )
    publication = json.loads((checkpoint / "publication.json").read_text())
    write_json(receipt, publication)
    shutil.rmtree(checkpoint)
    print(f"published_and_removed_local={prefix}", flush=True)


def run_budget(count: int, dataset: Path) -> None:
    global active_child
    output = OUTPUT_ROOT / f"trajectories-{count:03d}"
    status_path = output / "status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text())
        if status.get("status") == "complete" and all(
            (output / "publication_receipts" / f"epoch-{epoch:03d}.json").is_file()
            for epoch in range(1, EPOCHS + 1)
        ):
            print(f"already_complete=trajectories-{count:03d}", flush=True)
            return
        raise RuntimeError(f"Refusing to merge with an incomplete non-resumable run: {output}")

    print(f"starting=trajectories-{count:03d}", flush=True)
    active_child = subprocess.Popen(training_command(count, dataset, output))
    try:
        while active_child.poll() is None:
            for epoch in range(1, EPOCHS + 1):
                publish_checkpoint(count, output, epoch)
            time.sleep(10)
        return_code = active_child.wait()
        for epoch in range(1, EPOCHS + 1):
            publish_checkpoint(count, output, epoch)
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, active_child.args)
        status = json.loads(status_path.read_text())
        if status.get("status") != "complete":
            raise RuntimeError(f"Training exited without complete status: {status}")
    finally:
        if active_child is not None and active_child.poll() is None:
            active_child.terminate()
            active_child.wait(timeout=30)
        active_child = None


def handle_signal(signum: int, _frame: object) -> None:
    if active_child is not None and active_child.poll() is None:
        active_child.terminate()
    raise SystemExit(128 + signum)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    shared = create_or_validate_manifest(SOURCE_DATASET, SHARED_MANIFEST, SEED)
    views = {count: prepare_view(count, shared) for count in BUDGETS}
    if args.prepare_only:
        for count, view in views.items():
            manifest = json.loads((view / "trajectory_manifest.json").read_text())
            print(f"prepared={view} frames={manifest['total_frames']}")
        return
    if args.smoke:
        smoke_output = OUTPUT_ROOT / "smoke-trajectories-010"
        if smoke_output.exists():
            raise RuntimeError(f"Smoke output already exists: {smoke_output}")
        subprocess.run(training_command(10, views[10], smoke_output, smoke=True), check=True)
        return
    for count in BUDGETS:
        run_budget(count, views[count])
    print("campaign_complete=true", flush=True)


if __name__ == "__main__":
    main()
