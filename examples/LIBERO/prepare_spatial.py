"""Validate the downloaded Spatial corpus and recompute its normalization."""

import argparse
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from gr00t.experiment.checkpoint_publication import sha256, write_json


def prepare(root, libero_root, source_repo, source_revision):
    info = json.loads((root / "meta/info.json").read_text())
    episodes = [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines()]
    tasks = [json.loads(line) for line in (root / "meta/tasks.jsonl").read_text().splitlines()]
    spatial_files = sorted((libero_root / "libero/libero/bddl_files/libero_spatial").glob("*.bddl"))
    if len(spatial_files) != 10:
        raise ValueError("Expected ten official LIBERO Spatial BDDL definitions")
    # LIBERO's benchmark API derives task.language from the BDDL filename.
    # The prose inside BDDL files uses older object names (e.g. "akita").
    prompts = {path.stem.replace("_", " ") for path in spatial_files}
    task_map = {int(item["task_index"]): item["task"] for item in tasks}
    if set(task_map.values()) != prompts:
        raise ValueError(f"Dataset task descriptions differ from Spatial: {set(task_map.values()) ^ prompts}")
    if len(episodes) != info["total_episodes"] or len(tasks) != 10:
        raise ValueError("Incorrect episode or task totals")
    paths = sorted(root.glob("data/*/*.parquet"))
    if len(paths) != len(episodes):
        raise ValueError("Missing or extra parquet files")
    arrays = {"state": [], "actions": []}
    file_hashes, task_counts = {}, Counter()
    seen, frame_count, image_count = set(), 0, 0
    for item in sorted(episodes, key=lambda item: item["episode_index"]):
        episode = int(item["episode_index"])
        if episode in seen:
            raise ValueError("Duplicate episode ID")
        seen.add(episode)
        path = root / info["data_path"].format(episode_chunk=episode // info["chunks_size"], episode_index=episode)
        table = pq.read_table(path)
        if table.num_rows != item["length"]:
            raise ValueError(f"Incorrect episode length: {path}")
        frame_count += table.num_rows
        episode_ids = set(table["episode_index"].to_pylist())
        task_ids = set(table["task_index"].to_pylist())
        if episode_ids != {episode} or len(task_ids) != 1:
            raise ValueError(f"Mixed episode or task IDs: {path}")
        task_id = int(next(iter(task_ids)))
        if item["tasks"] != [task_map[task_id]]:
            raise ValueError(f"Parquet task differs from episode metadata: {path}")
        if table["frame_index"].to_pylist() != list(range(table.num_rows)):
            raise ValueError(f"Missing or disordered frames: {path}")
        task_counts[task_id] += 1
        for name, width in (("state", 8), ("actions", 7)):
            value = np.asarray(table[name].to_pylist(), dtype=np.float32)
            if value.shape != (table.num_rows, width) or not np.isfinite(value).all():
                raise ValueError(f"Invalid {name}: {path}")
            arrays[name].append(value)
        for name in ("image", "wrist_image"):
            for record in table[name].to_pylist():
                if not record.get("bytes"):
                    raise ValueError(f"Missing embedded image: {path}")
                with Image.open(io.BytesIO(record["bytes"])) as image:
                    if image.size != (256, 256) or image.mode != "RGB":
                        raise ValueError(f"Invalid image format: {path}")
                    image.verify()
                image_count += 1
        file_hashes[path.relative_to(root).as_posix()] = sha256(path)
        if len(seen) % 50 == 0:
            print(f"Verified {len(seen)}/{len(episodes)} episodes", flush=True)
    if frame_count != info["total_frames"]:
        raise ValueError("Incorrect total frame count")
    stats = {}
    for key, chunks in arrays.items():
        value = np.concatenate(chunks)
        stats[key] = {
            name: function(value, axis=0).tolist()
            for name, function in (("mean", np.mean), ("std", np.std), ("min", np.min), ("max", np.max))
        }
        stats[key].update(q01=np.quantile(value, 0.01, axis=0).tolist(), q99=np.quantile(value, 0.99, axis=0).tolist())
    keys = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    modality = {
        kind: {
            key: {"start": i, "end": 8 if kind == "state" and key == "gripper" else i + 1, "original_key": source}
            for i, key in enumerate(keys)
        }
        for kind, source in (("state", "state"), ("action", "actions"))
    }
    modality["video"] = {key: {"original_key": key} for key in ("image", "wrist_image")}
    modality["annotation"] = {"human.action.task_description": {"original_key": "task_index"}}
    write_json(root / "meta/modality.json", modality)
    write_json(root / "meta/stats.json", stats)
    write_json(root / "meta/spatial_file_checksums.json", file_hashes)
    verification = {
        "suite": "libero_spatial",
        "source_repo": source_repo,
        "source_revision": source_revision,
        "dataset_root": str(root),
        "episodes": len(seen),
        "frames": frame_count,
        "images_verified": image_count,
        "task_descriptions_match_official_spatial": True,
        "tasks": tasks,
        "task_episode_counts": dict(task_counts),
        "libero_commit": "8f1084e3132a39270c3a13ebe37270a43ece2a01",
        "statistics_method": "Full Spatial corpus: float32 mean/std/min/max; numpy q01/q99",
        "statistics_sha256": sha256(root / "meta/stats.json"),
        "data_checksums_sha256": sha256(root / "meta/spatial_file_checksums.json"),
        "normalization_scope": "all_verified_spatial_episodes_shared_across_budgets",
    }
    write_json(root / "meta/spatial_verification.json", verification)
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--source-repo", required=True)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    prepare(args.dataset_root, args.libero_root, args.source_repo, args.source_revision)
