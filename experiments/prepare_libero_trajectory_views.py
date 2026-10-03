#!/usr/bin/env python3
"""Prepare deterministic nested LIBERO trajectory-budget dataset views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from gr00t.data.trajectory_selection import create_or_validate_manifest, selected_episode_indices


DEFAULT_BUDGETS = (10, 15, 25, 50)


def read_json_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_json_lines(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def prepare_view(root: Path, suite: str, count: int, seed: int, shared: dict) -> Path:
    source = root / "datasets" / f"libero_{suite}_no_noops_1.0.0_lerobot"
    view = root / "datasets" / f"libero_{suite}_seed{seed}_trajectories_{count:03d}"
    selected = selected_episode_indices(shared, count)

    expected_manifest = {
        "suite": f"libero_{suite}",
        "trajectory_count": count,
        "total_frames": 0,
        "seed": seed,
        "selection": shared["selection"],
        "dataset_metadata_sha256": shared["dataset_metadata_sha256"],
        "ordered_episode_indices": selected,
        "episodes": [],
    }

    source_meta = source / "meta"
    episodes_by_id = {
        int(row["episode_index"]): row for row in read_json_lines(source_meta / "episodes.jsonl")
    }
    episodes = [episodes_by_id[index] for index in selected]
    expected_manifest["total_frames"] = sum(int(row["length"]) for row in episodes)
    records_by_id = {int(row["episode_index"]): row for row in shared["episodes"]}
    expected_manifest["episodes"] = [records_by_id[index] for index in selected]

    if view.exists():
        manifest_path = view / "trajectory_manifest.json"
        if not manifest_path.is_file() or json.loads(manifest_path.read_text()) != expected_manifest:
            raise RuntimeError(f"Existing view is incomplete or does not match: {view}")
        return view

    view.mkdir(parents=True)
    shutil.copytree(source_meta, view / "meta")
    (view / "data").symlink_to(source / "data", target_is_directory=True)
    (view / "videos").symlink_to(source / "videos", target_is_directory=True)
    write_json_lines(view / "meta/episodes.jsonl", episodes)

    episode_stats_path = source_meta / "episodes_stats.jsonl"
    if episode_stats_path.exists():
        stats_by_id = {
            int(row["episode_index"]): row for row in read_json_lines(episode_stats_path)
        }
        write_json_lines(
            view / "meta/episodes_stats.jsonl",
            [stats_by_id[index] for index in selected],
        )

    info = json.loads((source_meta / "info.json").read_text())
    info["total_episodes"] = count
    info["total_frames"] = expected_manifest["total_frames"]
    info["total_videos"] = count * 2
    info["splits"] = {"train": f"0:{count}"}
    write_json(view / "meta/info.json", info)
    write_json(view / "trajectory_manifest.json", expected_manifest)
    return view


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--suite", choices=("goal", "object"), action="append", required=True)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--budgets", type=int, nargs="+", default=DEFAULT_BUDGETS)
    args = parser.parse_args()

    root = args.root.resolve()
    for suite in args.suite:
        source = root / "datasets" / f"libero_{suite}_no_noops_1.0.0_lerobot"
        shared_path = root / "datasets/trajectory_manifests" / f"libero_{suite}_seed{args.seed}.json"
        shared = create_or_validate_manifest(source, shared_path, args.seed)
        for count in args.budgets:
            view = prepare_view(root, suite, count, args.seed, shared)
            manifest = json.loads((view / "trajectory_manifest.json").read_text())
            print(f"prepared={view} frames={manifest['total_frames']}")


if __name__ == "__main__":
    main()
