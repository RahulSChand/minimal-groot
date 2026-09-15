"""Audit all twenty completed runs and publish a compact campaign summary."""

import csv
import hashlib
import json
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from huggingface_hub.utils import RevisionNotFoundError

from gr00t.experiment.checkpoint_publication import write_json
from gr00t.experiment.sample_efficiency import stopping_state

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/spatial-trajectory-efficiency-20260915"


def main():
    api = HfApi()
    manifest = json.loads((OUTPUT / "trajectory_manifest_seed42.json").read_text())
    existing_runs = json.loads((OUTPUT / "existing_runs_at_budget_extension.json").read_text())
    completed_original_runs = json.loads((OUTPUT / "completed_original_runs_at_extension_start.json").read_text())
    assert len(completed_original_runs) == 16
    spatial_stats_sha256 = hashlib.sha256(Path("/root/liber_spatial_post/meta/stats.json").read_bytes()).hexdigest()
    rows, verified_epochs = [], 0
    for version in ("1", "1.5", "1.6", "1.7"):
        directory = OUTPUT / f"n{version}"
        repo = f"kaweees/gr00tn{version}-libero-spatial-trajectory-efficiency"
        plan = json.loads((directory / "run_plan.json").read_text())
        assert plan["base_model_repo"].startswith("nvidia/GR00T-N")
        assert plan["base_model_revision"] and plan["training"]["max_epochs"] is None
        assert plan["training"]["seed"] == 42 and plan["training"]["patience"] == 2
        assert plan["dataset_root"] == "/root/liber_spatial_post"
        assert plan["dataset_provenance"]["suite"] == "libero_spatial"
        assert plan["dataset_provenance"]["statistics_sha256"] == spatial_stats_sha256
        assert plan["evaluation"]["episodes"] == 20 and plan["evaluation"]["task_id"] == 0
        remote = {entry.path: entry for entry in api.list_repo_tree(repo, recursive=True) if hasattr(entry, "size")}
        cleanup = json.loads((OUTPUT / "legacy_history_cleanup" / f"n{version}.json").read_text())
        assert cleanup["repo_id"] == repo and cleanup["repository_recreated_from_verified_snapshot"]
        assert cleanup["all_preserved_file_hashes_unchanged"] and cleanup["legacy_paths_absent"]
        assert cleanup["legacy_exclusive_lfs_objects_absent"] and cleanup["old_revision_file_url_returns_404"]
        assert not cleanup["oldest_pre_experiment_revision_resolvable"]
        if version == "1":
            assert not any(
                path.startswith("n1/trajectories-") and "/epoch-" not in path and path.endswith(".safetensors")
                for path in remote
            )
        else:
            legacy_folder = {"1.5": "n15/", "1.6": "n16/", "1.7": "n17/"}[version]
            assert not any(path.startswith(legacy_folder) for path in remote)
        try:
            api.repo_info(repo, revision=cleanup["oldest_pre_experiment_revision"])
        except RevisionNotFoundError:
            pass
        else:
            raise AssertionError(f"Pre-experiment revision still accessible: {repo}")
        license_path = hf_hub_download(plan["base_model_repo"], "LICENSE", revision=plan["base_model_revision"])
        supplemental_files = []
        version_rows = []
        assert [run["trajectory_count"] for run in plan["runs"]] == [5, 10, 15, 25, 50]
        for budget in (5, 10, 15, 25, 50):
            run = directory / f"trajectories-{budget:03d}"
            summary = json.loads((run / "run_summary.json").read_text())
            assert summary["complete"] and summary["stop_reason"] == "early_stopping"
            previous = existing_runs.get(str(run.relative_to(OUTPUT)))
            if previous:
                assert summary["epochs"][: len(previous["epochs"])] == previous["epochs"]
                if previous["complete"]:
                    assert summary == previous, f"Previously completed run changed: {run}"
            completed_original = completed_original_runs.get(str(run.relative_to(OUTPUT)))
            if completed_original:
                assert summary == completed_original, f"Completed original run changed: {run}"
            assert summary["selected_episode_indices"] == manifest["ordered_episode_indices"][:budget]
            assert summary["total_parameters"] == summary["trainable_parameters"]
            state = None
            for epoch, record in enumerate(summary["epochs"], 1):
                assert record["epoch"] == epoch and record["episodes"] == 20
                state = stopping_state(state, epoch, record["successes"])
                assert (state["epochs_without_improvement"] >= 2) == (epoch == len(summary["epochs"]))
                receipt = record["upload"]
                for key in (
                    "strict_weight_load",
                    "checksums_match",
                    "fresh_process",
                    "empty_hub_cache",
                    "offline",
                    "inference_passed",
                    "local_checkpoint_deleted",
                ):
                    assert receipt[key], (run, epoch, key)
                folder = f"n{version}/trajectories-{budget:03d}/epoch-{epoch:03d}"
                assert receipt["repo_id"] == repo and receipt["folder"] == folder
                checksums = json.loads(
                    (run / "evaluations" / f"epoch-{epoch:03d}" / "artifact_checksums.json").read_text()
                )
                assert checksums["dataset_metadata/stats.json"] == spatial_stats_sha256
                for name, digest in checksums.items():
                    assert folder + "/" + name in remote, (repo, folder, name)
                    if name.endswith(".safetensors"):
                        assert remote[folder + "/" + name].lfs.sha256 == digest
                assert folder + "/upload_verification.json" in remote
                if folder + "/LICENSE" not in remote:
                    supplemental_files.append(
                        CommitOperationAdd(path_in_repo=folder + "/LICENSE", path_or_fileobj=license_path)
                    )
                assert not (run / f"epoch-{epoch:03d}").exists()
                verified_epochs += 1
            assert state["best_successes"] == summary["best_successes"]
            row = {
                "model_version": version,
                "trajectories": budget,
                "frames": summary["frames"],
                "epochs": len(summary["epochs"]),
                "best_epoch": summary["best_epoch"],
                "best_successes": summary["best_successes"],
                "rollouts": 20,
                "epoch_successes": ",".join(str(e["successes"]) for e in summary["epochs"]),
                "best_checkpoint": summary["best_checkpoint"],
            }
            rows.append(row)
            version_rows.append(row)
        supplemental_files.append(
            CommitOperationAdd(
                path_in_repo=f"n{version}/official_spatial_crosscheck.json",
                path_or_fileobj=str(OUTPUT / "official_spatial_crosscheck.json"),
            )
        )
        api.create_commit(
            repo_id=repo,
            operations=supplemental_files,
            commit_message="Include base license and independent Spatial source verification",
        )
        report = {
            "model_version": version,
            "complete": True,
            "runs": version_rows,
            "dataset": plan["dataset_provenance"],
            "base_model_repo": plan["base_model_repo"],
            "base_model_revision": plan["base_model_revision"],
            "training": plan["training"],
            "evaluation": plan["evaluation"],
            "legacy_files_and_history_removed": True,
        }
        write_json(directory / "campaign_summary.json", report)
        api.upload_file(
            repo_id=repo,
            path_or_fileobj=str(directory / "campaign_summary.json"),
            path_in_repo=f"n{version}/campaign_summary.json",
            commit_message="Audit completed Spatial campaign",
        )
        lines = [
            "---",
            f"base_model: {plan['base_model_repo']}",
            "tags: [gr00t, libero-spatial]",
            "---",
            "",
            f"# GR00T N{version}: LIBERO Spatial trajectory efficiency",
            "",
            "Five independent runs use 5, 10, 15, 25, and 50 total trajectories, seed 42, "
            "and the same nested subsets across all four model versions. Every run starts "
            "from the pinned official NVIDIA base. Previously completed runs were retained and skipped.",
            "",
            "| Trajectories | Saved epochs | Best epoch | Best task-0 successes / 20 |",
            "|---:|---:|---:|---:|",
        ]
        for row in version_rows:
            folder_url = f"https://huggingface.co/{repo}/tree/main/n{version}/trajectories-{row['trajectories']:03d}"
            lines.append(
                f"| [{row['trajectories']}]({folder_url}) | {row['epochs']} | "
                f"[{row['best_epoch']}]({row['best_checkpoint']}) | {row['best_successes']}/20 |"
            )
        lines.extend(
            [
                "",
                "Every epoch is evaluated on 20 Spatial task-0 rollouts. Training stops after two "
                "consecutive epochs fail to strictly exceed the best success count. Ties count as "
                "no improvement. There is no fixed epoch cap.",
                "",
                f"Load checkpoints from `n{version}/trajectories-NNN/epoch-EEE/`. Each includes weights, "
                "configs, processors, Spatial normalization statistics, embodiment mappings, runtime code, "
                "training manifests, evaluation results, and a verification receipt. No optimizer state is included.",
                "",
                "Every epoch checkpoint was downloaded into a fresh directory, checked against its "
                "artifact hashes, loaded strictly in an offline process with empty Hub caches, and used "
                "for action inference. Its local copies were then deleted before the next epoch.",
                "",
                f"See [campaign_summary.json](https://huggingface.co/{repo}/blob/main/n{version}/campaign_summary.json) "
                "for pinned base and dataset revisions and the complete recipe. Pre-experiment files "
                "and their history have been purged. Original verification receipts retain the revisions "
                "used at verification time; [history_cleanup.json](history_cleanup.json) maps those "
                "checkpoints to the preserved snapshot with identical artifact hashes.",
            ]
        )
        api.upload_file(
            repo_id=repo,
            path_in_repo="README.md",
            path_or_fileobj=("\n".join(lines) + "\n").encode(),
            commit_message="Index all five verified trajectory budgets",
        )
    assert len(rows) == 20
    with (OUTPUT / "results.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        OUTPUT / "final_audit.json",
        {
            "complete": True,
            "runs_verified": len(rows),
            "epochs_verified": verified_epochs,
            "all_nested_subsets_identical": True,
            "all_stops_follow_patience": True,
            "all_uploaded_weight_hashes_match": True,
            "all_epoch_copies_deleted": True,
            "previously_completed_runs_unchanged": True,
            "previously_completed_runs_verified": len(completed_original_runs),
            "legacy_files_and_history_removed": True,
            "runs": rows,
        },
    )
    print(f"Verified {len(rows)} runs and {verified_epochs} epoch checkpoints")


if __name__ == "__main__":
    main()
