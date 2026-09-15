"""Run missing experiments from the twenty-run campaign, one process at a time."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from gr00t.experiment.checkpoint_publication import write_json

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/spatial-trajectory-efficiency-20260915"
BASES = ROOT / "official-bases"
BUDGETS = (5, 10, 15, 25, 50)
MODELS = [
    ("1", "GR00T-N1-2B", "fc879581ca32f4f6d6e02cf0cc80452f6b0c3873"),
    ("1.5", "GR00T-N1.5-3B", "869830fc749c35f34771aa5209f923ac57e4564e"),
    ("1.6", "GR00T-N1.6-3B", "d0814e7ecb19202e7c8468b46098b0b7ef3a6d61"),
    ("1.7", "GR00T-N1.7-3B", "2fc962b973bccdd5d8ce4f67cc63b264d6886495"),
]


def main():
    import fcntl

    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = (OUTPUT / "campaign.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES="0",
        PYTHONUNBUFFERED="1",
        TOKENIZERS_PARALLELISM="false",
        NO_ALBUMENTATIONS_UPDATE="1",
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="1",
        PYTHONHASHSEED="42",
    )
    status = {"complete": False, "pid": os.getpid(), "runs_requested": 20, "completed_versions": []}
    try:
        for version, name, revision in MODELS:
            repo = "nvidia/" + name
            base = BASES / name
            target = f"kaweees/gr00tn{version}-libero-spatial-trajectory-efficiency"
            version_dir = OUTPUT / f"n{version}"
            summary = version_dir / "run_summary.json"
            budget_summaries = [version_dir / f"trajectories-{budget:03d}" / "run_summary.json" for budget in BUDGETS]
            if all(path.exists() and json.loads(path.read_text()).get("complete") for path in budget_summaries):
                status["completed_versions"].append(version)
                continue
            status.update(current_version=version, phase="download_base")
            write_json(OUTPUT / "campaign_status.json", status)
            subprocess.run(
                ["hf", "download", repo, "--revision", revision, "--local-dir", str(base)],
                env=env,
                check=True,
            )
            subprocess.run(
                ["hf", "cache", "verify", repo, "--revision", revision, "--local-dir", str(base)],
                env=env,
                check=True,
            )
            status["phase"] = "training_evaluation_publication"
            write_json(OUTPUT / "campaign_status.json", status)
            with (OUTPUT / f"n{version}.log").open("a") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "gr00t.experiment.sample_efficiency",
                        "--base-model-path",
                        str(base),
                        "--model-version",
                        version,
                        "--base-model-repo",
                        repo,
                        "--base-model-revision",
                        revision,
                        "--dataset-root",
                        "/root/liber_spatial_post",
                        "--manifest",
                        str(OUTPUT / "trajectory_manifest_seed42.json"),
                        "--output-dir",
                        str(version_dir),
                        "--trajectory-budgets",
                        *map(str, BUDGETS),
                        "--extend-run-plan",
                        "--seed",
                        "42",
                        "--minimum-epochs",
                        "1",
                        "--patience",
                        "2",
                        "--hub-repo-id",
                        target,
                        "--hub-version-folder",
                        f"n{version}",
                        "--reference-project",
                        "/root/post_train_vla",
                        "--eval-python",
                        str(ROOT / "gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python"),
                    ],
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            if not json.loads(summary.read_text())["complete"]:
                raise RuntimeError(f"Version {version} exited without completing its five runs")
            status["completed_versions"].append(version)
            write_json(OUTPUT / "campaign_status.json", status)
            # These are downloaded official bases owned by this campaign.
            shutil.rmtree(base)
        status.update(complete=True, phase="complete", current_version=None)
        write_json(OUTPUT / "campaign_status.json", status)
    except BaseException as error:
        status.update(phase="failed", error=str(error))
        write_json(OUTPUT / "campaign_status.json", status)
        raise


if __name__ == "__main__":
    main()
