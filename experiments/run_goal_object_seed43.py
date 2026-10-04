#!/usr/bin/env python3
"""Run N1.5/N1.6/N1.7 LIBERO Goal/Object trajectory experiments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


ROOT = Path("/workspace/minimal-groot")
TRAIN_SCRIPT = ROOT / "experiments/finetune_groot.py"
PYTHON = ROOT / ".venv/bin/python"
OUTPUT_ROOT = ROOT / "outputs/groot-goal-object-seed43"
HF_REPO = "Chand0320/groot-libero-goal-object-trajectory-efficiency"
VERSIONS = {
    "n1d5": ROOT / "checkpoints/GR00T-N1.5-3B",
    "n1d6": ROOT / "checkpoints/GR00T-N1.6-3B",
    "n1d7": ROOT / "checkpoints/GR00T-N1.7-3B",
}
SUITES = ("goal", "object")
BUDGETS = (10, 15, 25, 50)
SEED = 43
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


def dataset_path(suite: str, count: int) -> Path:
    return ROOT / "datasets" / f"libero_{suite}_seed{SEED}_trajectories_{count:03d}"


def output_path(version: str, suite: str, count: int) -> Path:
    return OUTPUT_ROOT / version / suite / f"trajectories-{count:03d}"


def validate_inputs() -> None:
    if not TRAIN_SCRIPT.is_file():
        raise FileNotFoundError(TRAIN_SCRIPT)
    for version, checkpoint in VERSIONS.items():
        if not (checkpoint / "config.json").is_file():
            raise FileNotFoundError(f"Missing {version} checkpoint: {checkpoint}")
    for suite in SUITES:
        for count in BUDGETS:
            dataset = dataset_path(suite, count)
            manifest_path = dataset / "trajectory_manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(manifest_path)
            manifest = json.loads(manifest_path.read_text())
            expected = {
                "suite": f"libero_{suite}",
                "trajectory_count": count,
                "seed": SEED,
            }
            actual = {key: manifest.get(key) for key in expected}
            if actual != expected:
                raise RuntimeError(f"Manifest mismatch at {manifest_path}: {actual} != {expected}")


def training_command(version: str, suite: str, count: int, output: Path, resume: dict | None = None) -> list[str]:
    command = [
        str(PYTHON),
        str(TRAIN_SCRIPT),
        "--version", version,
        "--checkpoint", str(VERSIONS[version]),
        "--dataset", str(dataset_path(suite, count)),
        "--output", str(output),
        "--epochs", str(EPOCHS),
        "--batch-size", str(BATCH_SIZE),
        "--accumulation", str(ACCUMULATION),
        "--learning-rate", str(LEARNING_RATE),
        "--seed", str(SEED),
    ]
    if resume is not None:
        command.extend([
            "--resume-weights", resume["resume_weights"],
            "--start-epoch", str(resume["start_epoch"]),
            "--start-step", str(resume["start_step"]),
        ])
    return command


def publish_checkpoint(version: str, suite: str, count: int, output: Path, epoch: int) -> bool | None:
    checkpoint = output / f"epoch-{epoch:03d}"
    receipt = output / "publication_receipts" / f"epoch-{epoch:03d}.json"
    if receipt.is_file() or not (checkpoint / "epoch.json").is_file():
        return True if receipt.is_file() else None
    prefix = (
        f"{version}/{suite}/seed-{SEED:03d}/"
        f"trajectories-{count:03d}/epoch-{epoch:03d}"
    )
    try:
        subprocess.run(
            [
                str(PYTHON), "-m", "gr00t.experiment.minimal_checkpoint_publication",
                str(checkpoint), "--repo", HF_REPO, "--prefix", prefix,
            ],
            check=True,
        )
    except subprocess.CalledProcessError as error:
        write_json(output / 'publication_retry.json', {
            'prefix': prefix, 'return_code': error.returncode, 'retryable_by_campaign': True,
            'recorded_at': time.time(),
        })
        print(f"publication_failed_will_retry={prefix} return_code={error.returncode}", flush=True)
        return False
    publication = json.loads((checkpoint / "publication.json").read_text())
    write_json(receipt, publication)
    shutil.rmtree(checkpoint)
    print(f"published_and_removed_local={prefix}", flush=True)
    return True


def run_one(version: str, suite: str, count: int) -> None:
    global active_child
    output = output_path(version, suite, count)
    status_path = output / "status.json"
    resume = None
    if status_path.is_file():
        status = json.loads(status_path.read_text())
        all_uploaded = all(
            (output / "publication_receipts" / f"epoch-{epoch:03d}.json").is_file()
            for epoch in range(1, EPOCHS + 1)
        )
        if status.get("status") == "complete" and all_uploaded:
            print(f"already_complete={version}/{suite}/trajectories-{count:03d}", flush=True)
            return
        resume_path = output / 'resume_plan.json'
        if not resume_path.is_file():
            raise RuntimeError(f"Incomplete run requires an explicit resume plan: {output}")
        resume = json.loads(resume_path.read_text())
        expected = {'version': version, 'suite': suite, 'trajectory_count': count}
        actual = {key: resume.get(key) for key in expected}
        if actual != expected:
            raise RuntimeError(f"Resume plan mismatch: {actual} != {expected}")
        resume_weights = Path(resume['resume_weights'])
        if not (resume_weights / 'epoch.json').is_file():
            raise FileNotFoundError(f"Resume weights are missing: {resume_weights}")
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Refusing to use nonempty output directory: {output}")

    print(f"starting={version}/{suite}/trajectories-{count:03d}", flush=True)
    resume_marker = output / 'resume_loaded.json'
    if resume is not None and resume_marker.exists():
        resume_marker.unlink()
    active_child = subprocess.Popen(training_command(version, suite, count, output, resume))
    try:
        if resume is not None:
            while active_child.poll() is None and not resume_marker.is_file():
                time.sleep(1)
            if not resume_marker.is_file():
                raise RuntimeError('Continuation exited before loading its resume weights')
        while active_child.poll() is None:
            publication_failed = False
            for epoch in range(1, EPOCHS + 1):
                if publish_checkpoint(version, suite, count, output, epoch) is False:
                    publication_failed = True
            time.sleep(60 if publication_failed else 10)
        return_code = active_child.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, active_child.args)
        status = json.loads(status_path.read_text())
        if status.get("status") != "complete":
            raise RuntimeError(f"Training exited without complete status: {status}")
        while True:
            pending = []
            for epoch in range(1, EPOCHS + 1):
                result = publish_checkpoint(version, suite, count, output, epoch)
                if result is not True:
                    pending.append(epoch)
            if not pending:
                break
            print(f"waiting_for_checkpoint_uploads={pending}", flush=True)
            time.sleep(60)
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    validate_inputs()
    if args.validate_only:
        print("campaign_inputs_valid=true", flush=True)
        return
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(
        OUTPUT_ROOT / "campaign.json",
        {
            "versions": list(VERSIONS),
            "suites": list(SUITES),
            "budgets": list(BUDGETS),
            "epochs": EPOCHS,
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation": ACCUMULATION,
            "learning_rate": LEARNING_RATE,
            "hf_repo": HF_REPO,
            "checkpoint_type": "model_and_processor_only_no_optimizer_or_scheduler",
        },
    )
    for version in VERSIONS:
        for suite in SUITES:
            for count in BUDGETS:
                run_one(version, suite, count)
    print("campaign_complete=true", flush=True)


if __name__ == "__main__":
    main()
