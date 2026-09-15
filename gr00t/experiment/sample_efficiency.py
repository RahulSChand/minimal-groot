"""Independent GR00T fine-tunes following post_train_vla's trajectory protocol."""

import argparse
import gc
import hashlib
import itertools
import json
import math
import random
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from gr00t.configs.base_config import Config
from gr00t.configs.data.libero_spatial import libero_spatial_config
from gr00t.data.dataset.trajectory_subset import (
    TrajectorySubsetDataset,
    create_or_validate_manifest,
)
from gr00t.data.types import EmbodimentTag
from gr00t.experiment.launch_finetune import select_model_config


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, parameters, optimizer, loader, accumulation, starting_step):
    """Match the reference's shorter final accumulation group and grad clipping."""
    model.train()
    total_loss, steps = 0.0, 0
    iterator = iter(loader)
    for start in range(0, len(loader), accumulation):
        group_size = min(accumulation, len(loader) - start)
        optimizer.zero_grad(set_to_none=True)
        group_loss = 0.0
        for _ in range(group_size):
            batch = next(iterator)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                loss = model(**batch)["loss"].mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at step {starting_step + steps + 1}")
            group_loss += loss.detach().float().item() / group_size
            (loss / group_size).backward()
            del batch, loss
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        steps += 1
        total_loss += group_loss
        print(
            f"step={starting_step + steps} loss={group_loss:.6f} grad_norm={float(norm):.6f}",
            flush=True,
        )
    optimizer.zero_grad(set_to_none=True)
    return starting_step + steps, total_loss / steps, steps


def stopping_state(previous, epoch, successes):
    if previous is None or successes > previous["best_successes"]:
        return {"best_epoch": epoch, "best_successes": successes, "epochs_without_improvement": 0}
    return {**previous, "epochs_without_improvement": previous["epochs_without_improvement"] + 1}


def should_stop(state, epoch, *, patience, minimum_epochs=1, max_epochs=None):
    return (epoch >= minimum_epochs and state["epochs_without_improvement"] >= patience) or (
        max_epochs is not None and epoch >= max_epochs
    )


def is_plan_extension(previous, proposed):
    """Allow additional budgets only when every existing run and recipe match."""
    # New runs record the current source commit. Existing checkpoint manifests
    # and their checksummed runtime source remain unchanged after a code commit.
    metadata_fields = {"runs", "source_git_commit"}
    previous_recipe = {key: value for key, value in previous.items() if key not in metadata_fields}
    proposed_recipe = {key: value for key, value in proposed.items() if key not in metadata_fields}
    if previous_recipe != proposed_recipe:
        return False
    proposed_runs = {run["trajectory_count"]: run for run in proposed["runs"]}
    return all(proposed_runs.get(run["trajectory_count"]) == run for run in previous["runs"])


def build_model_and_processor(args, run_dir):
    from gr00t.model import MODEL_REGISTRY

    config = Config()
    config.model = select_model_config(args.base_model_path, args.model_version)
    for name in ("tune_llm", "tune_visual", "tune_projector", "tune_diffusion_model"):
        setattr(config.model, name, True)
    config.model.state_dropout_prob = 0.0
    config.training.start_from_checkpoint = args.base_model_path
    config.training.transformers_trust_remote_code = True
    artifact_dir = run_dir / "experiment_cfg"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    pipeline = MODEL_REGISTRY[type(config.model)](config, artifact_dir)
    model = pipeline._create_model()
    if model.config.model_type == "Gr00tN1d7":
        model.config.backbone_config = model.backbone.model.config.to_dict()
    config.model.action_horizon = model.config.action_horizon
    modalities = libero_spatial_config(model.config.action_horizon)
    config.data.modality_configs = {EmbodimentTag.LIBERO_PANDA.value: modalities}
    processor = pipeline._create_processor()
    model.to("cuda")
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    config.training.global_batch_size = args.batch_size
    config.training.gradient_accumulation_steps = args.gradient_accumulation_steps
    config.training.learning_rate = args.learning_rate
    config.training.weight_decay = 0.01
    config.training.weight_decay_all_parameters = True
    config.training.lr_scheduler_type = "constant"
    config.training.warmup_ratio = 0.0
    config.data.seed = args.seed
    config.save(artifact_dir / "config.yaml")
    return model, processor, modalities


def run_budget(args, budget, manifest, plan):
    from gr00t.eval.reference_libero import evaluate_live_model

    run_dir = args.output_dir / f"trajectories-{budget:03d}"
    summary_path = run_dir / "run_summary.json"
    if summary_path.exists():
        previous = json.loads(summary_path.read_text())
        if previous.get("complete"):
            print(f"Skipping completed budget {budget}: {summary_path}", flush=True)
            return previous
        raise RuntimeError(
            f"Incomplete run at {run_dir}. Use a new output directory to restart this budget from base weights."
        )
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"Refusing to mix a new run with existing artifacts: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    selected = manifest["ordered_episode_indices"][:budget]
    write_json(run_dir / "trajectory_manifest.json", manifest)
    seed_everything(args.seed)
    model, processor, modalities = build_model_and_processor(args, run_dir)
    dataset = TrajectorySubsetDataset(args.dataset_root, selected, modalities, processor)
    processor.state_action_processor.statistics.clear()
    processor.state_action_processor.norm_params.clear()
    processor.set_statistics({EmbodimentTag.LIBERO_PANDA.value: dataset.get_dataset_statistics()}, override=True)
    processor.train()
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=processor.collator,
        generator=torch.Generator().manual_seed(args.seed),
    )
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
    summary = {
        "complete": False,
        "model_version": args.model_version,
        "trajectory_count": budget,
        "selected_episode_indices": selected,
        "frames": len(dataset),
        "trajectory_manifest_sha256": plan["trajectory_manifest_sha256"],
        "training": plan["training"],
        "evaluation": plan["evaluation"],
        "epochs": [],
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in parameters),
        "action_horizon": model.config.action_horizon,
    }
    write_json(summary_path, summary)
    print(
        f"budget={budget} frames={len(dataset)} steps_per_epoch={math.ceil(len(loader) / args.gradient_accumulation_steps)} "
        f"trainable_parameters={summary['trainable_parameters']} effective_batch={args.batch_size * args.gradient_accumulation_steps}",
        flush=True,
    )
    step, state = 0, None
    try:
        for epoch in itertools.count(1):
            started = time.monotonic()
            processor.train()
            step, loss, steps = train_one_epoch(
                model, parameters, optimizer, loader, args.gradient_accumulation_steps, step
            )
            checkpoint = run_dir / f"epoch-{epoch:03d}"
            temporary = run_dir / f".epoch-{epoch:03d}.tmp"
            # Save standalone model/processor artifacts before simulation. Failed
            # evaluations leave this trained checkpoint available for inspection.
            model.save_pretrained(temporary, max_shard_size="5GB")
            processor.save_pretrained(temporary)
            from gr00t.experiment.checkpoint_publication import add_checkpoint_assets

            add_checkpoint_assets(temporary, processor, dataset, plan, Path(__file__).resolve().parents[2])
            shutil.copytree(run_dir / "experiment_cfg", temporary / "experiment_cfg")
            shutil.copy2(args.manifest, temporary / "trajectory_manifest.json")
            write_json(
                temporary / "metadata.json",
                {**summary, "epoch": epoch, "global_step": step, "optimizer_saved": False},
            )
            temporary.rename(checkpoint)
            gc.collect()
            torch.cuda.empty_cache()
            evaluation = evaluate_live_model(
                model,
                processor,
                reference_project=args.reference_project,
                eval_python=args.eval_python,
                output_dir=checkpoint / "evaluation",
                port=args.eval_port,
                workers=args.eval_workers,
                max_batch_size=args.eval_max_batch_size,
                episodes=args.eval_episodes,
                seed=7,
                timeout=args.eval_timeout,
            )
            successes = int(evaluation["successes"])
            state = stopping_state(state, epoch, successes)
            record = {
                "epoch": epoch,
                "global_step": step,
                "optimizer_steps": steps,
                "train_loss": loss,
                "successes": successes,
                "episodes": evaluation["episodes"],
                "success_rate": evaluation["success_rate"],
                "elapsed_seconds": time.monotonic() - started,
            }
            summary["epochs"].append(record)
            summary.update(state)
            summary["best_checkpoint"] = str(run_dir / f"epoch-{state['best_epoch']:03d}")
            keep_last = args.checkpoint_retention == "best-and-last" or state["best_epoch"] == epoch
            summary["last_checkpoint"] = str(checkpoint) if keep_last else None
            finished = should_stop(
                state, epoch, patience=args.patience, minimum_epochs=args.minimum_epochs, max_epochs=args.max_epochs
            )
            summary["stop_reason"] = (
                ("max_epochs" if epoch == args.max_epochs else "early_stopping") if finished else None
            )
            summary["complete"] = False  # Completion includes remote verification and local cleanup.
            write_json(
                checkpoint / "metadata.json",
                {**summary, "epoch": epoch, "global_step": step, "optimizer_saved": False},
            )
            if args.hub_repo_id:
                import importlib

                from gr00t.experiment import checkpoint_publication

                folder = f"{args.hub_version_folder}/trajectories-{budget:03d}/epoch-{epoch:03d}"
                while True:
                    try:
                        receipt = checkpoint_publication.publish_verify_delete(
                            checkpoint,
                            args.hub_repo_id,
                            folder,
                            run_dir / "evaluations" / checkpoint.name,
                        )
                        break
                    except Exception as error:
                        # Preserve the live optimizer while an upload/load
                        # failure is repaired. No next epoch can start here.
                        pending = run_dir / "pending_publication.json"
                        retry = run_dir / "retry_publication"
                        write_json(pending, {"epoch": epoch, "error": repr(error), "retry_file": str(retry)})
                        print(f"Publication paused: {error!r}; waiting for {retry}", flush=True)
                        while not retry.exists():
                            time.sleep(5)
                        retry.unlink()
                        pending.unlink()
                        importlib.reload(checkpoint_publication)
                record["upload"] = receipt
                summary["best_checkpoint"] = (
                    f"https://huggingface.co/{args.hub_repo_id}/tree/main/{args.hub_version_folder}/"
                    f"trajectories-{budget:03d}/epoch-{state['best_epoch']:03d}"
                )
                summary["last_checkpoint"] = f"https://huggingface.co/{args.hub_repo_id}/tree/main/{folder}"
                summary["complete"] = finished
                write_json(summary_path, summary)
                print(
                    f"budget={budget} epoch={epoch} loss={loss:.6f} success={successes}/{evaluation['episodes']} "
                    f"best_epoch={state['best_epoch']} patience={state['epochs_without_improvement']} complete={finished}",
                    flush=True,
                )
                if finished:
                    from huggingface_hub import HfApi

                    HfApi().upload_file(
                        repo_id=args.hub_repo_id,
                        path_or_fileobj=str(summary_path),
                        path_in_repo=f"{args.hub_version_folder}/trajectories-{budget:03d}/run_summary.json",
                        commit_message=f"Completed {args.hub_version_folder}, {budget} Spatial trajectories",
                    )
                    break
                continue
            summary["complete"] = finished
            write_json(summary_path, summary)
            # Preserve every epoch's metrics. Only remove checkpoints created
            # by this invocation, after the evaluated best checkpoint is saved.
            retained_epochs = {state["best_epoch"]}
            if args.checkpoint_retention == "best-and-last":
                retained_epochs.add(epoch)
            for old in summary["epochs"]:
                old_dir = run_dir / f"epoch-{old['epoch']:03d}"
                if old_dir.exists() and old["epoch"] not in retained_epochs:
                    archive = run_dir / "evaluations" / old_dir.name
                    archive.parent.mkdir(exist_ok=True)
                    shutil.copytree(old_dir / "evaluation", archive, dirs_exist_ok=True)
                    shutil.copy2(old_dir / "metadata.json", archive / "metadata.json")
                    shutil.rmtree(old_dir)
            print(
                f"budget={budget} epoch={epoch} loss={loss:.6f} success={successes}/{evaluation['episodes']} "
                f"best_epoch={state['best_epoch']} complete={summary['complete']}",
                flush=True,
            )
            if summary["complete"]:
                break
        if not args.hub_repo_id:
            best_metadata_path = Path(summary["best_checkpoint"]) / "metadata.json"
            best_metadata = json.loads(best_metadata_path.read_text())
            best_metadata.update(complete=True, epochs_run=len(summary["epochs"]), run_summary_file=str(summary_path))
            write_json(best_metadata_path, best_metadata)
        return summary
    finally:
        del optimizer, parameters, model, loader, dataset, processor
        gc.collect()
        torch.cuda.empty_cache()


def make_plan(args, manifest):
    records = {item["episode_index"]: item for item in manifest["episodes"]}
    return {
        "base_model_path": args.base_model_path,
        "base_model_repo": args.base_model_repo,
        "base_model_revision": args.base_model_revision,
        "model_version": args.model_version,
        "dataset_root": str(args.dataset_root),
        "trajectory_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "reference_project": str(args.reference_project),
        "checkpoint_retention": args.checkpoint_retention,
        "hub_repo_id": args.hub_repo_id,
        "hub_version_folder": args.hub_version_folder,
        "source_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dataset_provenance": json.loads((args.dataset_root / "meta/spatial_verification.json").read_text())
        if (args.dataset_root / "meta/spatial_verification.json").exists()
        else None,
        "training": {
            "seed": args.seed,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "lr_scheduler": "constant",
            "warmup": 0,
            "optimizer": "AdamW",
            "weight_decay": 0.01,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "max_grad_norm": 1.0,
            "full_model_finetune": True,
            "gradient_checkpointing": args.gradient_checkpointing,
            "minimum_epochs": args.minimum_epochs,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "num_workers": args.num_workers,
            "normalization": "shared_dataset_metadata",
            "precision": "bf16_autocast",
            "optimizer_saved": False,
        },
        "evaluation": {
            "suite": "libero_spatial",
            "task_id": 0,
            "episodes": args.eval_episodes,
            "seed": 7,
            "initial_states": list(range(args.eval_episodes)),
            "replan_steps": 5,
            "wait_steps": 10,
            "max_episode_steps": 220,
            "workers": args.eval_workers,
            "max_batch_size": args.eval_max_batch_size,
            "render_resolution": 256,
            "resize": "GR00T processor",
        },
        "runs": [
            {
                "trajectory_count": budget,
                "episode_indices": manifest["ordered_episode_indices"][:budget],
                "frames": sum(records[i]["length"] for i in manifest["ordered_episode_indices"][:budget]),
                "task_counts": dict(
                    Counter(str(records[i]["task_index"]) for i in manifest["ordered_episode_indices"][:budget])
                ),
            }
            for budget in args.trajectory_budgets
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--model-version", default="auto")
    parser.add_argument("--dataset-root", type=Path, default=Path("/root/libero_spatial_post"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--trajectory-budgets", type=int, nargs="+", default=[5, 10, 15, 25, 50])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--minimum-epochs", type=int, default=3)
    parser.add_argument("--max-epochs", type=int, default=None, help="Optional cap; default: patience only")
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--reference-project", type=Path, default=Path("/root/post_train_vla"))
    parser.add_argument("--eval-python", type=Path, default=Path("gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python"))
    parser.add_argument("--eval-workers", type=int, default=20)
    parser.add_argument("--eval-max-batch-size", type=int, default=8)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-port", type=int, default=8765)
    parser.add_argument("--eval-timeout", type=int, default=1800)
    parser.add_argument("--checkpoint-retention", choices=["best", "best-and-last"], default="best")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--extend-run-plan", action="store_true", help="Add budgets while preserving the existing recipe"
    )
    parser.add_argument("--hub-repo-id")
    parser.add_argument("--hub-version-folder")
    parser.add_argument("--base-model-repo")
    parser.add_argument("--base-model-revision")
    args = parser.parse_args()
    if (
        args.minimum_epochs < 1
        or args.patience < 1
        or (args.max_epochs is not None and args.minimum_epochs > args.max_epochs)
    ):
        parser.error("Require positive minimum_epochs/patience and max_epochs >= minimum_epochs if supplied")
    if args.hub_repo_id and not (args.hub_version_folder and args.base_model_repo and args.base_model_revision):
        parser.error("Hub publishing requires version folder and pinned base model provenance")
    if args.hub_repo_id and not (args.dataset_root / "meta/spatial_verification.json").is_file():
        parser.error("Hub publishing requires verified Spatial dataset provenance")
    if (
        min(
            [
                *args.trajectory_budgets,
                args.batch_size,
                args.gradient_accumulation_steps,
                args.eval_workers,
                args.eval_max_batch_size,
                args.eval_episodes,
            ]
        )
        < 1
    ):
        parser.error("Budgets and batch/evaluation sizes must be positive")
    if len(args.trajectory_budgets) != len(set(args.trajectory_budgets)):
        parser.error("Trajectory budgets must be unique")
    for key in ("dataset_root", "output_dir", "reference_project"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    # Do not resolve the virtualenv Python symlink to its system executable.
    args.eval_python = args.eval_python.expanduser().absolute()
    args.manifest = (
        (args.manifest or args.output_dir / f"trajectory_manifest_seed{args.seed}.json").expanduser().resolve()
    )
    manifest = create_or_validate_manifest(args.dataset_root, args.manifest, args.seed)
    if max(args.trajectory_budgets) > manifest["total_episodes"]:
        parser.error("Trajectory budget exceeds the number of episodes")
    model_config = select_model_config(args.base_model_path, args.model_version)
    args.model_version = {
        "gr00t_n1": "1",
        "gr00t_n1_5": "1.5",
        "Gr00tN1d6": "1.6",
        "Gr00tN1d7": "1.7",
    }[model_config.model_type]
    plan = make_plan(args, manifest)
    plan_path = args.output_dir / "run_plan.json"
    if plan_path.exists():
        previous_plan = json.loads(plan_path.read_text())
        if previous_plan != plan:
            if not args.extend_run_plan or not is_plan_extension(previous_plan, plan):
                raise ValueError(f"Existing run plan differs; use a new output directory: {plan_path}")
            previous_hash = hashlib.sha256(plan_path.read_bytes()).hexdigest()[:12]
            shutil.copy2(plan_path, args.output_dir / f"run_plan.before-extension-{previous_hash}.json")
    write_json(plan_path, plan)
    print(json.dumps(plan, indent=2), flush=True)
    if args.prepare_only:
        return
    if not args.eval_python.is_file():
        raise FileNotFoundError(f"Missing LIBERO environment: {args.eval_python}")
    if not (args.reference_project / "src/post_train_vla/policy_server.py").is_file():
        raise FileNotFoundError(f"Missing reference evaluator: {args.reference_project}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GR00T post-training")
    results = []
    for budget in args.trajectory_budgets:
        results.append(run_budget(args, budget, manifest, plan))
        write_json(
            args.output_dir / "run_summary.json",
            {
                "model_version": args.model_version,
                "complete": len(results) == len(args.trajectory_budgets),
                "runs": results,
            },
        )


if __name__ == "__main__":
    main()
