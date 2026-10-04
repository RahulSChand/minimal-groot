# GR00T LIBERO checkpoint evaluation runbook

This is the reusable procedure for evaluating GR00T checkpoints on new GPU machines.
Checkpoint-to-GPU assignment is intentionally kept separate because it changes with the
machine, repository contents, and number of GPUs.

## Required evaluation protocol

Unless a run explicitly specifies otherwise:

- Evaluate 40 rollouts per checkpoint.
- Evaluate task 0 using initial states 0 through 39.
- Use seed 7.
- Use 40 parallel simulator workers when the machine has enough CPU and RAM.
- Cap model inference batches at 8 with a 10 ms batching window.
- Use five-step replanning and ten initial settling steps.
- Render and send 256 px source images.
- Do not save videos during checkpoint sweeps.
- Use the suite-specific policy-step limit from `post_train_vla.evaluator.MAX_STEPS`.

Treat a checkpoint result as complete only when it contains exactly 40 unique episodes,
has no rollout errors, and its summary agrees with the episode records. A 20-rollout
result is not interchangeable with this protocol and must not be resumed or aggregated
as a 40-rollout result.

## Reusable machine setup

1. Read `/etc/vast-agents-guide.md` before changing a Vast instance.
2. Check network throughput, at least 100 GB free disk, both GPU visibility, GPU memory
   use, GPU utilization, and hidden compute-process holders.
3. Bootstrap credentials and repositories according to `.config/start_prompt.txt`.
4. Use HTTPS GitHub authentication. Persist GitHub, Hugging Face, and W&B credentials
   without printing tokens.
5. Clone `post_train_vla`, `openpi_easy`, and `minimal-groot`. `openpi_easy` is a source
   reference only and does not need its own environment.
6. For multiversion GR00T work, check out
   `feat/gr00t-multiversion-libero-training` in `minimal-groot`.
7. Install the locked `post_train_vla` environment and the locked minimal-groot
   performance/evaluation environment.
8. Install `libegl1-mesa-dev` and `libglu1-mesa` when absent. Do not install or replace
   NVIDIA drivers on a Vast image.
9. Run `examples/LIBERO/setup_reference_eval.sh`. Require its headless LIBERO smoke test
   to pass before downloading checkpoint weights.
10. Verify GitHub pull access, `hf auth whoami`, `wandb login --verify`, CUDA imports,
    and that all intended GPUs remain free.

The PaliGemma tokenizer is required for PI0/PI0.5 work, but it is not part of GR00T
checkpoint inference.

## Checkpoint compatibility checks

Perform these checks before launching a sweep:

1. Download only a representative checkpoint's small metadata first: `runtime.json`,
   `config.json`, `processor_config.json`, `artifact_checksums.json`, `run_config.json`,
   and `epoch.json`.
2. Compare `runtime.json.commit` with the checked-out minimal-groot commit. Do not assume
   that a repository and the current branch head match merely because their names are
   related.
3. Confirm the checkpoint's suite, model generation, model action horizon, processor
   decoded-action horizon, embodiment mapping, and normalization statistics.
4. Confirm that all referenced runtime files match before using an archival evaluator.
5. Run one checkpoint end to end and verify real GPU utilization plus at least one valid
   model action before starting the full sweep.

For `Chand0320/groot-libero-goal-object-trajectory-efficiency`, the inspected checkpoint
runtime is minimal-groot commit `8c8e1ebfd75f8140d775df34b6714e9153932e12`, matching
the multiversion branch at the time this runbook was written. Recheck this on every new
machine and repository revision.

## Correct model and policy path

For checkpoints produced by the current multiversion branch:

1. Import `gr00t.model` so the version-specific Hugging Face model and processor classes
   are registered.
2. Load the local checkpoint with `AutoModel.from_pretrained(...,
   output_loading_info=True, local_files_only=True)`.
3. Reject missing, unexpected, mismatched, or loading-error keys.
4. Load the processor with `AutoProcessor.from_pretrained(...,
   local_files_only=True)`.
5. Put both in evaluation mode, disable gradients, and move the model to its assigned
   CUDA device.
6. Serve the model through `gr00t.eval.reference_libero.ReferenceLiberoPolicy` and
   `post_train_vla.policy_server.PolicyServer`.
7. Seed the model runtime once with `gr00t.utils.determinism.seed_everything(7)` when
   using the standard parallel evaluator.
8. Launch `post_train_vla.eval_libero` from the pinned simulator environment with the
   40-rollout protocol above.

The standard `post_train_vla.eval_libero` client does not attach the private
`_evaluation_noise_seed` request field. Do not wrap its policy with code that requires
that field. Exact per-episode/per-inference noise seeds require the campaign evaluator's
context-aware client on both sides; they cannot be added only at the model server.

## Disk-bounded checkpoint streaming

Do not clone or download an entire checkpoint repository when it is larger than local
disk. The Goal/Object repository was approximately 213 GB while the evaluated machine
had a 128 GB filesystem.

Use this sequence for each assigned checkpoint:

1. Download the checkpoint directory into a staging area pinned to the repository
   revision discovered at queue start.
2. Verify the download and load weights strictly.
3. Run the complete 40-rollout evaluation into a fresh result directory.
4. Validate the summary and raw episode records.
5. Preserve metadata, logs, summaries, and episode records.
6. Only after successful validation, delete that checkpoint's downloaded
   `model*.safetensors` shards. Hugging Face remains the source of truth.
7. Continue to the next checkpoint. On any failure, stop the affected queue and retain
   the checkpoint weights for diagnosis.

Use Supervisor for long queues so they survive SSH disconnections. Write explicit queue
status files containing the GPU, current checkpoint, state, completed checkpoints, and
last update time. Completed checkpoints must be skipped safely when a queue restarts.

## Machine-specific choices

Decide these separately for every machine and run:

- Which suites, trajectory budgets, and epochs are in scope.
- How checkpoints are divided across GPUs.
- CUDA device IDs and unique policy-server ports.
- Simulator workers per GPU, based on CPU and RAM capacity.
- Whether a repository is still receiving new checkpoints. Pinning a revision gives a
  reproducible finite queue; discovering later uploads requires a deliberate new scan.
- Staging and result directories.

Never bake a particular even/odd split, Goal/Object split, or number of GPUs into the
reusable evaluator. Queue construction should provide these assignments as data.

## Known bad paths and previous failures

- Do not use `post_train_vla.groot_evaluation.SavedPolicy` for the current N1.5
  multiversion checkpoint format. That helper still imports `gr00t.data.schema`, which
  is absent from the matching branch.
- Do not assume `scripts/evaluate_saved_groot.py` accepts every published checkpoint
  family. Its strict archival format may require `load_fixture.npz`, exact runtime
  snapshots, and model/processor horizon assumptions not present in older repositories.
- Do not bypass runtime-integrity failures. Switch to the checkpoint-pinned runtime or
  use the model-generation-native loader that the matching branch provides.
- Do not run every rollout in a single simulator process when throughput matters. Use
  the parallel evaluator and verify that GPU inference batches are actually formed.
- Do not begin all queues before one representative checkpoint has loaded, produced a
  valid action, and started real simulator steps.

## Handoff checklist

Before declaring a sweep started, report:

- Repository ID and pinned revision.
- minimal-groot branch and commit.
- Suites/checkpoints in scope.
- Rollouts, task IDs, initial-state range, seed, workers, batch size, and replanning.
- GPU-to-queue assignment for this machine.
- First checkpoint's strict-load result.
- First valid completed checkpoint result, including rollout-error count.
- Supervisor process names, result directory, disk remaining, and resume behavior.

