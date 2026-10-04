# Agent instructions for saved GR00T LIBERO evaluations

Before setting up or running saved-checkpoint LIBERO benchmarks, read
[.config/groot_libero_eval_runbook.md](.config/groot_libero_eval_runbook.md)
completely. It is the authoritative execution/protocol guide for this workflow.

- Use the minimal-groot multiversion branch and one agreed Git SHA across machines.
- Use only `.venv/bin/python -m gr00t.eval.evaluate_checkpoint --compile ...`
  for the user's compiled bulk campaign. Eager requires an explicit diagnostic
  or user request. Do not omit the flag or silently fall back.
- Do not use ReferenceLiberoPolicy, reference_simulator, post_train_vla, checkpoint
  runtime snapshots, probe scripts, or hand-assembled server/rollout commands.
- Do not duplicate gripper conversion; native LiberoEnv owns it.
- Full benchmark: all ten tasks, 40 episodes per task, GPU EGL, no videos,
  and per-subtask SR. Follow the runbook's explicit worker/batch/seed settings.
- Run its completion gate before accepting or skipping a checkpoint result.
- Do not start a bulk queue without the user's assigned checkpoint list.
- Preserve unrelated work, failed artifacts, weights and credentials. Do not
  overwrite result directories or delete weights without an approved policy.

These rules are scoped to saved-checkpoint LIBERO benchmarks, not unrelated
training, application, or other-model workflows. Explicit user instructions
can request a different protocol; record and label that protocol separately.
