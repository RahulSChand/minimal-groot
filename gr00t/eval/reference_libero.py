"""Deprecated import compatibility; there is no separate reference evaluator."""

from gr00t.eval.libero_evaluation import evaluate_live_model  # noqa: F401


class ReferenceLiberoPolicy:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "ReferenceLiberoPolicy is retired. Use Gr00tPolicy + Gr00tSimPolicyWrapper "
            "with native LiberoEnv; run python -m gr00t.eval.evaluate_checkpoint. "
            "Do not convert gripper commands in the policy: LiberoEnv owns conversion."
        )
