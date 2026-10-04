"""Regression coverage for the single native LIBERO evaluation path."""

import ast
import importlib
import json
import random
import signal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from gr00t.data.types import EmbodimentTag, ModalityConfig
from gr00t.eval.compile_policy import COMPILE_MODEL_TYPES, compile_action_transformer
from gr00t.eval.evaluate_checkpoint import (
    handle_termination,
    main,
    parse_args,
    run_rollouts,
    simulator_command,
    simulator_environment,
    validate_results,
    warmup_policy,
)
from gr00t.eval.libero_simulator import batch_observations
from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper


def arguments(tmp_path, *extra):
    return parse_args(
        [
            "--checkpoint",
            str(tmp_path / "ckpt"),
            "--suite",
            "libero_goal",
            "--output-dir",
            str(tmp_path / "out"),
            *extra,
        ]
    )


def test_defaults_and_compile_flag(tmp_path):
    args = arguments(tmp_path)
    assert not args.compile_model
    assert (args.episodes_per_task, args.workers, args.max_batch_size, args.seed) == (40, 4, 8, 7)
    assert args.task_ids is None
    assert arguments(tmp_path, "--compile").compile_model


@pytest.mark.parametrize(
    "extra",
    [
        ["--workers", "0"],
        ["--max-batch-size", "0"],
        ["--episodes-per-task", "-1"],
        ["--gpu", "-1"],
        ["--port", "65536"],
        ["--timeout", "0"],
        ["--task-id", "0", "--task-id", "0"],
        ["--reference-project", "/old/runner"],
    ],
)
def test_invalid_options(tmp_path, extra):
    with pytest.raises(SystemExit):
        arguments(tmp_path, *extra)


@pytest.mark.parametrize("model_type", COMPILE_MODEL_TYPES)
def test_existing_output_is_never_overwritten(tmp_path, model_type):
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({"model_type": model_type}))
    output = tmp_path / "out"
    output.mkdir()
    marker = output / "summary.json"
    marker.write_text("keep")
    import sys

    with pytest.raises(FileExistsError):
        main(
            [
                "--checkpoint",
                str(checkpoint),
                "--suite",
                "libero_goal",
                "--output-dir",
                str(output),
                "--eval-python",
                sys.executable,
                "--compile",
            ]
        )
    assert marker.read_text() == "keep"


def test_native_command_and_physical_gpu(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    args = arguments(tmp_path, "--gpu", "1", "--task-id", "7")
    env = simulator_environment(args)
    assert "CUDA_VISIBLE_DEVICES" not in env
    assert env["MUJOCO_EGL_DEVICE_ID"] == "1"
    assert env["MUJOCO_GL"] == env["PYOPENGL_PLATFORM"] == "egl"
    command = simulator_command(args)
    assert "gr00t.eval.libero_simulator" in command
    assert "post_train_vla" not in " ".join(command)
    assert "--save-video" not in command


@pytest.mark.parametrize("model_type", COMPILE_MODEL_TYPES)
def test_compile_only_native_transformer(monkeypatch, model_type):
    model = SimpleNamespace(
        config=SimpleNamespace(model_type=model_type),
        training=False,
        parameters=lambda: iter([SimpleNamespace(device=torch.device("cuda"))]),
        action_head=SimpleNamespace(model=SimpleNamespace(forward=Mock())),
    )
    original = model.action_head.model.forward
    compile_mock = Mock(return_value=Mock())
    monkeypatch.setattr(torch, "compile", compile_mock)
    metadata = compile_action_transformer(model)
    expected = {"backend": "cudagraphs"} if model_type == "Gr00tN1d7" else {"mode": "reduce-overhead"}
    compile_mock.assert_called_once_with(original, dynamic=False, fullgraph=True, **expected)
    assert metadata["target"] == "action_head.model.forward"


def native_fixture(model_type):
    keys = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    raw = np.arange(2 * 7 * 7, dtype=np.float32).reshape(2, 7, 7) / 100
    raw[..., -1] = [0, 0.25, 0.49, 0.5, 0.51, 0.75, 1]
    configs = {
        "video": ModalityConfig(delta_indices=[0], modality_keys=["image", "wrist_image"]),
        "state": ModalityConfig(delta_indices=[0], modality_keys=keys),
        "action": ModalityConfig(delta_indices=list(range(7)), modality_keys=keys),
        "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.action.task_description"]),
    }

    class Processor:
        collator = staticmethod(lambda inputs: {})
        get_modality_configs = staticmethod(lambda: {EmbodimentTag.LIBERO_PANDA.value: configs})

        def __call__(self, messages):
            return {}

        def decode_action(self, prediction, tag, states):
            return {key: raw[..., i : i + 1] for i, key in enumerate(keys)}

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))
            self.config = SimpleNamespace(model_type=model_type, action_horizon=7)

        def get_action(self, **kwargs):
            return {"action_pred": torch.zeros(2, 7, 7)}

    observation = {f"video.{key}": np.zeros((256, 256, 3), dtype=np.uint8) for key in ("image", "wrist_image")}
    observation.update({f"state.{k}": np.zeros(2 if k == "gripper" else 1, dtype=np.float32) for k in keys})
    observation["annotation.human.action.task_description"] = "test"
    return Model().eval(), Processor(), raw, observation


@pytest.mark.parametrize("model_type", COMPILE_MODEL_TYPES)
@pytest.mark.parametrize("compiled", [False, True])
def test_native_policy_never_converts_gripper(monkeypatch, model_type, compiled):
    model, processor, raw, observation = native_fixture(model_type)
    compiler = Mock(return_value={"backend": "test"})
    monkeypatch.setattr("gr00t.eval.compile_policy.compile_action_transformer", compiler)
    monkeypatch.setattr(torch.compiler, "cudagraph_mark_step_begin", lambda: None)
    policy = Gr00tPolicy.from_model(model, processor, compile_model=compiled)
    wrapper = Gr00tSimPolicyWrapper(policy)
    actions, info = wrapper.get_action(batch_observations([observation, observation]))
    result = np.concatenate([actions[f"action.{k}"] for k in policy.modality_configs["action"].modality_keys], axis=-1)
    np.testing.assert_array_equal(result, raw)
    assert compiler.call_count == int(compiled)
    assert info["batch_size"] == 2


def test_skipped_capture_fails(monkeypatch):
    from torch._dynamo.utils import counters

    model, processor, _, observation = native_fixture("Gr00tN1d6")
    monkeypatch.setattr("gr00t.eval.compile_policy.compile_action_transformer", lambda model: {})
    monkeypatch.setattr(torch.compiler, "cudagraph_mark_step_begin", lambda: None)
    original = model.get_action

    def skipped(**kwargs):
        counters["inductor"]["cudagraph_skips"] += 1
        return original(**kwargs)

    model.get_action = skipped
    policy = Gr00tSimPolicyWrapper(Gr00tPolicy.from_model(model, processor, compile_model=True))
    with pytest.raises(RuntimeError, match="refusing silent fallback"):
        policy.get_action(batch_observations([observation, observation]))


def test_native_environment_owns_exactly_one_gripper_conversion():
    # Isolate the real native methods without importing MuJoCo in the model env.
    source = Path(__file__).resolve().parents[3] / "gr00t/eval/sim/LIBERO/libero_env.py"
    tree = ast.parse(source.read_text())
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        and node.name in ("normalize_gripper_action", "invert_gripper_action", "LiberoEnv")
    ]
    namespace = {"np": np, "gym": SimpleNamespace(Env=object)}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), namespace)
    env = namespace["LiberoEnv"].__new__(namespace["LiberoEnv"])
    received = []
    env._env = SimpleNamespace(
        step=lambda action: (received.append(action.copy()) or {}, 0, False, {}), check_success=lambda: False
    )
    env._process_observation = lambda obs: obs
    keys = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
    for value, expected in ((0, 1), (0.49, 1), (0.5, 0), (0.51, -1), (1, -1)):
        action = {f"action.{k}": np.array([value if k == "gripper" else 0.125]) for k in keys}
        env.step(action)
        np.testing.assert_array_equal(received[-1][:6], np.full(6, 0.125))
        assert received[-1][-1] == expected
        assert action["action.gripper"][0] == value
    states = []
    env._env.reset = lambda: {}
    env._env.seed = lambda seed: None
    env._env.set_init_state = lambda state: states.append(state) or {}
    env.reset(seed=7, options={"initial_state": "fixed", "wait_steps": 10})
    assert states == ["fixed"] and len(received) == 15


@pytest.mark.parametrize("key", ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"])
def test_native_loader_rejects_weight_mismatch(monkeypatch, key):
    monkeypatch.setattr("gr00t.policy.gr00t_policy.AutoModel.from_pretrained", lambda *a, **k: (None, {key: ["bad"]}))
    with pytest.raises(RuntimeError, match="strict loading"):
        Gr00tPolicy(EmbodimentTag.LIBERO_PANDA, "/checkpoint", device="cpu")


def test_retired_reference_adapter_fails_clearly():
    from gr00t.eval.reference_libero import ReferenceLiberoPolicy

    with pytest.raises(RuntimeError, match="retired"):
        ReferenceLiberoPolicy(None, None)


def test_warmup_preserves_rng_and_covers_shapes(monkeypatch):
    fork = torch.random.fork_rng
    monkeypatch.setattr(torch.random, "fork_rng", lambda devices: fork(devices=[]))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    calls = []

    def infer(observation):
        calls.append((observation["annotation.human.action.task_description"][0], len(observation["video.image"])))
        random.random()
        np.random.random()
        torch.rand(1)

    state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    manifest = {"tasks": [{"task_id": 5, "prompt": "plate"}, {"task_id": 7, "prompt": "stove"}]}
    assert len(warmup_policy(SimpleNamespace(get_action=infer), manifest, 4)) == 8
    for task in ("plate", "stove"):
        for batch in range(1, 5):
            assert calls.count((task, batch)) == 3
    assert random.getstate() == state[0]
    np.testing.assert_equal(np.random.get_state(), state[1])
    assert torch.equal(torch.get_rng_state(), state[2])


def test_sigterm_unwinds_for_cleanup():
    with pytest.raises(KeyboardInterrupt):
        handle_termination(signal.SIGTERM, None)


def test_timeout_stops_owned_process_group(tmp_path, monkeypatch):
    args = arguments(tmp_path, "--timeout", "0.001")
    args.output_dir.mkdir()
    process = SimpleNamespace(pid=12345, returncode=None)
    process.poll = lambda: process.returncode
    process.wait = lambda timeout: process.returncode

    class Server:
        def __init__(self, *args, **kwargs):
            assert kwargs["host"] == "127.0.0.1"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def serve_once(self, timeout_ms):
            pass

    def spawn(*args, **kwargs):
        assert kwargs["start_new_session"]
        return process

    stopped = []

    def stop(pid, sig):
        stopped.append((pid, sig))
        process.returncode = -sig

    monkeypatch.setattr("gr00t.policy.server_client.PolicyServer", Server)
    monkeypatch.setattr("gr00t.eval.evaluate_checkpoint.subprocess.Popen", spawn)
    monkeypatch.setattr("gr00t.eval.evaluate_checkpoint.os.killpg", stop)
    with pytest.raises(TimeoutError):
        run_rollouts(None, args)
    assert stopped == [(12345, signal.SIGTERM)]


@pytest.mark.parametrize("module_name", ["gr00t.model.gr00t_n1d6.modules.dit", "gr00t.model.modules.dit"])
def test_alternate_dit_fullgraph_preserves_padding_mask(module_name, monkeypatch):
    """Small CPU graph-capture regression; real CUDA codegen is tested on checkpoints."""
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, "_should_force_math_sdpa", lambda: False)
    model = (
        module.AlternateVLDiT(
            num_attention_heads=2,
            attention_head_dim=4,
            output_dim=8,
            num_layers=4,
            cross_attention_dim=12,
            dropout=0,
            final_dropout=False,
            positional_embeddings=None,
            interleave_self_attention=True,
        )
        .eval()
        .requires_grad_(False)
    )
    inputs = {
        "hidden_states": torch.randn(2, 3, 8),
        "encoder_hidden_states": torch.randn(2, 6, 12),
        "timestep": torch.tensor([0, 250]),
        "image_mask": torch.tensor([[True, True, True, False, False, False]] * 2),
        "backbone_attention_mask": torch.tensor([[True, True, True, True, True, False]] * 2),
    }
    with torch.inference_mode():
        eager = model(**inputs)
        compiled = torch.compile(model.forward, backend="eager", fullgraph=True, dynamic=False)
        torch.testing.assert_close(compiled(**inputs), eager)
        # Padded encoder tokens must remain invisible to image AND text blocks.
        inputs["encoder_hidden_states"][:, -1] += 1000
        torch.testing.assert_close(model(**inputs), eager)
        torch.testing.assert_close(compiled(**inputs), eager)


def result_fixture(tmp_path):
    manifest = {
        "suite": "libero_goal",
        "tasks": [{"task_id": i, "prompt": f"task {i}"} for i in range(10)],
        "episodes_per_task": 40,
    }
    rows = [
        {"suite": "libero_goal", "task_id": i, "episode_index": j, "success": j % 2 == 0, "error": None}
        for i in range(10)
        for j in range(40)
    ]
    summary = {"suite": "libero_goal", "episodes": 400, "successes": 200, "success_rate": 0.5}
    return manifest, rows, summary


def write_results(tmp_path, rows, summary):
    (tmp_path / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (tmp_path / "summary.json").write_text(json.dumps(summary))


def test_complete_400_with_subtask_sr(tmp_path):
    manifest, rows, summary = result_fixture(tmp_path)
    write_results(tmp_path, rows, summary)
    result = validate_results(tmp_path, manifest)
    assert result["status"] == "complete"
    assert len(result["per_task"]) == 10
    assert all(task["episodes"] == 40 and task["success_rate"] == 0.5 for task in result["per_task"])


@pytest.mark.parametrize("failure", ["duplicate", "missing", "unexpected", "error", "summary", "rate", "suite"])
def test_invalid_result_is_not_complete(tmp_path, failure):
    manifest, rows, summary = result_fixture(tmp_path)
    if failure == "duplicate":
        rows[-1] = rows[0]
    elif failure == "missing":
        rows.pop()
    elif failure == "unexpected":
        rows[-1]["episode_index"] = 40
    elif failure == "error":
        rows[0]["error"] = "inference failed"
    elif failure == "summary":
        summary["successes"] = 201
    elif failure == "rate":
        summary["success_rate"] = 0.51
    else:
        rows[0]["suite"] = "libero_object"
    write_results(tmp_path, rows, summary)
    with pytest.raises(RuntimeError):
        validate_results(tmp_path, manifest)
