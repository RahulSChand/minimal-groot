from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_unselected_subsystems_are_absent():
    for relative_path in (
        "external_dependencies/SimplerEnv",
        "external_dependencies/robocasa",
        "external_dependencies/robocasa-gr1-tabletop-tasks",
        "scripts/deployment",
        "gr00t/eval/real_robot",
        "gr00t/eval/sim/SimplerEnv",
        "gr00t/eval/sim/robocasa",
        "gr00t/eval/sim/robocasa365",
        "gr00t/eval/sim/robocasa-gr1-tabletop-tasks",
    ):
        assert not (ROOT / relative_path).exists()


def test_required_training_resources_are_present():
    required_paths = (
        "gr00t/experiment/launch_finetune.py",
        "gr00t/model/gr00t_n1d7/gr00t_n1d7.py",
        "gr00t/model/modules/qwen3_backbone.py",
        "gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py",
        "gr00t/data/dataset/lerobot_episode_loader.py",
        "gr00t/configs/deepspeed/zero2_config.json",
        "gr00t/configs/deepspeed/zero3_config.json",
        "examples/SO100/so100_config.py",
    )
    for relative_path in required_paths:
        assert (ROOT / relative_path).is_file()


def test_finetune_entrypoint_imports():
    import gr00t.experiment.launch_finetune  # noqa: F401


def test_required_evaluation_resources_are_present():
    required_paths = (
        "gr00t/policy/gr00t_policy.py",
        "gr00t/policy/server_client.py",
        "gr00t/eval/run_gr00t_server.py",
        "gr00t/eval/rollout_policy.py",
        "gr00t/eval/_horizon_contract.py",
        "gr00t/eval/sim/wrapper/multistep_wrapper.py",
        "gr00t/eval/sim/LIBERO/libero_env.py",
        "gr00t/eval/sim/LIBERO/setup_libero.sh",
    )
    for relative_path in required_paths:
        assert (ROOT / relative_path).is_file()
