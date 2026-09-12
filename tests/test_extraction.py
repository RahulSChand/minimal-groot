from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_excluded_subsystems_are_absent():
    for relative_path in (
        "gr00t/deployment",
        "gr00t/eval",
        "gr00t/policy",
        "external_dependencies",
        "scripts/deployment",
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
