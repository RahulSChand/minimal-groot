"""LIBERO's two cameras, 8D EEF state and 7D controller actions."""

from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


def libero_spatial_config(action_horizon: int = 16):
    keys = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    return {
        "video": ModalityConfig(delta_indices=[0], modality_keys=["image", "wrist_image"]),
        "state": ModalityConfig(delta_indices=[0], modality_keys=keys),
        "action": ModalityConfig(
            delta_indices=list(range(action_horizon)),
            modality_keys=keys,
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.DELTA if key != "gripper" else ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                )
                for key in keys
            ],
        ),
        "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.action.task_description"]),
    }
