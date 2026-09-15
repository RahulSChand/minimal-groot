"""Training options for the Eagle-based GR00T generations.

Checkpoint architecture dimensions are read from the native Hugging Face config.
These dataclasses configure the shared experiment and preprocessing pipeline.
"""

from dataclasses import dataclass

from . import register_model_config
from .gr00t_n1d7 import Gr00tN1d7Config


@dataclass
class Gr00tN1d6FinetuneConfig(Gr00tN1d7Config):
    model_type: str = "Gr00tN1d6"
    model_name: str = "nvidia/Eagle-Block2A-2B-v2"
    backbone_model_type: str = "eagle"
    load_bf16: bool = True
    action_horizon: int = 50
    max_state_dim: int = 128
    max_action_dim: int = 128
    use_relative_action: bool = False
    image_crop_size: tuple[int, int] | None = None
    image_target_size: tuple[int, int] | None = None
    shortest_image_edge: int | None = 256
    crop_fraction: float | None = 0.95
    letter_box_transform: bool = False


@dataclass
class Gr00tN1d5FinetuneConfig(Gr00tN1d6FinetuneConfig):
    model_type: str = "gr00t_n1_5"
    model_name: str = "gr00t-n1.5-eagle"
    backbone_model_type: str = "eagle15"
    action_horizon: int = 16
    max_state_dim: int = 64
    max_action_dim: int = 32
    apply_sincos_state_encoding: bool = False
    formalize_language: bool = False


@dataclass
class Gr00tN1FinetuneConfig(Gr00tN1d5FinetuneConfig):
    model_type: str = "gr00t_n1"
    model_name: str = "gr00t-n1-eagle"
    backbone_model_type: str = "eagle1"


register_model_config("Gr00tN1", Gr00tN1FinetuneConfig)
register_model_config("Gr00tN1d5", Gr00tN1d5FinetuneConfig)
register_model_config("Gr00tN1d6", Gr00tN1d6FinetuneConfig)
