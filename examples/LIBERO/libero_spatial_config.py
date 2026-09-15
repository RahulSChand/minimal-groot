"""Use 16-step action chunks (supported by every GR00T generation)."""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.configs.data.libero_spatial import libero_spatial_config
from gr00t.data.embodiment_tags import EmbodimentTag

register_modality_config(libero_spatial_config(), embodiment_tag=EmbodimentTag.LIBERO_PANDA)
