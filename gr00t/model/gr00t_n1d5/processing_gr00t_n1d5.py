"""N1.5 Eagle prompting with the shared state/action dataset interface."""

from transformers import AutoProcessor

from gr00t.model.gr00t_n1d6.processing_gr00t_n1d6 import Gr00tN1d6Processor

from .gr00t_n1 import GR00T_N1_5_Config


class Gr00tN1d5Processor(Gr00tN1d6Processor):
    def __init__(self, **kwargs):
        defaults = dict(
            model_name="gr00t-n1.5-eagle",
            model_type="eagle15",
            max_state_dim=64,
            max_action_dim=32,
            max_action_horizon=16,
            formalize_language=False,
            apply_sincos_state_encoding=False,
            use_relative_action=False,
            embodiment_id_mapping={
                "new_embodiment": 31,
                "libero_sim": 31,
                "libero_panda": 31,
                "oxe_droid": 17,
                "agibot_genie1": 26,
                "gr1": 24,
            },
        )
        defaults.update(kwargs)
        super().__init__(**defaults)


AutoProcessor.register(GR00T_N1_5_Config, Gr00tN1d5Processor)
