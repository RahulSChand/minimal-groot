"""Original N1 Eagle prompting with the shared state/action interface."""

import numpy as np
import torch
from transformers import AutoProcessor
from transformers.feature_extraction_utils import BatchFeature

from gr00t.model.gr00t_n1d5.processing_gr00t_n1d5 import Gr00tN1d5Processor

from .backbone.eagle2_hg_model.inference_eagle_repo import EagleProcessor
from .gr00t_n1 import GR00T_N1Config


class Gr00tN1DataCollator:
    def __init__(self, model_name="gr00t-n1-eagle", model_type="eagle1", transformers_loading_kwargs=None):
        self.processor = EagleProcessor(max_input_tiles=1)

    def __call__(self, features):
        batch = {}
        keys = set().union(*(feature.keys() for feature in features))
        for key in keys:
            values = [feature[key] for feature in features]
            if key == "vlm_content":
                encoded = [self.processor.prepare_input(value) for value in values]
                batch.update(self.processor.collate_fn(encoded))
            else:
                batch[key] = torch.from_numpy(np.stack(values))
        return BatchFeature(data={"inputs": batch})


class Gr00tN1Processor(Gr00tN1d5Processor):
    data_collator_class = Gr00tN1DataCollator

    def __init__(self, **kwargs):
        defaults = {
            "model_name": "gr00t-n1-eagle",
            "model_type": "eagle1",
            "embodiment_id_mapping": {
                "gr1": 24,
                "new_embodiment": 31,
                "libero_sim": 31,
                "libero_panda": 31,
            },
        }
        defaults.update(kwargs)
        super().__init__(**defaults)

    def _apply_vlm_processing(self, images, language):
        return {
            "vlm_content": {
                "prompt": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": language,
                        "image": [{"np_array": np.transpose(image, (1, 2, 0))} for image in images],
                    },
                ]
            }
        }


AutoProcessor.register(GR00T_N1Config, Gr00tN1Processor)
