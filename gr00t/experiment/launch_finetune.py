# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Launch fine-tuning for N1, N1.5, N1.6, or N1.7 on a single node.
# This script tries to provide a similar user experience as current OSS.

import json
import os
from pathlib import Path

import tyro

from gr00t.configs.base_config import get_default_config
from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.experiment.experiment import run


def select_model_config(base_model_path: str, model_version: str = "auto"):
    """Select a pipeline by checkpoint metadata, including local checkpoints."""
    from huggingface_hub import hf_hub_download

    from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
    from gr00t.configs.model.legacy import (
        Gr00tN1d5FinetuneConfig,
        Gr00tN1d6FinetuneConfig,
        Gr00tN1FinetuneConfig,
    )

    path = Path(base_model_path) / "config.json"
    if not path.is_file():
        path = Path(hf_hub_download(base_model_path, "config.json"))
    metadata = json.loads(path.read_text())
    versions = {
        "gr00t_n1": ("1", Gr00tN1FinetuneConfig),
        "gr00t_n1_5": ("1.5", Gr00tN1d5FinetuneConfig),
        "Gr00tN1d6": ("1.6", Gr00tN1d6FinetuneConfig),
        "Gr00tN1d7": ("1.7", Gr00tN1d7Config),
    }
    model_type = metadata.get("model_type")
    if model_type not in versions:
        raise ValueError(f"Unsupported checkpoint model_type: {model_type!r}")
    version, config_class = versions[model_type]
    requested = model_version.lower().removeprefix("n")
    if requested not in ("auto", version):
        raise ValueError(f"Requested {model_version}, but checkpoint is GR00T N{version}")
    return config_class()


# Make sure the user provided modality config is registered.
def load_modality_config(modality_config_path: str):
    import importlib
    import sys

    path = Path(modality_config_path)
    if path.exists() and path.suffix == ".py":
        sys.path.append(str(path.parent))
        importlib.import_module(path.stem)
        print(f"Loaded modality config: {path}")
    else:
        raise FileNotFoundError(f"Modality config path does not exist: {modality_config_path}")


if __name__ == "__main__":
    # Set LOGURU_LEVEL environment variable if not already set (default: INFO)
    if "LOGURU_LEVEL" not in os.environ:
        os.environ["LOGURU_LEVEL"] = "INFO"
    # Use tyro for clean CLI
    ft_config = tyro.cli(FinetuneConfig, description=__doc__)
    from gr00t.data.embodiment_tags import EmbodimentTag

    ft_config.embodiment_tag = EmbodimentTag.resolve(ft_config.embodiment_tag)
    embodiment_tag = ft_config.embodiment_tag.value

    # all rank workers should register for the modality config
    if ft_config.modality_config_path is not None:
        load_modality_config(ft_config.modality_config_path)

    dataset_paths = [path for path in ft_config.dataset_path.split(os.pathsep) if path]

    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": dataset_paths,
                        "mix_ratio": 1.0,
                        "embodiment_tag": embodiment_tag,
                    }
                ],
            }
        }
    )
    config.load_config_path = None
    config.model = select_model_config(ft_config.base_model_path, ft_config.model_version)

    # overwrite with finetune config supplied by the user
    config.model.tune_llm = ft_config.tune_llm
    config.model.tune_visual = ft_config.tune_visual
    config.model.tune_projector = ft_config.tune_projector
    config.model.tune_diffusion_model = ft_config.tune_diffusion_model
    config.model.state_dropout_prob = (
        ft_config.state_dropout_prob
        if ft_config.state_dropout_prob is not None
        else (0.2 if config.model.model_type == "Gr00tN1d7" else 0.0)
    )
    if config.model.model_type in ("gr00t_n1", "gr00t_n1_5") and config.model.state_dropout_prob != 0:
        raise ValueError("N1/N1.5 do not support state dropout; use --state-dropout-prob 0")
    config.model.random_rotation_angle = ft_config.random_rotation_angle
    config.model.color_jitter_params = ft_config.color_jitter_params
    config.model.use_percentiles = ft_config.use_percentiles
    if (ft_config.shortest_image_edge is None) != (ft_config.crop_fraction is None):
        raise ValueError("shortest_image_edge and crop_fraction must be set together")
    if ft_config.shortest_image_edge is not None:
        config.model.shortest_image_edge = ft_config.shortest_image_edge
        config.model.crop_fraction = ft_config.crop_fraction
        config.model.image_crop_size = None
        config.model.image_target_size = None
    if ft_config.extra_augmentation_config:
        config.model.extra_augmentation_config = json.loads(ft_config.extra_augmentation_config)
    else:
        config.model.extra_augmentation_config = None

    if config.model.model_type == "Gr00tN1d7":
        config.model.load_bf16 = False
        config.model.reproject_vision = False
        config.model.backbone_trainable_params_fp32 = True
        config.model.use_relative_action = True

    config.training.experiment_name = ft_config.experiment_name
    config.training.start_from_checkpoint = ft_config.base_model_path
    config.training.optim = "adamw_torch"
    config.training.global_batch_size = ft_config.global_batch_size
    config.training.dataloader_num_workers = ft_config.dataloader_num_workers
    config.training.learning_rate = ft_config.learning_rate
    config.training.gradient_accumulation_steps = ft_config.gradient_accumulation_steps
    config.training.output_dir = ft_config.output_dir
    config.training.save_steps = ft_config.save_steps
    config.training.save_total_limit = ft_config.save_total_limit
    config.training.num_gpus = ft_config.num_gpus
    config.training.use_wandb = ft_config.use_wandb
    config.training.max_steps = ft_config.max_steps
    config.training.weight_decay = ft_config.weight_decay
    config.training.weight_decay_all_parameters = ft_config.weight_decay_all_parameters
    config.training.warmup_ratio = ft_config.warmup_ratio
    config.training.wandb_project = ft_config.wandb_project
    config.training.lr_scheduler_type = ft_config.lr_scheduler_type
    config.training.max_grad_norm = ft_config.max_grad_norm
    config.training.gradient_checkpointing = ft_config.gradient_checkpointing
    config.training.logging_steps = ft_config.logging_steps
    config.data.seed = ft_config.seed
    config.data.allow_padding = ft_config.allow_padding

    config.data.shard_size = ft_config.shard_size
    config.data.episode_sampling_rate = ft_config.episode_sampling_rate
    config.data.num_shards_per_epoch = ft_config.num_shards_per_epoch
    config.data.ds_weights_alpha = ft_config.ds_weights_alpha

    config.training.save_only_model = ft_config.save_only_model
    config.training.save_final_model = ft_config.save_final_model
    config.training.resume_from_checkpoint = ft_config.resume_from_checkpoint
    config.training.skip_weight_loading = ft_config.skip_weight_loading

    run(config)
