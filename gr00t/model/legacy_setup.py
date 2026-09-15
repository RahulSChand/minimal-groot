"""Adapters from native N1/N1.5/N1.6 checkpoints to the shared training pipeline."""

import json
import logging
from pathlib import Path

from transformers import AutoConfig

from gr00t.configs.model.legacy import (
    Gr00tN1d5FinetuneConfig,
    Gr00tN1d6FinetuneConfig,
    Gr00tN1FinetuneConfig,
)
from gr00t.data.dataset.factory import DatasetFactory
from gr00t.model.gr00t_n1.gr00t_n1 import GR00T_N1
from gr00t.model.gr00t_n1.processing_gr00t_n1 import Gr00tN1Processor
from gr00t.model.gr00t_n1d5.gr00t_n1 import GR00T_N1_5
from gr00t.model.gr00t_n1d5.processing_gr00t_n1d5 import Gr00tN1d5Processor
from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6
from gr00t.model.gr00t_n1d6.processing_gr00t_n1d6 import Gr00tN1d6Processor
from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline, convert_tensors_to_lists
from gr00t.model.registry import register_model
from gr00t.utils.dist_utils import run_or_wait_on_rank0


class EaglePipeline(Gr00tN1d7Pipeline):
    def _create_model(self):
        checkpoint = self.config.training.start_from_checkpoint
        if checkpoint is None:
            raise ValueError("N1/N1.5/N1.6 training requires a base checkpoint")
        native = AutoConfig.from_pretrained(checkpoint, **self.transformers_loading_kwargs)
        tuning = {
            k: getattr(self.model_config, k)
            for k in ("tune_llm", "tune_visual", "tune_projector", "tune_diffusion_model")
        }
        if self.model_class in (GR00T_N1, GR00T_N1_5):
            native.backbone_cfg.update({k: tuning[k] for k in ("tune_llm", "tune_visual")})
            native.action_head_cfg.update({k: tuning[k] for k in ("tune_projector", "tune_diffusion_model")})
            self.model_config.max_state_dim = native.action_head_cfg["max_state_dim"]
            self.model_config.max_action_dim = native.action_dim
        else:
            for key, value in tuning.items():
                setattr(native, key, value)
            native.state_dropout_prob = self.model_config.state_dropout_prob
            self.model_config.max_state_dim = native.max_state_dim
            self.model_config.max_action_dim = native.max_action_dim
            self.model_config.apply_sincos_state_encoding = native.apply_sincos_state_encoding
        self.model_config.action_horizon = native.action_horizon
        if self.config.training.skip_weight_loading:
            model = self.model_class(native)
        else:
            model, info = self.model_class.from_pretrained(
                checkpoint,
                config=native,
                output_loading_info=True,
                **(tuning if self.model_class in (GR00T_N1, GR00T_N1_5) else {}),
                **self.transformers_loading_kwargs,
            )
            # A state-dropout mask is the only intentionally added parameter.
            missing = [k for k in info["missing_keys"] if k != "action_head.mask_token"]
            if missing or info["unexpected_keys"] or info["mismatched_keys"]:
                raise RuntimeError(f"Checkpoint architecture mismatch: {info}")
            if "action_head.mask_token" in info["missing_keys"]:
                import torch

                torch.nn.init.normal_(model.action_head.mask_token, std=0.02)
        with run_or_wait_on_rank0(label="legacy model config") as is_rank0:
            if is_rank0:
                (self.save_cfg_dir / "final_model_config.json").write_text(model.config.to_json_string())
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info("Parameters: %s total, %s trainable", f"{total:,}", f"{trainable:,}")
        return model

    def _create_processor(self):
        cfg = self.model_config
        kwargs = {
            key: getattr(cfg, key)
            for key in (
                "model_name",
                "formalize_language",
                "max_state_dim",
                "max_action_dim",
                "apply_sincos_state_encoding",
                "use_relative_action",
                "use_percentiles",
                "image_crop_size",
                "image_target_size",
                "shortest_image_edge",
                "crop_fraction",
                "random_rotation_angle",
                "color_jitter_params",
                "extra_augmentation_config",
                "letter_box_transform",
            )
        }
        kwargs.update(
            modality_configs=self.config.data.modality_configs,
            model_type=cfg.backbone_model_type,
            max_action_horizon=cfg.action_horizon,
            use_albumentations=cfg.use_albumentations_transforms,
            transformers_loading_kwargs=self.transformers_loading_kwargs,
        )
        checkpoint = self.config.training.start_from_checkpoint
        # Native N1.5 releases have no processor file. Fine-tuned checkpoints do.
        from transformers.utils import cached_file

        processor_file = cached_file(checkpoint, "processor_config.json", _raise_exceptions_for_missing_entries=False)
        if processor_file:
            saved = json.loads(Path(processor_file).read_text())["processor_kwargs"]
            modalities = dict(saved["modality_configs"])
            modalities.update(kwargs["modality_configs"])
            kwargs["modality_configs"] = modalities
            saved.update(kwargs)
            # Retain checkpoint-specific embodiment IDs and statistics without
            # constructing a second copy of the vision/language processor.
            for key, filename in (
                ("statistics", "statistics.json"),
                ("embodiment_id_mapping", "embodiment_id.json"),
            ):
                saved[key] = json.loads(Path(cached_file(checkpoint, filename)).read_text())
            kwargs = saved
        return self.processor_class(**kwargs)

    def _create_dataset(self, save_cfg_dir: Path):
        self.processor = self._create_processor()
        train_dataset, eval_dataset = DatasetFactory(self.config).build(self.processor)
        with run_or_wait_on_rank0(label="legacy dataset statistics") as is_rank0:
            if is_rank0:
                stats = train_dataset.get_dataset_statistics()
                (save_cfg_dir / "dataset_statistics.json").write_text(
                    json.dumps(convert_tensors_to_lists(stats), indent=2)
                )
        return train_dataset, eval_dataset


class Gr00tN1Pipeline(EaglePipeline):
    model_class = GR00T_N1
    processor_class = Gr00tN1Processor


class Gr00tN1d5Pipeline(EaglePipeline):
    model_class = GR00T_N1_5
    processor_class = Gr00tN1d5Processor


class Gr00tN1d6Pipeline(EaglePipeline):
    model_class = Gr00tN1d6
    processor_class = Gr00tN1d6Processor


register_model(Gr00tN1FinetuneConfig, Gr00tN1Pipeline)
register_model(Gr00tN1d5FinetuneConfig, Gr00tN1d5Pipeline)
register_model(Gr00tN1d6FinetuneConfig, Gr00tN1d6Pipeline)
