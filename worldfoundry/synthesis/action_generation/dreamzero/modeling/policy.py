"""Inference policy wrapper for DreamZero checkpoints."""

from __future__ import annotations

import importlib
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf
from tianshou.data import Batch
from torch.distributed.device_mesh import DeviceMesh

from ..preprocessing import ComposedModalityTransform, DatasetMetadata, EmbodimentTag
from .targets import rewrite_targets, validate_inference_targets

logger = logging.getLogger(__name__)


def _update_tokenizer_path_in_config(config, new_path: str) -> None:
    """Override tokenizer paths without resolving unrelated checkpoint fields."""

    if isinstance(config, DictConfig):
        if "tokenizer_path" in config:
            config.tokenizer_path = new_path
        transforms = config.get("transforms")
        if isinstance(transforms, DictConfig):
            for value in transforms.values():
                _update_tokenizer_path_in_config(value, new_path)
        elif isinstance(transforms, ListConfig):
            for value in transforms:
                _update_tokenizer_path_in_config(value, new_path)
    elif isinstance(config, ListConfig):
        for value in config:
            _update_tokenizer_path_in_config(value, new_path)


def _model_class_from_target(target: str):
    for loader_name in (
        ".from_pretrained",
        ".from_pretrained_for_tuning",
        ".from_pretrained_with_wrapped_action_head",
    ):
        if target.endswith(loader_name):
            target = target[: -len(loader_name)]
            break
    if target != "worldfoundry.synthesis.action_generation.dreamzero.modeling.vla.VLA":
        raise ValueError(f"DreamZero model target is not allowlisted: {target}")
    module_name, class_name = target.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)


def _copy_observation(observation: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (value.copy() if isinstance(value, np.ndarray) else value.clone() if torch.is_tensor(value) else value)
        for key, value in observation.items()
    }


class GrootSimPolicy:
    """Load a DreamZero checkpoint and expose its causal video/action inference."""

    def __init__(
        self,
        embodiment_tag: EmbodimentTag,
        model_path: str,
        device: int | str,
        model_config_overrides: list[str] | None = None,
        tokenizer_path_override: str | None = None,
        skip_assert_delta_indices: bool = False,
        skip_img_transform: bool = False,
        lazy_load: bool = False,
        device_mesh: DeviceMesh | None = None,
    ):
        self.embodiment_tag = embodiment_tag
        self.model_path = model_path
        self.device = torch.device(f"cuda:{device}" if isinstance(device, int) else device)
        self.rank = dist.get_rank() if dist.is_initialized() else 0

        model_dir = Path(model_path)
        experiment_dir = model_dir / "experiment_cfg"
        checkpoint_cfg = OmegaConf.load(experiment_dir / "conf.yaml")
        rewrite_targets(checkpoint_cfg)
        if tokenizer_path_override is not None:
            _update_tokenizer_path_in_config(checkpoint_cfg, tokenizer_path_override)
        self.checkpoint_cfg = checkpoint_cfg

        model_class = _model_class_from_target(str(checkpoint_cfg.model._target_))
        model_config = None
        if model_config_overrides:
            with (model_dir / "config.json").open(encoding="utf-8") as handle:
                model_config_dict = json.load(handle)
            rewrite_targets(model_config_dict)
            merged = OmegaConf.create(model_config_dict)
            merged.merge_with_dotlist(model_config_overrides)
            model_config = model_class.config_class.from_dict(OmegaConf.to_container(merged, resolve=True))

        if bool(checkpoint_cfg.get("save_lora_only", False)):
            model = model_class.load_lora(model_path)
        else:
            model = model_class.from_pretrained(
                model_path,
                config=model_config,
                local_files_only=True,
                trust_remote_code=False,
            )

        model.requires_grad_(False)
        model.eval()
        if getattr(model.action_head, "train_architecture", None) == "lora":
            if not hasattr(model.action_head.model, "merge_and_unload"):
                raise RuntimeError("DreamZero LoRA checkpoint did not materialize PEFT adapters")
            model.action_head.model = model.action_head.model.merge_and_unload()

        self.eval_bf16 = bool(checkpoint_cfg.get("eval_bf16", False))
        if self.eval_bf16:
            model.to(dtype=torch.bfloat16)
        model.to(self.device)
        model.post_initialize()
        if device_mesh is not None:
            model.parallelize(device_mesh=device_mesh)
        self.trained_model = model

        if lazy_load:
            self.offload_to_cpu()

        with (experiment_dir / "metadata.json").open(encoding="utf-8") as handle:
            all_metadata = json.load(handle)
        if "gr1_unified_offline_rl" in all_metadata and self.embodiment_tag.value == "gr1_unified":
            self.embodiment_tag = EmbodimentTag.GR1_UNIFIED_OFFLINE_RL
        metadata = DatasetMetadata.model_validate(all_metadata[self.embodiment_tag.value])

        action_head_config = self.trained_model.action_head.config
        target_height = getattr(action_head_config, "target_video_height", None)
        target_width = getattr(action_head_config, "target_video_width", None)
        if target_height is not None and target_width is not None:
            for video_metadata in metadata.modalities.video.values():
                video_metadata.resolution = (int(target_width), int(target_height))

        if self.embodiment_tag.value not in checkpoint_cfg.transforms:
            raise KeyError(f"No DreamZero transform for embodiment {self.embodiment_tag.value!r}")
        transform_config = checkpoint_cfg.transforms[self.embodiment_tag.value]
        if skip_img_transform:
            self._remove_image_transforms(transform_config, metadata)

        validate_inference_targets(transform_config, path="checkpoint.transforms")
        eval_transform = instantiate(transform_config)
        if not isinstance(eval_transform, ComposedModalityTransform):
            raise TypeError("DreamZero checkpoint must use ComposedModalityTransform")
        eval_transform.set_metadata(metadata)
        if bool(checkpoint_cfg.get("relative_action_per_horizon", False)):
            per_horizon = self._per_horizon_statistics(metadata)
            if not per_horizon:
                raise ValueError("relative per-horizon actions require 2-D action statistics")
            eval_transform.set_per_horizon_statistics(per_horizon)
        eval_transform.eval()
        self.eval_transform = eval_transform

        modality_configs = checkpoint_cfg.modality_configs
        if self.embodiment_tag.value in modality_configs:
            modality_configs = modality_configs[self.embodiment_tag.value]
        validate_inference_targets(modality_configs, path="checkpoint.modality_configs")
        self.modality_configs = instantiate(modality_configs)
        self._video_delta_indices = np.asarray(self.modality_configs.video.eval_delta_indices)
        self._video_horizon = len(self._video_delta_indices)
        if "state" in self.modality_configs:
            self._state_delta_indices = np.asarray(self.modality_configs.state.eval_delta_indices)
            if not skip_assert_delta_indices:
                self.assert_delta_indices(self._state_delta_indices)
            self._state_horizon = len(self._state_delta_indices)
        else:
            self._state_delta_indices = None
            self._state_horizon = None

    @staticmethod
    def _remove_image_transforms(config, metadata: DatasetMetadata) -> None:
        for transform in config.transforms:
            target = str(transform._target_)
            if target.endswith(".VideoCrop"):
                for video_metadata in metadata.modalities.video.values():
                    width, height = video_metadata.resolution
                    video_metadata.resolution = (
                        int(width * transform.scale),
                        int(height * transform.scale),
                    )
            elif target.endswith(".VideoResize"):
                for video_metadata in metadata.modalities.video.values():
                    video_metadata.resolution = (
                        int(transform.width),
                        int(transform.height),
                    )
        skipped = (".VideoCrop", ".VideoResize", ".VideoColorJitter")
        config.transforms = [
            transform for transform in config.transforms if not str(transform._target_).endswith(skipped)
        ]

    @staticmethod
    def _per_horizon_statistics(metadata: DatasetMetadata) -> dict[str, dict]:
        result = {}
        for action_key, statistics in metadata.statistics.action.items():
            values = statistics.model_dump()
            q01 = values.get("q01")
            if isinstance(q01, (list, np.ndarray)) and len(q01) and isinstance(q01[0], (list, np.ndarray)):
                result[action_key] = {
                    key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in values.items()
                }
        return result

    def offload_to_cpu(self) -> None:
        self.trained_model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_to_gpu(self) -> None:
        self.trained_model.to(self.device)
        if self.eval_bf16:
            self.trained_model.to(dtype=torch.bfloat16)

    def ensure_model_on_gpu(self) -> None:
        if next(self.trained_model.parameters()).device.type == "cpu":
            self.load_to_gpu()

    @staticmethod
    def assert_delta_indices(delta_indices: np.ndarray) -> None:
        if not np.all(delta_indices <= 0) or delta_indices[-1] != 0:
            raise ValueError(f"observation delta indices must end at zero: {delta_indices}")
        if len(delta_indices) > 1:
            steps = np.diff(delta_indices)
            if not np.all(steps == steps[0]) or steps[0] <= 0:
                raise ValueError(f"observation delta indices must be regular: {delta_indices}")

    def apply(self, batch: Batch) -> Batch:
        batch.normalized_obs = self.eval_transform(batch.obs)
        return batch

    def unapply(self, batch: Batch, obs: dict | None = None) -> Batch:
        action = self.eval_transform.unapply({"action": batch.normalized_action.cpu()})
        config = self.checkpoint_cfg
        relative = bool(config.get("relative_action", False)) or bool(config.get("relative_action_per_horizon", False))
        relative_keys = config.get("relative_action_keys", [])
        if relative and relative_keys and obs is not None:
            self._relative_to_absolute(action, obs, relative_keys)
        batch.act = action
        return batch

    @staticmethod
    def _relative_to_absolute(
        action: dict,
        observation: dict,
        relative_keys,
    ) -> None:
        for key in relative_keys:
            action_key = f"action.{key}"
            state_key = f"state.{key}"
            if action_key not in action:
                continue
            state = observation.get(state_key)
            if state is None:
                state = next(
                    (value for name, value in observation.items() if "state" in name and key in name),
                    None,
                )
            if state is None and "state" in observation:
                candidate = observation["state"]
                if getattr(candidate, "shape", (None,))[-1] == action[action_key].shape[-1]:
                    state = candidate
            if state is None:
                continue
            if torch.is_tensor(state):
                state = state.detach().cpu().numpy()
            state = np.asarray(state)
            if state.ndim >= 2:
                state = state[..., -1, :]
            if action[action_key].ndim > state.ndim:
                state = np.expand_dims(state, axis=-2)
            action[action_key] = action[action_key] + state

    @torch.inference_mode()
    def lazy_joint_forward_causal(
        self,
        batch: Batch,
        video=None,
        latent_video=None,
        state=None,
        video_only: bool = False,
        **kwargs,
    ) -> tuple[Batch, torch.Tensor]:
        del state, kwargs
        self.ensure_model_on_gpu()
        start = time.perf_counter()
        original_observation = _copy_observation(batch.obs)
        is_batched = self._check_state_is_batched(batch.obs)
        if not is_batched:
            batch.obs = unsqueeze_dict_values(batch.obs)
            original_observation = unsqueeze_dict_values(original_observation)

        batch = self.apply(batch)
        normalized_input = batch.normalized_obs
        if isinstance(normalized_input, Batch):
            normalized_input = normalized_input.__getstate__()
        if video is not None:
            for key in normalized_input:
                if "images" in key:
                    normalized_input[key] = video
        if self.eval_bf16:
            for key, value in normalized_input.items():
                if torch.is_tensor(value) and value.dtype == torch.float32:
                    normalized_input[key] = value.to(dtype=torch.bfloat16)

        model_output = self.trained_model.lazy_joint_video_action_causal(
            normalized_input,
            latent_video=latent_video,
        )
        normalized_action = model_output["action_pred"].float()
        video_prediction = model_output["video_pred"]
        if video_only:
            result = Batch(normalized_action=normalized_action)
        else:
            result = self.unapply(
                Batch(normalized_action=normalized_action),
                obs=original_observation,
            )
            if not is_batched:
                result.act = squeeze_dict_values(result.act)
        if self.rank == 0:
            logger.info("DreamZero inference completed in %.3f s", time.perf_counter() - start)
        return result, video_prediction

    @staticmethod
    def _check_state_is_batched(observation: dict[str, Any]) -> bool:
        for key, value in observation.items():  # noqa: SIM110 - Keeps the explicit loop.
            if "state" in key and len(value.shape) < 3:
                return False
        return True

    @property
    def raw_data_image_transform(self) -> Callable:
        return lambda value: value

    @property
    def state_delta_indices(self) -> np.ndarray | None:
        return self._state_delta_indices

    @property
    def video_delta_indices(self) -> np.ndarray:
        return self._video_delta_indices


def unsqueeze_dict_values(data: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            result[key] = np.expand_dims(value, axis=0)
        elif isinstance(value, list):
            result[key] = np.asarray(value)
        elif isinstance(value, torch.Tensor):
            result[key] = value.unsqueeze(0)
        elif isinstance(value, str):
            result[key] = np.asarray([value])
        else:
            result[key] = value
    return result


def squeeze_dict_values(data: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (
            np.squeeze(value)
            if isinstance(value, np.ndarray)
            else value.squeeze()
            if isinstance(value, torch.Tensor)
            else value
        )
        for key, value in data.items()
    }
