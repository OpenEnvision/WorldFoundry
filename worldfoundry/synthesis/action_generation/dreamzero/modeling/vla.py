"""Inference-only DreamZero VLA assembly and checkpoint loading."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import os

from hydra.utils import instantiate
import numpy as np
from safetensors.torch import load_file
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
import tree
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature

from .targets import rewrite_targets, validate_inference_targets


BACKBONE_FEATURE_KEY = "backbone_features"
ACTION_KEY = "action_pred"
N_COLOR_CHANNELS = 3
logger = logging.getLogger(__name__)


@dataclass
class VLAConfig(PretrainedConfig):
    model_type = "vla"
    backbone_cfg: PretrainedConfig = field(default=None)
    action_head_cfg: PretrainedConfig = field(default=None)
    action_horizon: int = field(default=None)
    action_dim: int = field(default=None)
    compute_dtype: str = field(default="float32")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


def _checkpoint_state_dict(checkpoint_dir: str) -> dict[str, torch.Tensor]:
    """Read a local single-file or sharded safetensors checkpoint."""

    single_path = os.path.join(checkpoint_dir, "model.safetensors")
    index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as handle:
            index = json.load(handle)
        shard_names = sorted(set(index["weight_map"].values()))
        state_dict: dict[str, torch.Tensor] = {}
        for shard_name in shard_names:
            state_dict.update(load_file(os.path.join(checkpoint_dir, shard_name)))
    elif os.path.isfile(single_path):
        state_dict = dict(load_file(single_path))
    else:
        raise FileNotFoundError(
            f"No model.safetensors checkpoint found in {checkpoint_dir!r}"
        )

    if any(".base_layer." in key for key in state_dict):
        state_dict = {
            key.replace(".base_layer.", "."): value
            for key, value in state_dict.items()
        }
    return state_dict


def _action_head_config(config: VLAConfig) -> dict:
    action_head = config.action_head_cfg
    if not isinstance(action_head, dict):
        raise TypeError("DreamZero action_head_cfg must be a mapping")
    nested = action_head.get("config")
    return nested if isinstance(nested, dict) else action_head


class VLA(PreTrainedModel):
    """Minimal runtime used by the DreamZero causal policy server."""

    supports_gradient_checkpointing = False
    config_class = VLAConfig

    def __init__(self, config: VLAConfig):
        if not isinstance(config.backbone_cfg, dict):
            raise TypeError("DreamZero backbone_cfg must be a mapping")
        if not isinstance(config.action_head_cfg, dict):
            raise TypeError("DreamZero action_head_cfg must be a mapping")
        super().__init__(config)
        validate_inference_targets(config.backbone_cfg, path="model.backbone_cfg")
        validate_inference_targets(config.action_head_cfg, path="model.action_head_cfg")
        self.backbone = instantiate(config.backbone_cfg)
        self.action_head = instantiate(config.action_head_cfg)
        self.action_horizon = config.action_horizon
        self.action_dim = config.action_dim
        self.compute_dtype = config.compute_dtype
        self.rank = dist.get_rank() if dist.is_initialized() else 0

    def _validate_inputs(self, inputs: dict) -> None:
        action = inputs.get("action")
        if action is not None:
            if not isinstance(action, torch.Tensor):
                raise TypeError(f"action must be a tensor, got {type(action)!r}")
            if (
                action.ndim != 3
                or action.shape[1] % self.action_horizon
                or action.shape[2] != self.action_dim
            ):
                raise ValueError(f"invalid action shape: {tuple(action.shape)}")

        video = inputs.get("video")
        if video is not None:
            if not isinstance(video, np.ndarray) or video.dtype != np.uint8:
                raise TypeError("video must be a uint8 numpy array")
            if video.ndim != 6 or video.shape[3] != N_COLOR_CHANNELS:
                raise ValueError(f"invalid video shape: {video.shape}")

    def _validate_outputs(
        self,
        action_head_outputs: BatchFeature,
        backbone_outputs: BatchFeature,
    ) -> None:
        if not isinstance(backbone_outputs, BatchFeature):
            raise TypeError("backbone output must be a BatchFeature")
        if BACKBONE_FEATURE_KEY not in backbone_outputs:
            raise KeyError(f"backbone output is missing {BACKBONE_FEATURE_KEY!r}")
        if not isinstance(action_head_outputs, BatchFeature):
            raise TypeError("action-head output must be a BatchFeature")
        if ACTION_KEY not in action_head_outputs:
            raise KeyError(f"action-head output is missing {ACTION_KEY!r}")
        action = action_head_outputs[ACTION_KEY]
        if action.shape[1:] != (self.action_horizon, self.action_dim):
            raise ValueError(
                "invalid predicted action shape: "
                f"{tuple(action.shape)}; expected (*, {self.action_horizon}, {self.action_dim})"
            )

    def prepare_input(self, inputs: dict) -> tuple[BatchFeature, BatchFeature]:
        self._validate_inputs(inputs)
        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        def move(value):
            if not isinstance(value, torch.Tensor):
                return value
            if torch.is_floating_point(value):
                return value.to(self.device, dtype=self.action_head.dtype)
            return value.to(self.device)

        return (
            tree.map_structure(move, backbone_inputs),
            tree.map_structure(move, action_inputs),
        )

    @torch.inference_mode()
    def lazy_joint_video_action_causal(
        self,
        inputs: dict,
        latent_video: torch.Tensor | None = None,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        outputs = self.action_head.lazy_joint_video_action(
            backbone_outputs,
            action_inputs,
            latent_video=latent_video,
        )
        self._validate_outputs(outputs, backbone_outputs)
        return outputs

    @classmethod
    def _from_local_checkpoint(
        cls,
        checkpoint_dir: str,
        *,
        load_base_components: bool,
        config: VLAConfig | None = None,
    ) -> "VLA":
        if config is None:
            config_path = os.path.join(checkpoint_dir, "config.json")
            with open(config_path, encoding="utf-8") as handle:
                config_dict = json.load(handle)
            rewrite_targets(config_dict)
            config = VLAConfig(**config_dict)
        action_head = _action_head_config(config)
        # PEFT modules must exist before loading adapter-shaped checkpoint keys.
        action_head["defer_lora_injection"] = False
        # Full DreamZero checkpoints already contain DiT, text/image encoders,
        # and VAE. Older DROID configs omit this flag; loading those large base
        # assets first is redundant and can exhaust host memory. Adapter-only
        # checkpoints, by contrast, do need the frozen base components.
        action_head["skip_component_loading"] = not load_base_components

        model = cls(config)
        incompatible = model.load_state_dict(
            _checkpoint_state_dict(checkpoint_dir),
            strict=False,
        )
        if incompatible.unexpected_keys:
            logger.warning(
                "DreamZero checkpoint has %d unexpected compatibility keys: %s",
                len(incompatible.unexpected_keys),
                incompatible.unexpected_keys[:20],
            )
        # Base-component LoRA checkpoints intentionally omit frozen base tensors.
        if incompatible.missing_keys and not load_base_components:
            logger.warning(
                "DreamZero checkpoint has %d missing compatibility keys: %s",
                len(incompatible.missing_keys),
                incompatible.missing_keys[:20],
            )
        model.requires_grad_(False)
        model.eval()
        return model

    @classmethod
    def load_lora(cls, pretrained_model_name_or_path: str) -> "VLA":
        """Load an adapter-only checkpoint on top of its frozen Wan components."""

        return cls._from_local_checkpoint(
            pretrained_model_name_or_path,
            load_base_components=True,
            config=None,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        config: VLAConfig | None = None,
        **_: object,
    ) -> "VLA":
        return cls._from_local_checkpoint(
            pretrained_model_name_or_path,
            load_base_components=False,
            config=config,
        )

    def post_initialize(self) -> None:
        self.action_head.post_initialize()

    def parallelize(self, device_mesh: DeviceMesh) -> None:
        self.action_head.parallelize(device_mesh=device_mesh)


AutoConfig.register("vla", VLAConfig)
AutoModel.register(VLAConfig, VLA)
