"""Backbone loaders shared by latent-action model variants.

The loaders prefer WorldFoundry's model capability paths while retaining the
legacy LARYBench environment variables for checkpoint compatibility.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import torch

from worldfoundry.base_models.perception_core.general_perception.dino_embeddings import (
    _preferred_hf_model_source,
)


def _source(
    legacy_env: str,
    capability_id: str,
    asset_id: str,
    fallback_repo_id: str,
    env: Mapping[str, str] | None = None,
) -> str:
    values = os.environ if env is None else env
    legacy = values.get(legacy_env)
    if legacy:
        return legacy.replace("hdfs:///", "/mnt/hdfs/")
    return _preferred_hf_model_source(capability_id, asset_id, fallback_repo_id, values)


def freeze_backbone(backbone: torch.nn.Module) -> torch.nn.Module:
    backbone.requires_grad_(False)
    return backbone.eval()


def get_dino_tokenizer(*, device: str | torch.device = "cuda", env: Mapping[str, str] | None = None) -> torch.nn.Module:
    from transformers import Dinov2Model

    source = _source(
        "DINO_V2_PATH",
        "dinov2_base",
        "dinov2_base_model_dir",
        "facebook/dinov2-base",
        env,
    )
    return freeze_backbone(Dinov2Model.from_pretrained(source).to(device))


def get_dinov2_vitb14_reg_tokenizer(
    *,
    device: str | torch.device = "cpu",
    checkpoint_path: str | os.PathLike[str] | None = None,
) -> torch.nn.Module:
    """Load the in-tree DINOv2 ViT-B/14 register-token backbone for UniVLA."""

    from worldfoundry.base_models.perception_core.general_perception.dinov2.hub.backbones import (
        dinov2_vitb14_reg,
    )

    configured = checkpoint_path or os.environ.get("UNIVLA_DINO_CKPT_PATH") or os.environ.get("DINO_V2_REG_PATH")
    legacy_path = Path("huggingface/dinov2_vitb14_reg4_pretrain.pth")
    if not configured and legacy_path.is_file():
        configured = legacy_path

    if configured:
        model = dinov2_vitb14_reg(pretrained=False)
        checkpoint = torch.load(Path(configured), map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        model.load_state_dict(state_dict, strict=True)
    else:
        model = dinov2_vitb14_reg(pretrained=True)
    return freeze_backbone(model.to(device))


def get_dinov3_tokenizer(
    *, device: str | torch.device = "cuda", env: Mapping[str, str] | None = None
) -> torch.nn.Module:
    from transformers import DINOv3ViTModel

    source = _source(
        "DINO_V3_PATH",
        "dinov3_base",
        "dinov3_base_model_dir",
        "facebook/dinov3-vitb16-pretrain-lvd1689m",
        env,
    )
    return freeze_backbone(DINOv3ViTModel.from_pretrained(source).to(device))


def get_siglip2_tokenizer(
    *, device: str | torch.device = "cuda", env: Mapping[str, str] | None = None
) -> torch.nn.Module:
    from transformers import AutoModel

    values = os.environ if env is None else env
    source = values.get("SIGLIP2_PATH") or values.get("WORLDFOUNDRY_SIGLIP2_MODEL_DIR")
    if not source:
        raise ValueError("Set SIGLIP2_PATH or WORLDFOUNDRY_SIGLIP2_MODEL_DIR to the SigLIP2 checkpoint directory.")
    model = AutoModel.from_pretrained(
        source.replace("hdfs:///", "/mnt/hdfs/"),
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).vision_model
    return freeze_backbone(model.to(device))


def get_dino_reps(input_tensor: torch.Tensor, encoder: torch.nn.Module) -> torch.Tensor:
    outputs = encoder(input_tensor.squeeze(2) * 2 - 1)
    tokens = outputs.last_hidden_state[:, 1:, :].detach()
    batch, _, dim = tokens.shape
    return tokens.reshape(batch, 1, 16, 16, dim)


def get_dinov3_reps(input_tensor: torch.Tensor, encoder: torch.nn.Module) -> torch.Tensor:
    outputs = encoder(input_tensor.squeeze(2) * 2 - 1)
    tokens = outputs.last_hidden_state[:, 5:, :].detach()
    batch, _, dim = tokens.shape
    return tokens.reshape(batch, 1, 14, 14, dim)


def get_siglip2_reps(input_tensor: torch.Tensor, encoder: torch.nn.Module) -> torch.Tensor:
    outputs = encoder(
        pixel_values=input_tensor.squeeze(2),
        output_hidden_states=False,
        return_dict=True,
    )
    tokens = outputs.last_hidden_state.detach()
    batch, _, dim = tokens.shape
    return tokens.reshape(batch, 1, 14, 14, dim)


def get_reps_magvit2(input_tensor: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    return model.encoder(input_tensor.squeeze(2)).detach()


def get_magvit2_tokenizer(
    model_type: str,
    *,
    device: str | torch.device = "cuda",
    config_path: str | os.PathLike[str] | None = None,
    checkpoint_path: str | os.PathLike[str] | None = None,
) -> torch.nn.Module:
    from omegaconf import OmegaConf

    # Importing the runtime establishes its compatibility alias for ``src``.
    from worldfoundry.synthesis.visual_generation.open_magvit2 import (
        open_magvit2_runtime as _open_magvit2_runtime,  # noqa: F401
    )
    from worldfoundry.synthesis.visual_generation.open_magvit2.open_magvit2_runtime.src.Open_MAGVIT2.models.lfqgan import (
        VQModel,
    )

    config_path = config_path or os.environ.get("MAGVIT2_CONFIG_PATH")
    checkpoint_path = checkpoint_path or os.environ.get("MAGVIT2_TOKENIZER_PATH")
    if not config_path or not checkpoint_path:
        raise ValueError(
            "MAGVIT2_CONFIG_PATH and MAGVIT2_TOKENIZER_PATH must point to the "
            "Open-MAGVIT2 feature-tokenizer config and checkpoint."
        )

    config = OmegaConf.load(Path(config_path))
    model = VQModel(**config.model.init_args, model_type=model_type)
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    return freeze_backbone(model.to(device))


__all__ = [
    "freeze_backbone",
    "get_dino_reps",
    "get_dino_tokenizer",
    "get_dinov2_vitb14_reg_tokenizer",
    "get_dinov3_reps",
    "get_dinov3_tokenizer",
    "get_magvit2_tokenizer",
    "get_reps_magvit2",
    "get_siglip2_reps",
    "get_siglip2_tokenizer",
]
