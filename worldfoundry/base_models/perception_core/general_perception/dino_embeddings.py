"""Shared DINO-family embedding loaders for benchmark metrics."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch

from worldfoundry.base_models.capabilities import get_base_model_capability


def _preferred_hf_model_source(
    capability_id: str,
    asset_id: str,
    fallback_repo_id: str,
    env: Mapping[str, str] | None = None,
) -> str:
    capability = get_base_model_capability(capability_id)
    for asset in capability.assets:
        if asset.id != asset_id:
            continue
        status = asset.check(env)
        matched_path = status.get("matched_path")
        if status.get("ready") and matched_path:
            return str(Path(matched_path).expanduser())
        break
    return fallback_repo_id


class HFVisionEmbeddingModel(torch.nn.Module):
    """Wrap a Hugging Face vision backbone with the torch.hub-style tensor API."""

    def __init__(self, model: torch.nn.Module, *, use_pooler_output: bool = True):
        super().__init__()
        self.model = model
        self.use_pooler_output = use_pooler_output

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=pixel_values)
        pooler_output = getattr(outputs, "pooler_output", None)
        if self.use_pooler_output and pooler_output is not None:
            return pooler_output
        last_hidden_state = getattr(outputs, "last_hidden_state", None)
        if last_hidden_state is None:
            raise TypeError("vision embedding model output has no pooler_output or last_hidden_state")
        return last_hidden_state[:, 0]


def load_dino_vitb16_feature_model(
    *,
    env: Mapping[str, str] | None = None,
    device: str | torch.device | None = None,
) -> HFVisionEmbeddingModel:
    from transformers import ViTModel

    source = _preferred_hf_model_source("dino_vitb16", "dino_vitb16_model_dir", "facebook/dino-vitb16", env)
    model = HFVisionEmbeddingModel(ViTModel.from_pretrained(source, add_pooling_layer=False), use_pooler_output=False)
    if device is not None:
        model = model.to(device)
    return model.eval()


def load_dinov2_feature_model(
    checkpoint: str | Path,
    *,
    device: str | torch.device | None = None,
    use_pooler_output: bool = False,
    local_files_only: bool = True,
) -> HFVisionEmbeddingModel:
    """Load a selected DINOv2 checkpoint with explicit CLS/pooler semantics."""
    from transformers import AutoModel

    model = HFVisionEmbeddingModel(
        AutoModel.from_pretrained(str(checkpoint), local_files_only=local_files_only),
        use_pooler_output=use_pooler_output,
    )
    if device is not None:
        model = model.to(device)
    return model.eval()


def load_dinov2_base_feature_model(
    *,
    env: Mapping[str, str] | None = None,
    device: str | torch.device | None = None,
) -> HFVisionEmbeddingModel:
    from transformers import Dinov2Model

    source = _preferred_hf_model_source("dinov2_base", "dinov2_base_model_dir", "facebook/dinov2-base", env)
    model = HFVisionEmbeddingModel(Dinov2Model.from_pretrained(source))
    if device is not None:
        model = model.to(device)
    return model.eval()
