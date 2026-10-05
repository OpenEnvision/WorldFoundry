"""Environment-token encoder shared by native Wan world-model variants.

:class:`WanVGGTEnvironmentEncoder` loads a VGGT-1B backbone
and emits environment tokens for world-model recipes.
:func:`resolve_vggt_1b_root` / :func:`load_vggt_1b_backbone`
locate weights.

Outputs join Wan ``context`` / control maps; they are not
UMT5 prompt tokens.  Prompt encode stays on
:class:`~.component.WanTextConditioner`.

World-model extra, not required for plain T2V.
"""

from __future__ import annotations

import json
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Mapping

import torch
from einops import rearrange
from safetensors import safe_open
from torch import nn

from worldfoundry.base_models.three_dimensions.point_clouds.vggt.vggt.models.vggt import (
    VGGT,
)
from worldfoundry.core.io.paths import worldfoundry_path_tokens
from worldfoundry.core.model_loading import load_model, load_state_dict


def resolve_vggt_1b_root(path: str | Path | None = None) -> Path:
    """Resolve a local VGGT-1B checkout, downloading only when necessary."""

    candidates: list[Path] = []
    if path is not None and str(path).strip():
        candidates.append(Path(path).expanduser())
    for name in ("WORLDFOUNDRY_VGGT_1B_MODEL_DIR", "WORLDFOUNDRY_VGGT_1B_ROOT"):
        value = os.environ.get(name)
        if value:
            candidates.append(Path(value).expanduser())
    tokens = worldfoundry_path_tokens()
    checkpoint_root = Path(tokens["WORLDFOUNDRY_CKPT_DIR"]).expanduser()
    candidates.extend(
        (
            checkpoint_root / "VGGT-1B",
            checkpoint_root / "hfd" / "facebook--VGGT-1B",
            checkpoint_root / "facebook--VGGT-1B",
        )
    )
    for candidate in candidates:
        if (candidate / "config.json").is_file() and any(
            (candidate / filename).is_file()
            for filename in ("model.safetensors", "model.pt", "pytorch_model.bin")
        ):
            return candidate.resolve()
    if path is not None and str(path).strip():
        raise FileNotFoundError(f"VGGT-1B checkpoint directory is incomplete: {path}")

    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id="facebook/VGGT-1B")).resolve()


def _vggt_aggregator_state(
    root: Path,
    *,
    torch_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    safetensors_path = root / "model.safetensors"
    if safetensors_path.is_file():
        state: dict[str, torch.Tensor] = {}
        with safe_open(str(safetensors_path), framework="pt", device="cpu") as checkpoint:
            for name in checkpoint.keys():
                if name.startswith("aggregator."):
                    state[name] = checkpoint.get_tensor(name).to(dtype=torch_dtype)
        return state
    for filename in ("model.pt", "pytorch_model.bin"):
        checkpoint_path = root / filename
        if checkpoint_path.is_file():
            return {
                name: value.to(dtype=torch_dtype)
                for name, value in load_state_dict(str(checkpoint_path), device="cpu").items()
                if name.startswith("aggregator.") and torch.is_tensor(value)
            }
    raise FileNotFoundError(f"VGGT-1B model weights not found under {root}")


def load_vggt_1b_backbone(
    checkpoint_root: str | Path | None = None,
    *,
    torch_dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cpu",
) -> VGGT:
    """Load only VGGT's aggregator, omitting four unused prediction heads."""

    root = resolve_vggt_1b_root(checkpoint_root)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    config.update(
        enable_camera=False,
        enable_point=False,
        enable_depth=False,
        enable_track=False,
    )
    return load_model(
        VGGT,
        path=None,
        config=config,
        state_dict=_vggt_aggregator_state(root, torch_dtype=torch_dtype),
        torch_dtype=torch_dtype,
        device=device,
        strict=True,
    ).requires_grad_(False)


class WanVGGTEnvironmentEncoder(nn.Module):
    """Project frozen VGGT scene tokens into the Wan hidden width."""

    def __init__(
        self,
        backbone: VGGT,
        input_dim: int = 2048,
        output_dim: int = 3072,
    ) -> None:
        super().__init__()
        self.backbone = backbone.eval().requires_grad_(False)
        self.connector = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, output_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 6:
            raise ValueError("MultiWorld environment observations must have shape [B,F,K,3,H,W]")
        batch, frames = images.shape[:2]
        images = rearrange(images, "b f k c h w -> (b f) k c h w").contiguous()
        device_type = images.device.type
        compute_dtype = next(self.backbone.parameters()).dtype
        autocast = (
            torch.autocast(device_type=device_type, dtype=compute_dtype)
            if device_type in {"cuda", "xpu", "npu"}
            else nullcontext()
        )
        with torch.no_grad(), autocast:
            hidden_states, _ = self.backbone.aggregator(images)
            # Each iteration is [(B F), K, N, 2 * embed_dim].
            hidden_states = [hidden.mean(dim=1) for hidden in hidden_states]
            hidden = torch.stack(hidden_states, dim=1).mean(dim=1)
        hidden = rearrange(hidden, "(b f) n d -> (b f n) d", b=batch, f=frames)
        connector_dtype = self.connector[0].weight.dtype
        hidden = self.connector(hidden.to(dtype=connector_dtype))
        return rearrange(hidden, "(b f n) d -> b (f n) d", b=batch, f=frames)


def load_wan_vggt_environment_encoder(
    connector_state: Mapping[str, torch.Tensor],
    *,
    checkpoint_root: str | Path | None = None,
    input_dim: int = 2048,
    output_dim: int = 3072,
    torch_dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cpu",
) -> WanVGGTEnvironmentEncoder:
    """Assemble the frozen VGGT role and released MultiWorld connector."""

    backbone = load_vggt_1b_backbone(
        checkpoint_root,
        torch_dtype=torch_dtype,
        device=device,
    )
    encoder = WanVGGTEnvironmentEncoder(
        backbone,
        input_dim=input_dim,
        output_dim=output_dim,
    )
    incompatible = encoder.connector.load_state_dict(dict(connector_state), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(f"MultiWorld environment connector mismatch: {incompatible}")
    return encoder.to(device=device, dtype=torch_dtype).eval()


__all__ = [
    "WanVGGTEnvironmentEncoder",
    "load_vggt_1b_backbone",
    "load_wan_vggt_environment_encoder",
    "resolve_vggt_1b_root",
]
