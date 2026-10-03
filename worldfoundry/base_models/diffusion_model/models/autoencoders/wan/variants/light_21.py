"""Matched pruned Wan2.1 LightVAE architecture and strict checkpoint admission.

The LightX2V ``lightvaew2_1.pth`` student uses the native Wan2.1 block layout
at base width 24, with 16 latent channels and the teacher's normalization.
Matching shapes proves architectural compatibility, not teacher-image parity.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache

import torch

from ..model import WanVideoVAE

LIGHTVAE_WAN21_VARIANT = "lightvae-wan21"
LIGHTVAE_WAN21_CHECKPOINT = "https://huggingface.co/lightx2v/Autoencoders/resolve/main/lightvaew2_1.pth"


class Wan21LightVAE(WanVideoVAE):
    """Native 24/48/96-channel student, retaining Wan16 latent/cache semantics."""

    codec_variant = LIGHTVAE_WAN21_VARIANT
    codec_base_dim = 24
    codec_pruning_rate = 0.75

    def __init__(self, z_dim=16, vae_pretrained_path: str | None = None):
        if z_dim != 16:
            raise ValueError("Wan2.1 LightVAE requires exactly 16 latent channels")
        super().__init__(z_dim, vae_pretrained_path, base_dim=self.codec_base_dim)


@lru_cache(maxsize=1)
def _expected_shapes() -> Mapping[str, tuple[int, ...]]:
    with torch.device("meta"):
        expected = Wan21LightVAE().state_dict()
    return {name: tuple(value.shape) for name, value in expected.items()}


def convert_lightvae_wan21_state_dict(state_dict: Mapping[str, object]) -> Mapping[str, torch.Tensor]:
    """Require every matched student tensor before the generic loader restores it.

    Accept the checkpoint's raw keys or the native wrapper's ``model.`` keys.
    Missing, extra, malformed and wrong-width tensors fail before allocation
    of the loaded model; teacher or Wan2.2 checkpoints cannot be substituted.
    """
    if "model_state" in state_dict:
        state_dict = state_dict["model_state"]
    if not isinstance(state_dict, Mapping):
        raise TypeError("Wan2.1 LightVAE checkpoint must contain a tensor mapping")
    converted = {}
    for name, value in state_dict.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise ValueError("Wan2.1 LightVAE checkpoint entries must be floating-point tensors with string keys")
        target = name if name.startswith("model.") else "model." + name
        if target in converted:
            raise ValueError(f"Wan2.1 LightVAE checkpoint has duplicate canonical tensor {target}")
        converted[target] = value
    expected = _expected_shapes()
    missing = sorted(expected.keys() - converted.keys())
    extra = sorted(converted.keys() - expected.keys())
    wrong_shapes = [
        name for name in expected.keys() & converted.keys() if tuple(converted[name].shape) != expected[name]
    ]
    if missing or extra or wrong_shapes:
        raise ValueError(
            "checkpoint does not match the Wan2.1 LightVAE base24/16-latent architecture: "
            f"missing={missing[:3]}, extra={extra[:3]}, wrong_shapes={sorted(wrong_shapes)[:3]}; "
            "use the matching lightvaew2_1.pth student, not teacher Wan2.1 or Wan2.2 weights"
        )
    return converted


__all__ = [
    "LIGHTVAE_WAN21_CHECKPOINT",
    "LIGHTVAE_WAN21_VARIANT",
    "Wan21LightVAE",
    "convert_lightvae_wan21_state_dict",
]
