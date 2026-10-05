# Copyright (C) 2026 Tencent. Adapted for WorldFoundry inference.
# Source revision: 7e57e4d0448e8661eb6ae078f53e19638bd07102
# Academic-use terms: THIRD-PARTY-NOTICES (WorldCrafter)
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class RepEncoderConfig:
    format: str = "worldcrafter_repencoder_config_v1"
    source_views: int = 9
    target_views: int = 4
    latent_channels: int = 16
    latent_height: int = 48
    latent_width: int = 80
    dino_dim: int = 1024
    dino_input_block: int = 2
    dino_depth: int = 24
    dino_heads: int = 16
    latent_special_tokens: int = 5
    patch_height: int = 22
    patch_width: int = 37
    vggt_depth: int = 24
    vggt_heads: int = 16
    vggt_register_tokens: int = 4
    scene_dim: int = 768
    repfeature_depth: int = 10
    repfeature_heads: int = 12
    target_patch_size: int = 8
    ray_height: int = 384
    ray_width: int = 640
    output_kernel: tuple[int, int, int] = (3, 3, 3)
    target_slots: tuple[int, int, int, int] = (2, 4, 6, 8)

    def __post_init__(self) -> None:
        if (
            self.format != "worldcrafter_repencoder_config_v1"
            or self.source_views != 9
            or self.target_views != 4
            or self.latent_channels != 16
            or (self.latent_height, self.latent_width) != (48, 80)
            or self.dino_dim != 1024
            or (self.dino_input_block, self.dino_depth, self.dino_heads) != (2, 24, 16)
            or self.latent_special_tokens != 5
            or (self.patch_height, self.patch_width) != (22, 37)
            or (self.vggt_depth, self.vggt_heads, self.vggt_register_tokens)
            != (24, 16, 4)
            or self.scene_dim != 768
            or (self.repfeature_depth, self.repfeature_heads) != (10, 12)
            or self.target_patch_size != 8
            or (self.ray_height, self.ray_width) != (384, 640)
            or tuple(self.output_kernel) != (3, 3, 3)
            or tuple(self.target_slots) != (2, 4, 6, 8)
        ):
            raise ValueError("RepEncoder v1 has a frozen architecture")

    @property
    def patch_tokens(self) -> int:
        return int(self.patch_height * self.patch_width)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["output_kernel"] = list(self.output_kernel)
        value["target_slots"] = list(self.target_slots)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RepEncoderConfig":
        payload = dict(value)
        if "output_kernel" in payload:
            payload["output_kernel"] = tuple(int(x) for x in payload["output_kernel"])
        if "target_slots" in payload:
            payload["target_slots"] = tuple(int(x) for x in payload["target_slots"])
        config = cls(**payload)
        if config.to_dict() != cls().to_dict():
            raise ValueError("RepEncoder config does not match the frozen v1 contract")
        return config


__all__ = ["RepEncoderConfig"]
