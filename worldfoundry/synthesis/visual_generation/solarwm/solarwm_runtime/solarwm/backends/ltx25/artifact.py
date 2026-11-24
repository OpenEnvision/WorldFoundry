"""Exact trainer-facing schema for LTX-2.5 artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import numpy as np

from .geometry import STABLE_GEOMETRY

PREENCODE_SCHEMA: Final = "solarwm_ltx25_video_preencoded_v1"
PREENCODE_VERSION: Final = "solarwm_ltx25_video_153f_h512_w768_v1"
READER_CONTRACT: Final = "ltx25.v1"

_DTYPE_BYTES = MappingProxyType({"BF16": 2, "F32": 4, "I64": 8})


@dataclass(frozen=True)
class TensorSpec:
    dtype: str
    shape: tuple[int, ...]

    @property
    def nbytes(self) -> int:
        count = int(np.prod(self.shape, dtype=np.int64))
        return count * _DTYPE_BYTES[self.dtype]


TENSOR_SPECS: Final = MappingProxyType(
    {
        "video_latent": TensorSpec("BF16", STABLE_GEOMETRY.latent_shape),
        "first_frame_latent": TensorSpec("BF16", STABLE_GEOMETRY.first_frame_latent_shape),
        "video_prompt_embeds": TensorSpec("BF16", (1024, 4096)),
        "prompt_attention_mask": TensorSpec("I64", (1024,)),
        "relative_w2c": TensorSpec("F32", (20, 4, 4)),
        "camera_K": TensorSpec("F32", (20, 3, 3)),
        "source_indices": TensorSpec("I64", (153,)),
        "camera_source_indices": TensorSpec("I64", (20,)),
    }
)


__all__ = [
    "PREENCODE_SCHEMA",
    "PREENCODE_VERSION",
    "READER_CONTRACT",
    "TENSOR_SPECS",
    "TensorSpec",
]
