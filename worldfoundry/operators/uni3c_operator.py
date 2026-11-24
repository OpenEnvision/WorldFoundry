"""Input shaping for Uni3C camera and unified-motion controls."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .three_d_four_d_runtime_operator import ThreeDFourDRuntimeOperator


class Uni3COperator(ThreeDFourDRuntimeOperator):
    """Normalize the reference image and pre-rendered 3D control bundle."""

    INPUT_SCHEMA: Mapping[str, Any] = {
        "prompt": True,
        "image": True,
        "video": False,
        "actions": ["camera_trajectory", "human_motion"],
    }

    def __init__(self, input_schema: Mapping[str, Any] | None = None, **_: Any) -> None:
        super().__init__(input_schema={**self.INPUT_SCHEMA, **dict(input_schema or {})}, model_id="uni3c")


__all__ = ["Uni3COperator"]
