"""Input shaping for the MoVerse image-to-navigable-world runtime."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .three_d_four_d_runtime_operator import ThreeDFourDRuntimeOperator


class MoVerseOperator(ThreeDFourDRuntimeOperator):
    """Normalize an input image, prompt, and optional camera trajectory."""

    INPUT_SCHEMA: Mapping[str, Any] = {
        "prompt": True,
        "image": True,
        "video": False,
        "actions": ["camera_trajectory"],
    }

    def __init__(self, input_schema: Mapping[str, Any] | None = None, **_: Any) -> None:
        super().__init__(input_schema={**self.INPUT_SCHEMA, **dict(input_schema or {})}, model_id="moverse")


__all__ = ["MoVerseOperator"]
