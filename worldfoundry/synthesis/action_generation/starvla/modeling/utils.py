"""Small inference-only utilities for StarVLA."""

from __future__ import annotations

import logging
from collections.abc import MutableMapping
from logging import LoggerAdapter
from typing import Any, ClassVar

from PIL import Image


class _ContextAdapter(LoggerAdapter):
    _PREFIXES: ClassVar[dict[int, str]] = {0: "[*] ", 1: "    |=> ", 2: "        |=> "}

    def process(self, msg: str, kwargs: MutableMapping[str, Any]) -> tuple[str, MutableMapping[str, Any]]:
        level = int(kwargs.pop("ctx_level", 0))
        return f"{self._PREFIXES.get(level, '')}{msg}", kwargs


def initialize_logger(name: str) -> _ContextAdapter:
    return _ContextAdapter(logging.getLogger(name), extra={})


def resize_images(images: Any, target_size: int | tuple[int, int] = (224, 224)) -> Any:
    if isinstance(target_size, int):
        target_size = (target_size, target_size)
    if isinstance(images, Image.Image):
        return images.resize(target_size, Image.Resampling.BICUBIC)
    if isinstance(images, list):
        return [resize_images(item, target_size) for item in images]
    if isinstance(images, tuple):
        return tuple(resize_images(item, target_size) for item in images)
    raise ValueError(f"Unsupported StarVLA image type: {type(images)!r}")


__all__ = ["initialize_logger", "resize_images"]
