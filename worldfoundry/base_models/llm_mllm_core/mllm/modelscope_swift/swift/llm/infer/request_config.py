# Copyright (c) Alibaba, Inc. and its affiliates.
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RequestConfig:
    """Generation options shared by local and server-backed inference engines."""

    max_tokens: int | None = None
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    repetition_penalty: float | None = None
    num_beams: int = 1
    stop: list[str] | None = field(default_factory=list)

    seed: int | None = None
    stream: bool = False
    logprobs: bool = False
    top_logprobs: int | None = None

    n: int = 1
    best_of: int | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    length_penalty: float = 1.0

    def __post_init__(self) -> None:
        if self.stop is None:
            self.stop = []


__all__ = ["RequestConfig"]
