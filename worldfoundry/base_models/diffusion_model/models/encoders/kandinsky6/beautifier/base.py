"""Switchable prompt beautifiers. One class per method."""

from __future__ import annotations

from typing import Protocol


class PromptBeautifier(Protocol):
    name: str

    def expand(
        self,
        prompt: str,
        *,
        image: object | None = None,
        audio: bool = False,
        seed: int | None = None,
        text_embedder: object | None = None,
    ) -> str: ...


class NullBeautifier:
    """Leave the prompt unchanged."""

    name = "none"

    def expand(
        self,
        prompt: str,
        *,
        image: object | None = None,
        audio: bool = False,
        seed: int | None = None,
        text_embedder: object | None = None,
    ) -> str:
        del image, audio, seed, text_embedder
        return prompt
