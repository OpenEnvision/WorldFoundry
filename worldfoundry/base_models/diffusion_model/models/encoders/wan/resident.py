"""Resident Wan 2.1 prompt-conditioning wrapper (path-loaded UMT5).

:class:`WanTextEncoder` here loads UMT5 from a filesystem
path for resident / camera-control processes that do not go
through :class:`~.component.WanTextConditioner`.

Token / hidden-size contract matches :mod:`.model`.
Prefer the native component in new recipes.

Compatibility wrapper only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import nn

from .model import HuggingfaceTokenizer
from .reference import umt5_xxl


class WanTextEncoder(nn.Module):
    """Path-loaded UMT5 wrapper used by resident Wan 2.1 processes."""
    def __init__(
        self,
        model_root: str | Path | None = None,
        *,
        text_encoder_path: str | Path | None = None,
        tokenizer_path: str | Path | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if model_root is not None:
            root = Path(model_root)
            text_encoder_path = text_encoder_path or root / "models_t5_umt5-xxl-enc-bf16.pth"
            tokenizer_path = tokenizer_path or root / "google" / "umt5-xxl"
        if text_encoder_path is None or tokenizer_path is None:
            raise ValueError("WanTextEncoder requires model_root or explicit checkpoint paths")
        self.text_encoder = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=dtype,
            device=torch.device("cpu"),
        ).eval().requires_grad_(False)
        # The UMT5 encoder checkpoint is a pure tensor state dict, so the safe
        # weights-only deserialization path is sufficient
        # (plan/code_review/11_vendored_integration.md [VI-23]).
        self.text_encoder.load_state_dict(
            torch.load(text_encoder_path, map_location="cpu", weights_only=True)
        )
        self.tokenizer = HuggingfaceTokenizer(
            name=str(tokenizer_path),
            seq_len=512,
            clean="whitespace",
        )

    @property
    def device(self) -> torch.device:
        return next(self.text_encoder.parameters()).device

    def forward(self, text_prompts: Sequence[str]) -> dict[str, torch.Tensor]:
        ids, mask = self.tokenizer(
            list(text_prompts),
            return_mask=True,
            add_special_tokens=True,
        )
        ids, mask = ids.to(self.device), mask.to(self.device)
        lengths = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)
        for embedding, length in zip(context, lengths):
            embedding[length:] = 0.0
        return {"prompt_embeds": context}


__all__ = ["WanTextEncoder"]
