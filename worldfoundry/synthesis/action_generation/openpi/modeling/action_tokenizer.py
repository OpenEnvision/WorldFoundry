"""Local FAST universal action tokenizer used by OpenPI inference.

The algorithm is adapted from Physical Intelligence's Apache-2.0 FAST
processor (revision ``fef8e13``).  Only checkpoint data is loaded at runtime;
checkpoint-provided Python is never imported or executed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.fft import dct, idct
from transformers import AutoTokenizer

from worldfoundry.core.io.paths import resolve_local_hf_model_path


class UniversalActionTokenizer:
    """DCT, quantization, and BPE codec for continuous action chunks."""

    def __init__(
        self,
        bpe_tokenizer: Any,
        *,
        scale: float = 10.0,
        vocab_size: int = 2048,
        min_token: int = -354,
        action_dim: int | None = None,
        time_horizon: int | None = None,
    ) -> None:
        self.bpe_tokenizer = bpe_tokenizer
        self.scale = float(scale)
        self.vocab_size = int(vocab_size)
        self.min_token = int(min_token)
        self.action_dim = action_dim
        self.time_horizon = time_horizon
        self.called_action_dim = action_dim
        self.called_time_horizon = time_horizon

    @classmethod
    def from_pretrained(cls, location: str | Path) -> UniversalActionTokenizer:
        """Load tokenizer assets from WorldFoundry-local storage only."""
        local_dir = resolve_local_hf_model_path(
            location,
            required_files=("processor_config.json", "tokenizer.json", "tokenizer_config.json"),
        )
        config = json.loads((local_dir / "processor_config.json").read_text(encoding="utf-8"))
        tokenizer = AutoTokenizer.from_pretrained(
            local_dir,
            trust_remote_code=False,
            local_files_only=True,
        )
        return cls(
            tokenizer,
            scale=float(config.get("scale", 10.0)),
            vocab_size=int(config.get("vocab_size", 2048)),
            min_token=int(config.get("min_token", -354)),
            action_dim=config.get("action_dim"),
            time_horizon=config.get("time_horizon"),
        )

    def __call__(self, action_chunk: np.ndarray) -> list[list[int]]:
        actions = np.asarray(action_chunk)
        if actions.ndim == 2:
            actions = actions[None, ...]
        if actions.ndim != 3:
            raise ValueError(f"actions must have shape [B,T,D] or [T,D], got {actions.shape}")
        self.called_time_horizon = int(actions.shape[-2])
        self.called_action_dim = int(actions.shape[-1])
        coefficients = np.rint(dct(actions, axis=1, norm="ortho") * self.scale)
        encoded: list[list[int]] = []
        for sample in coefficients:
            text = "".join(map(chr, np.maximum(sample.reshape(-1) - self.min_token, 0).astype(int)))
            encoded.append(list(self.bpe_tokenizer(text)["input_ids"]))
        return encoded

    def decode(
        self,
        tokens: list[list[int]],
        *,
        time_horizon: int | None = None,
        action_dim: int | None = None,
    ) -> np.ndarray:
        horizon = time_horizon or self.time_horizon or self.called_time_horizon
        dimension = action_dim or self.action_dim or self.called_action_dim
        if horizon is None or dimension is None:
            raise ValueError("time_horizon and action_dim are required before decoding actions")
        self.called_time_horizon = self.time_horizon = int(horizon)
        self.called_action_dim = self.action_dim = int(dimension)

        decoded_actions: list[np.ndarray] = []
        for token_ids in tokens:
            decoded_value_count: int | None = None
            try:
                text = self.bpe_tokenizer.decode(token_ids)
                coefficients = np.asarray(list(map(ord, text)), dtype=np.float64) + self.min_token
                decoded_value_count = int(coefficients.size)
                coefficients = coefficients.reshape(self.time_horizon, self.action_dim)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "FAST action token decoding failed: "
                    f"token_count={len(token_ids)}, decoded_value_count={decoded_value_count}, "
                    f"time_horizon={self.time_horizon}, action_dim={self.action_dim}."
                ) from exc
            decoded_actions.append(idct(coefficients / self.scale, axis=0, norm="ortho"))
        return np.stack(decoded_actions).astype(np.float32, copy=False)


__all__ = ["UniversalActionTokenizer"]
