# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Explicit normalized robot actions and Cosmos-Reason1 text conditioning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ...components import ComponentBuildContext
from ...contracts import Conditioning, DiffusionRequest


def robot_actions(value: Any, *, num_frames: int | None = None) -> torch.Tensor:
    """Load normalized unified actions as ``[1,num_frames-1,384]``."""
    if isinstance(value, (str, Path)):
        path = Path(value).expanduser()
        if path.suffix == ".npy":
            import numpy as np

            value = np.load(path, allow_pickle=False)
        elif path.suffix == ".json":
            import json

            value = json.loads(path.read_text(encoding="utf-8"))
        else:
            value = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(value, dict):
        value = value["actions"]
    actions = torch.as_tensor(value, dtype=torch.float32)
    if actions.ndim == 2:
        actions = actions.unsqueeze(0)
    count = num_frames - 1 if num_frames is not None else (actions.shape[1] if actions.ndim == 3 else 0)
    if count <= 0 or tuple(actions.shape) != (1, count, 384) or not torch.isfinite(actions).all():
        raise ValueError(f"DreamDojo requires finite normalized actions with shape [1,{count},384]")
    return actions


class DreamDojoConditioner:
    """Combine actions with online Reason1 or supplied official text embeddings."""

    def __init__(self, text_encoder: Any = None, embeddings: torch.Tensor | None = None) -> None:
        self.text_encoder = text_encoder
        self.embeddings = embeddings

    def encode(self, request: DiffusionRequest, *, device: torch.device, dtype: torch.dtype) -> Conditioning:
        actions = robot_actions(request.inputs.get("actions"), num_frames=request.num_frames).to(
            device=device, dtype=dtype
        )
        if self.embeddings is not None:
            context = self.embeddings.to(device=device, dtype=dtype)
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context.ndim != 3 or context.shape[0] != 1 or context.shape[-1] != 100352:
                raise ValueError("DreamDojo Reason1 embeddings must have shape [1,S,100352]")
            positive = {"context": context}
            negative = {"context": torch.zeros_like(context)}
        else:
            encoded = self.text_encoder.encode(request, device=device, dtype=dtype)
            positive = dict(encoded.positive)
            negative = (
                dict(encoded.negative)
                if request.negative_prompt is not None
                else {"context": torch.zeros_like(positive["context"])}
            )
        return Conditioning(positive=positive, negative=negative, shared={"actions": actions})


def build_dreamdojo_conditioner(context: ComponentBuildContext) -> DreamDojoConditioner:
    """Reuse the Reason1 component or load explicit precomputed embeddings."""
    embedding_path = context.component_options.get("text_embeddings_path")
    if embedding_path is not None:
        embeddings = torch.load(Path(str(embedding_path)).expanduser(), map_location="cpu", weights_only=True)
        return DreamDojoConditioner(embeddings=embeddings)
    from .cosmos2p5 import build_cosmos25_prompt_conditioner

    return DreamDojoConditioner(build_cosmos25_prompt_conditioner(context))
