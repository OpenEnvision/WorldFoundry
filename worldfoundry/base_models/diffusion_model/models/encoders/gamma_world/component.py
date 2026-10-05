"""Native Gamma-World prompt and action conditioning.

Gamma-World uses the shared Cosmos Reason1 text backbone
(:class:`~..cosmos2p5.component.Cosmos25PromptConditioner`).
This module renames generic ``context`` outputs to the tensor
names consumed by Gamma denoisers.

Action tracks are frame-aligned and normalized from
``DiffusionRequest``.  Default negative prompt is English;
user strings are not rewritten.

This is a ConditionEncoder adapter, not a new LLM.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from ....components import ComponentBuildContext
from ....contracts import Conditioning, DiffusionRequest
from ..cosmos2p5.component import Cosmos25PromptConditioner, build_cosmos25_prompt_conditioner


DEFAULT_NEGATIVE_PROMPT = (
    "The video contains static scenes, motion blur, low resolution, poor lighting, "
    "visual artifacts, flicker, jerky motion, or implausible transitions."
)


def _action_sources(request: DiffusionRequest) -> list[object]:
    value = request.inputs.get("actions", request.inputs.get("action_paths"))
    if value is None:
        return []
    if isinstance(value, (str, Path, Mapping)):
        return [value]
    if not isinstance(value, Sequence):
        raise TypeError("Gamma-World actions must be paths, mappings, or a sequence of them")
    return list(value)


class GammaWorldConditioner:
    """Adapt the shared Reason1 encoder to Gamma's multi-agent condition map."""

    def __init__(self, prompt_encoder: Cosmos25PromptConditioner) -> None:
        self.prompt_encoder = prompt_encoder

    def encode(
        self,
        request: DiffusionRequest,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Conditioning:
        encoded = self.prompt_encoder.encode(request, device=device, dtype=dtype)
        positive = {"crossattn_emb": encoded.positive["context"]}
        negative = (
            {"crossattn_emb": encoded.negative["context"]}
            if "context" in encoded.negative
            else {}
        )

        actions = []
        sources = _action_sources(request)
        if sources:
            from worldfoundry.core.io import action_track_tensors

            for source in sources:
                keyboard, camera = action_track_tensors(source, num_frames=request.num_frames)
                if camera is None:
                    camera = torch.zeros(
                        keyboard.shape[0], keyboard.shape[1], 2, dtype=keyboard.dtype
                    )
                actions.append(
                    {
                        "keyboard": keyboard.to(device=device),
                        "camera": camera.to(device=device),
                    }
                )

        shared: dict[str, object] = {
            "fps": torch.full(
                (request.batch_size,),
                float(request.inputs.get("fps", request.inputs.get("frame_rate", 16.0))),
                device=device,
                dtype=torch.float32,
            ),
        }
        if actions:
            shared["action_inputs"] = {"actions": actions}
        return Conditioning(positive=positive, negative=negative, shared=shared)


def build_gamma_world_conditioner(context: ComponentBuildContext) -> GammaWorldConditioner:
    """Build Gamma conditioning by reusing the canonical Reason1 component."""

    return GammaWorldConditioner(build_cosmos25_prompt_conditioner(context))


__all__ = [
    "DEFAULT_NEGATIVE_PROMPT",
    "GammaWorldConditioner",
    "build_gamma_world_conditioner",
]
