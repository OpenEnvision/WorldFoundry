"""StepVideo :class:`~...contracts.ConditionEncoder` (STEP1 LLM + Hunyuan CLIP).

``encode`` concatenates optional ``positive_magic`` / ``negative_magic``
suffixes onto ``DiffusionRequest`` prompts, then returns
``prompt_embeds``, ``clip_embeds``, and ``attention_mask``.  Shared
``fps`` is forwarded to the denoiser.  Magic strings are not rewritten.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ....components import ComponentBuildContext
from ....contracts import Conditioning, DiffusionRequest
from ....loaders import NativeCheckpointResolver
from ....optimizations import OffloadMode
from .clip import HunyuanClip
from .step_llm import STEP1TextEncoder


class StepVideoPromptConditioner:
    """Dual-encoder conditioner: STEP1 hidden states + CLIP tokens."""

    def __init__(self, llm: STEP1TextEncoder, clip: HunyuanClip) -> None:
        self.llm = llm
        self.clip = clip

    @torch.no_grad()
    def _branch(self, prompts: Sequence[str]) -> dict[str, torch.Tensor]:
        prompt_embeds, mask = self.llm(list(prompts))
        clip_embeds, _ = self.clip(list(prompts))
        mask = torch.nn.functional.pad(mask, (clip_embeds.shape[1], 0), value=1)
        return {
            "prompt_embeds": prompt_embeds,
            "clip_embeds": clip_embeds.to(dtype=prompt_embeds.dtype),
            "attention_mask": mask,
        }

    def encode(self, request, *, device, dtype):
        """Return LLM + CLIP tensors for both CFG branches."""
        positive_magic = str(request.inputs.get("positive_magic", request.inputs.get("pos_magic", "")))
        negative_magic = str(request.inputs.get("negative_magic", request.inputs.get("neg_magic", "")))
        positive = tuple(prompt + positive_magic for prompt in request.prompts)
        negative_prompts = request.negative_prompts
        negative = (
            tuple(prompt + negative_magic for prompt in negative_prompts)
            if negative_prompts is not None
            else (negative_magic,) * request.batch_size
        )
        shared = {"fps": int(request.inputs.get("fps", 25))}
        return Conditioning(
            positive=self._branch(positive),
            negative=self._branch(negative),
            shared=shared,
        )


def build_step_video_prompt_conditioner(context: ComponentBuildContext) -> StepVideoPromptConditioner:
    root = NativeCheckpointResolver().materialize(context.require_checkpoint("resources")).root
    encoder_device = (
        torch.device(context.policy.offload.target)
        if context.policy.offload.mode is not OffloadMode.NONE
        else context.policy.device
    )
    llm = STEP1TextEncoder(root / "step_llm", max_length=320).to(
        device=encoder_device,
        dtype=context.policy.dtype,
    ).eval()
    clip = HunyuanClip(root / "hunyuan_clip", max_length=77).to(
        device=encoder_device,
        dtype=context.policy.dtype,
    ).eval()
    return StepVideoPromptConditioner(llm, clip)


__all__ = ["StepVideoPromptConditioner", "build_step_video_prompt_conditioner"]
