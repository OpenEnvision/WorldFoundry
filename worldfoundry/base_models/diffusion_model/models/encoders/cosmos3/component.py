"""Cosmos3 tokenizer-only :class:`~...contracts.ConditionEncoder`.

Does **not** run a separate text tower: the omni transformer embeds
``input_ids`` itself.  ``encode`` returns ``Conditioning.positive`` /
``negative`` with ``input_ids`` and optional ``action_domain_id``.
System / duration / FPS templates are applied in English; user prompt
strings are not rewritten.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from transformers import AutoTokenizer

from ....components import ComponentBuildContext
from ....contracts import Conditioning, DiffusionRequest
from ....loaders import NativeCheckpointResolver
from ...representations.cosmos3 import ACTION_DOMAIN_IDS

_IMAGE_SYSTEM_PROMPT = "You are a helpful assistant who will generate images from a give prompt."
_VIDEO_SYSTEM_PROMPT = "You are a helpful assistant who will generate videos from a give prompt."


class Cosmos3PromptConditioner:
    """Tokenize Cosmos3 prompts; text embeddings remain inside the omni transformer."""

    def __init__(self, tokenizer, *, use_system_prompt: bool = True) -> None:
        self.tokenizer = tokenizer
        self.use_system_prompt = bool(use_system_prompt)
        self.start_of_generation_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
        if self.start_of_generation_id is None or tokenizer.eos_token_id is None:
            raise ValueError("Cosmos3 tokenizer is missing required generation tokens")

    @staticmethod
    def _append(base: str, addition: str) -> str:
        base = base.rstrip(".")
        return f"{base}. {addition}" if base else addition

    def _template(self, text: str, request: DiffusionRequest, *, negative: bool) -> str:
        fps = float(request.inputs.get("fps", request.inputs.get("frame_rate", 24.0)))
        is_image = request.num_frames == 1
        if not is_image and bool(request.inputs.get("add_duration_template", True)):
            duration = request.num_frames / fps
            if negative:
                addition = f"The video is not {duration:.1f} seconds long and is not of {fps:.0f} FPS."
            else:
                addition = f"The video is {duration:.1f} seconds long and is of {fps:.0f} FPS."
            text = self._append(text, addition)
        if bool(request.inputs.get("add_resolution_template", True)):
            noun = "image" if is_image else "video"
            verb = "is not" if negative else "is"
            text = self._append(text, f"This {noun} {verb} of {request.height}x{request.width} resolution.")
        return text

    def _tokenize(self, text: str, *, is_image: bool) -> torch.Tensor:
        conversation: list[dict[str, str]] = []
        if self.use_system_prompt:
            conversation.append(
                {"role": "system", "content": _IMAGE_SYSTEM_PROMPT if is_image else _VIDEO_SYSTEM_PROMPT}
            )
        conversation.append({"role": "user", "content": text})
        encoded = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            add_vision_id=False,
            return_dict=True,
        )
        ids = list(encoded.input_ids)
        ids.extend((self.tokenizer.eos_token_id, self.start_of_generation_id))
        return torch.tensor(ids, dtype=torch.long)

    @staticmethod
    def _single(values: Sequence[str], *, name: str) -> str:
        if len(values) != 1:
            raise ValueError(f"Cosmos3 currently supports one {name} per joint sequence")
        return values[0]

    def encode(
        self,
        request: DiffusionRequest,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Conditioning:
        """Return tokenized ``input_ids`` (and optional ``action_domain_id``)."""
        del dtype
        prompt = self._single(request.prompts, name="prompt")
        positive = self._tokenize(
            self._template(prompt, request, negative=False),
            is_image=request.num_frames == 1,
        ).to(device)
        negative: dict[str, torch.Tensor] = {}
        if request.sampling.guidance_scale != 1.0:
            negative_prompt = self._single(request.negative_prompts or ("",), name="negative prompt")
            negative["input_ids"] = self._tokenize(
                self._template(negative_prompt, request, negative=True),
                is_image=request.num_frames == 1,
            ).to(device)

        shared: dict[str, object] = {"fps": float(request.inputs.get("fps", request.inputs.get("frame_rate", 24.0)))}
        domain_id = request.inputs.get("action_domain_id")
        domain_name = request.inputs.get("action_domain_name", request.inputs.get("domain_name"))
        if domain_id is None and domain_name is not None:
            try:
                domain_id = ACTION_DOMAIN_IDS[str(domain_name)]
            except KeyError as error:
                raise ValueError(f"unknown Cosmos3 action domain: {domain_name!r}") from error
        if domain_id is not None:
            shared["action_domain_id"] = torch.as_tensor(domain_id, dtype=torch.long, device=device).reshape(())
        return Conditioning(
            positive={"input_ids": positive},
            negative=negative,
            shared=shared,
        )


def build_cosmos3_prompt_conditioner(context: ComponentBuildContext) -> Cosmos3PromptConditioner:
    checkpoint = NativeCheckpointResolver().materialize(context.require_checkpoint("tokenizer"))
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint.directory("text_tokenizer"),
        local_files_only=True,
        trust_remote_code=False,
    )
    return Cosmos3PromptConditioner(
        tokenizer,
        use_system_prompt=bool(context.component_options.get("use_system_prompt", True)),
    )


__all__ = ["Cosmos3PromptConditioner", "build_cosmos3_prompt_conditioner"]
