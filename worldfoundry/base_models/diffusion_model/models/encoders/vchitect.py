"""Vchitect-2B :class:`~...contracts.ConditionEncoder` (CLIP + CLIP + T5).

Mirrors SD3: two CLIP-with-projection towers plus T5-XXL.
``encode`` returns ``prompt_embeds`` (concatenated sequence)
and ``pooled_prompt_embeds`` for
:class:`~...denoisers.vchitect.VchitectDenoiser`.

:func:`build_vchitect_prompt_conditioner` is the recipe
factory.  Latent decode is the SD3 frame VAE
(:mod:`~...autoencoders.sd3`).

SD3-family conditioning, not a video LLM encoder.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ...components import ComponentBuildContext
from ...contracts import Conditioning, DiffusionRequest
from ...loaders import NativeCheckpointResolver


class VchitectPromptConditioner:
    """Load the three Hugging Face text towers from a local snapshot."""

    def __init__(self, root, *, device: torch.device, dtype: torch.dtype) -> None:
        from transformers import CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

        self.device = device
        # The released pipeline loads all three text encoders in float32 and
        # applies bfloat16 autocast during inference. The sequence embedding
        # remains float32 while CLIP pooled vectors are bfloat16.
        self.dtype = torch.float32
        self.tokenizer = CLIPTokenizer.from_pretrained(root, subfolder="tokenizer", local_files_only=True)
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(root, subfolder="tokenizer_2", local_files_only=True)
        self.tokenizer_3 = T5TokenizerFast.from_pretrained(root, subfolder="tokenizer_3", local_files_only=True)
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(
            root,
            subfolder="text_encoder",
            local_files_only=True,
            torch_dtype=torch.float32,
        ).to(device)
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            root,
            subfolder="text_encoder_2",
            local_files_only=True,
            torch_dtype=torch.float32,
        ).to(device)
        self.text_encoder_3 = T5EncoderModel.from_pretrained(
            root,
            subfolder="text_encoder_3",
            local_files_only=True,
            torch_dtype=torch.float32,
        ).to(device)
        for model in (self.text_encoder, self.text_encoder_2, self.text_encoder_3):
            model.eval().requires_grad_(False)

    @torch.no_grad()
    def _clip(self, prompts: Sequence[str], *, second: bool) -> tuple[torch.Tensor, torch.Tensor]:
        tokenizer = self.tokenizer_2 if second else self.tokenizer
        encoder = self.text_encoder_2 if second else self.text_encoder
        tokens = tokenizer(
            list(prompts),
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)
        output = encoder(tokens, output_hidden_states=True)
        return output.hidden_states[-2].to(self.dtype), output.text_embeds

    @torch.no_grad()
    def _t5(self, prompts: Sequence[str]) -> torch.Tensor:
        tokens = self.tokenizer_3(
            list(prompts),
            padding="max_length",
            max_length=256,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        # The released pipeline encodes the full padded 256-token sequence.
        # Passing an attention mask changes the T5 context seen by the denoiser.
        return self.text_encoder_3(tokens.input_ids.to(self.device)).last_hidden_state.to(self.dtype)

    def _branch(self, prompts: Sequence[str]) -> dict[str, torch.Tensor]:
        clip_1, pooled_1 = self._clip(prompts, second=False)
        clip_2, pooled_2 = self._clip(prompts, second=True)
        clip = torch.cat((clip_1, clip_2), dim=-1)
        clip = torch.nn.functional.pad(clip, (0, 4096 - clip.shape[-1]))
        return {
            "prompt_embeds": torch.cat((clip, self._t5(prompts)), dim=-2),
            "pooled_prompt_embeds": torch.cat((pooled_1, pooled_2), dim=-1),
        }

    def encode(
        self,
        request: DiffusionRequest,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Conditioning:
        """Return SD3-style ``prompt_embeds`` and ``pooled_prompt_embeds``."""
        del device, dtype
        negative = request.negative_prompts or ("",) * request.batch_size
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            return Conditioning(
                positive=self._branch(request.prompts),
                negative=self._branch(negative),
            )


def build_vchitect_prompt_conditioner(context: ComponentBuildContext) -> VchitectPromptConditioner:
    """Build the dual-CLIP + T5 Vchitect prompt conditioner."""
    materialized = NativeCheckpointResolver().materialize(context.require_checkpoint("resources"))
    return VchitectPromptConditioner(
        materialized.root,
        device=context.policy.device,
        dtype=context.policy.dtype,
    )


__all__ = ["VchitectPromptConditioner", "build_vchitect_prompt_conditioner"]
