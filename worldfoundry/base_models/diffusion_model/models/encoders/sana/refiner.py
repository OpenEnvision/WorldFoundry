"""Gemma-3 + LTX-2 text conditioning for the SANA-WM streaming refiner."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from ....components import ComponentBuildContext
from ....contracts import Conditioning, DiffusionRequest
from ....loaders import NativeCheckpointResolver
from ....optimizations import OffloadMode


def _pack_text_embeds(
    hidden_states: torch.Tensor,
    sequence_lengths: torch.Tensor,
    *,
    padding_side: str,
    scale_factor: int = 8,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Normalize and flatten Gemma hidden-layer stacks for LTX-2 connectors."""

    batch, sequence, hidden, _layers = hidden_states.shape
    token_indices = torch.arange(sequence, device=hidden_states.device).unsqueeze(0)
    if padding_side == "right":
        mask = token_indices < sequence_lengths[:, None]
    elif padding_side == "left":
        mask = token_indices >= sequence - sequence_lengths[:, None]
    else:
        raise ValueError(f"unsupported tokenizer padding_side: {padding_side!r}")
    expanded = mask[:, :, None, None]
    masked = hidden_states.masked_fill(~expanded, 0.0)
    denominator = (sequence_lengths * hidden).view(batch, 1, 1, 1)
    mean = masked.sum(dim=(1, 2), keepdim=True) / (denominator + eps)
    minimum = hidden_states.masked_fill(~expanded, float("inf")).amin(
        dim=(1, 2), keepdim=True
    )
    maximum = hidden_states.masked_fill(~expanded, float("-inf")).amax(
        dim=(1, 2), keepdim=True
    )
    normalized = (hidden_states - mean) / (maximum - minimum + eps)
    normalized = (normalized * scale_factor).flatten(2)
    flattened_mask = mask[:, :, None].expand(-1, -1, normalized.shape[-1])
    return normalized.masked_fill(~flattened_mask, 0.0)


class SanaWMRefinerConditioner:
    """Encode prompts once for the video-only LTX-2 refinement stage."""

    def __init__(
        self,
        text_encoder: torch.nn.Module,
        tokenizer: Any,
        connectors: torch.nn.Module,
        *,
        max_length: int = 1024,
        offload_after_encode: bool = False,
    ) -> None:
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.connectors = connectors
        self.max_length = int(max_length)
        self.offload_after_encode = bool(offload_after_encode)
        if self.max_length <= 0:
            raise ValueError("SANA-WM refiner max_length must be positive")
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.inference_mode()
    def _branch(
        self,
        prompts: Sequence[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        try:
            self.text_encoder.to(device=device, dtype=dtype)
            self.connectors.to(device=device, dtype=dtype)
            tokens = self.tokenizer(
                [str(prompt).strip() for prompt in prompts],
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            input_ids = tokens.input_ids.to(device=device)
            attention_mask = tokens.attention_mask.to(device=device)
            backbone = getattr(self.text_encoder, "model", self.text_encoder)
            outputs = backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            if outputs.hidden_states is None:
                raise RuntimeError("Gemma-3 refiner encoder did not return hidden states")
            hidden_states = torch.stack(outputs.hidden_states, dim=-1)
            packed = _pack_text_embeds(
                hidden_states,
                attention_mask.sum(dim=-1),
                padding_side=self.tokenizer.padding_side,
            ).to(device=device, dtype=dtype)
            video, audio, connector_mask = self.connectors(packed, attention_mask)
            return {
                "refiner_video_context": video.to(device=device, dtype=dtype),
                "refiner_audio_context": audio.to(device=device, dtype=dtype),
                "refiner_context_mask": connector_mask.to(device=device),
            }
        finally:
            if self.offload_after_encode:
                self.text_encoder.to(device="cpu")
                self.connectors.to(device="cpu")
                if device.type == "cuda" and torch.cuda.is_available():
                    with torch.cuda.device(device):
                        torch.cuda.empty_cache()

    def encode(
        self,
        request: DiffusionRequest,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Conditioning:
        return Conditioning(
            positive=self._branch(request.prompts, device=device, dtype=dtype),
        )


def build_sana_wm_refiner_conditioner(
    context: ComponentBuildContext,
) -> SanaWMRefinerConditioner:
    """Load pinned Gemma-3 and LTX-2 connector directories without network fallback."""

    from diffusers.pipelines.ltx2 import LTX2TextConnectors
    from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

    gemma = NativeCheckpointResolver().materialize(context.require_checkpoint("weights"))
    connectors_checkpoint = NativeCheckpointResolver().materialize(
        context.require_checkpoint("connectors")
    )
    gemma_root = gemma.directory("gemma3_12b")
    connectors_root = connectors_checkpoint.directory("refiner_diffusers/connectors")
    tokenizer = AutoTokenizer.from_pretrained(str(gemma_root), local_files_only=True)
    text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
        str(gemma_root),
        dtype=context.policy.dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).eval()
    connectors = LTX2TextConnectors.from_pretrained(
        str(connectors_root),
        torch_dtype=context.policy.dtype,
        local_files_only=True,
    ).eval()
    staged_offload = bool(
        context.policy.device.type != "cpu"
        and context.component_options.get(
            "offload",
            context.policy.offload.mode is not OffloadMode.NONE,
        )
    )
    return SanaWMRefinerConditioner(
        text_encoder,
        tokenizer,
        connectors,
        max_length=int(context.component_options.get("max_length", 1024)),
        offload_after_encode=staged_offload,
    )


__all__ = [
    "SanaWMRefinerConditioner",
    "build_sana_wm_refiner_conditioner",
]
