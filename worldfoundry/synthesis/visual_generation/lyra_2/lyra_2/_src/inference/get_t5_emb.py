from __future__ import annotations

import os
from typing import List, Optional, Union

import torch

from worldfoundry.base_models.diffusion_model.models.encoders.wan import WanUMT5PromptEncoder

_t5_encoder: Optional[WanUMT5PromptEncoder] = None
_t5_offloaded: Optional[WanUMT5PromptEncoder] = None


def _build_encoder(*, device: str, text_len: int) -> WanUMT5PromptEncoder:
    return WanUMT5PromptEncoder(
        text_length=text_len,
        device=device,
        checkpoint_path=os.environ.get(
            "LYRA2_TEXT_ENCODER_CKPT",
            "./checkpoints/text_encoder/encoder.pth",
        ),
        tokenizer_path=os.environ.get("LYRA2_UMT5_TOKENIZER", "google/umt5-xxl"),
    )


def get_umt5_embedding(
    prompts: Union[str, List[str]],
    device: str = "cuda",
    max_length: int = 512,
) -> torch.Tensor:
    global _t5_encoder
    if _t5_encoder is None:
        _t5_encoder = _build_encoder(device=device, text_len=max_length)
    return _t5_encoder(prompts, device=device)


@torch.no_grad()
def get_umt5_embedding_offloaded(
    prompts: Union[str, List[str]],
    device: str = "cuda",
    max_length: int = 512,
) -> torch.Tensor:
    global _t5_offloaded
    if _t5_offloaded is None:
        _t5_offloaded = _build_encoder(device="cpu", text_len=max_length)

    _t5_offloaded.model.to(device)
    _t5_offloaded.device = torch.device(device)
    try:
        return _t5_offloaded(prompts, device=device)
    finally:
        _t5_offloaded.model.to("cpu")
        _t5_offloaded.device = torch.device("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
