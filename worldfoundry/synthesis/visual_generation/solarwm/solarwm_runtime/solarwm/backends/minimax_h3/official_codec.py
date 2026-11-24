"""Official Qwen/VisualVAE/AudioVAE codec shared by online and preencode paths."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...errors import DataContractError

H3_TEXT_ENCODER_LAYER = 50
H3_VIDEO_VAE_SEED = 42
H3_AUDIO_HOP_LENGTH = 800
H3_AUDIO_LATENTS = 263
H3_SILENCE_METADATA = {"artifact": "official-audiovae-encoded-stereo-zero-waveform"}


def _pixels(value: Any, *, device: Any) -> Any:
    import torch

    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.ndim == 4 and tensor.shape[-1] == 3:
        tensor = tensor.permute(3, 0, 1, 2).unsqueeze(0)
    elif tensor.ndim == 5 and tensor.shape[1] == 3:
        pass
    else:
        raise DataContractError("H3 pixels must be [T,H,W,3] or [B,3,T,H,W]")
    if tuple(tensor.shape) != (1, 3, 158, 768, 1344):
        raise DataContractError(f"H3 pixels must be [1,3,158,768,1344], got {tuple(tensor.shape)}")
    return tensor.to(device=device, non_blocking=True).contiguous()


def encode_joint_prompt_condition(
    keyframe: Any, caption: str, *, processor: Any, tokenizer: Any, text_encoder: Any, device: Any
) -> tuple[Any, Any]:
    import torch
    from PIL import Image

    if not isinstance(keyframe, Image.Image):
        array = np.asarray(keyframe, dtype=np.uint8)
        keyframe = Image.fromarray(array, mode="RGB")
    vision = processor.image_processor(images=[keyframe], return_tensors="pt")
    grid = vision["image_grid_thw"]
    merge = processor.image_processor.merge_size**2
    image_tokens = int(grid[0].prod()) // merge
    label_ids = tokenizer("<Picture 1>: ", add_special_tokens=False)["input_ids"]
    vision_ids = (
        [tokenizer.convert_tokens_to_ids("<|vision_start|>")]
        + [tokenizer.convert_tokens_to_ids("<|image_pad|>")] * image_tokens
        + [tokenizer.convert_tokens_to_ids("<|vision_end|>")]
    )
    caption_ids = tokenizer(str(caption), add_special_tokens=False)["input_ids"]
    token_ids = label_ids + vision_ids + caption_ids
    tags = [1] * len(label_ids) + [0] * len(vision_ids) + [1] * len(caption_ids)
    if not token_ids:
        raise DataContractError("H3 Qwen presentation produced no tokens")
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    token_types = torch.tensor(
        processor.create_mm_token_type_ids([token_ids]),
        dtype=torch.long,
        device=device,
    )
    outputs = text_encoder.model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        mm_token_type_ids=token_types,
        use_cache=False,
        output_hidden_states=True,
        pixel_values=vision["pixel_values"].to(device, text_encoder.dtype),
        image_grid_thw=grid.to(device),
    )
    if len(outputs.hidden_states) <= H3_TEXT_ENCODER_LAYER:
        raise DataContractError("H3 Qwen has no hidden state 50")
    hidden = outputs.hidden_states[H3_TEXT_ENCODER_LAYER][0].to(torch.bfloat16).contiguous().cpu()
    tags_tensor = torch.tensor(tags, dtype=torch.long)
    if tuple(hidden.shape) != (len(token_ids), 5120):
        raise DataContractError("H3 Qwen hidden state has the wrong shape")
    return hidden, tags_tensor


__all__ = [
    "H3_AUDIO_HOP_LENGTH",
    "H3_AUDIO_LATENTS",
    "H3_SILENCE_METADATA",
    "H3_TEXT_ENCODER_LAYER",
    "H3_VIDEO_VAE_SEED",
]
