"""Dependency-light contracts for MiniMax-H3 raw and encoded model inputs.

The concrete lazy runtime lives in :mod:`official_codec`; keeping its protocol
and geometry checks here lets CPU-only installations audit the data boundary.
"""

from __future__ import annotations

from .geometry import (
    STABLE_STAGE0P5_GEOMETRY,
)

H3_PREENCODE_VERSION = "h3.158f.v1"
H3_TEXT_HIDDEN_DIM = 5120
H3_AUDIO_HIDDEN_DIM = 32


def _shape(value: object, *, name: str) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise TypeError(f"{name} must expose a tensor-like shape")
    try:
        return tuple(int(size) for size in shape)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} has an invalid shape {shape!r}") from exc


def _dtype_name(value: object, *, name: str) -> str:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        raise TypeError(f"{name} must expose a tensor-like dtype")
    result = str(dtype).lower().replace("torch.", "").replace("numpy.", "")
    return result


def _require_shape(value: object, expected: tuple[int, ...], *, name: str) -> None:
    observed = _shape(value, name=name)
    if observed != expected:
        raise ValueError(f"{name} shape {observed} != {expected}")


def _require_dtype(value: object, expected: str, *, name: str) -> None:
    observed = _dtype_name(value, name=name)
    aliases = {
        "bfloat16": {"bfloat16", "bf16"},
        "float32": {"float32", "float"},
        "int64": {"int64", "long"},
    }
    if observed not in aliases[expected]:
        raise TypeError(f"{name} dtype {observed!r} is not {expected}")


def _validate_video_text_tensors(
    target_latents: object,
    anchor_latent: object,
    prompt_embeddings: object,
    text_token_tags: object,
) -> None:
    profile = STABLE_STAGE0P5_GEOMETRY
    _require_shape(
        target_latents,
        (
            profile.latent_channels,
            profile.encoded_latents,
            profile.latent_height,
            profile.latent_width,
        ),
        name="target_latents",
    )
    _require_shape(
        anchor_latent,
        (
            profile.latent_channels,
            1,
            profile.latent_height,
            profile.latent_width,
        ),
        name="anchor_latent",
    )
    prompt_shape = _shape(prompt_embeddings, name="prompt_embeddings")
    if len(prompt_shape) != 2 or prompt_shape[0] < 1 or prompt_shape[1] != H3_TEXT_HIDDEN_DIM:
        raise ValueError("prompt_embeddings must have shape [L,5120] with L >= 1")
    _require_shape(
        text_token_tags,
        (prompt_shape[0],),
        name="text_token_tags",
    )
    for name, value in (
        ("target_latents", target_latents),
        ("anchor_latent", anchor_latent),
        ("prompt_embeddings", prompt_embeddings),
    ):
        _require_dtype(value, "bfloat16", name=name)
    _require_dtype(text_token_tags, "int64", name="text_token_tags")


__all__ = [
    "H3_AUDIO_HIDDEN_DIM",
    "H3_PREENCODE_VERSION",
    "H3_TEXT_HIDDEN_DIM",
]
