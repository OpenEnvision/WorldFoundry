# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Shared one-latent Wan decoding with request-owned feature history."""

from collections.abc import Sequence
from typing import Any

import torch

from worldfoundry.core.execution.realtime.feature_cache import commit_feature_cache


def new_decode_cache(model: Any) -> list[torch.Tensor | str | None]:
    return [None] * sum(isinstance(layer, torch.nn.Conv3d) for layer in model.decoder.modules())


def prepare_wan_chunk_decoder(decoder: torch.nn.Module, *, compile_decoder: bool = False) -> torch.nn.Module:
    """Reuse the optional full-graph one-latent wrapper, not a whole-clip graph."""
    from worldfoundry.core.platform.compile_cache import CompilePolicy, compile_module_cached

    return compile_module_cached(
        decoder,
        policy=CompilePolicy(enabled=compile_decoder, mode=None, dynamic=False, fullgraph=True),
        options={"triton.cudagraphs": False},
        strict=True,
    )


@torch.no_grad()
def decode_wan_chunk(
    model: Any,
    scale: Sequence[torch.Tensor],
    latents: torch.Tensor,
    cache: list[torch.Tensor | str | None],
    *,
    channels_last_3d: bool = False,
    decoder: torch.nn.Module | None = None,
) -> torch.Tensor:
    """Decode BCTHW chunks; on failure clear history and require a rollout restart."""
    try:
        dtype = model.conv2.weight.dtype
        latents = latents.to(device=model.conv2.weight.device, dtype=dtype)
        offset, inverse_std = (value.to(latents).view(1, latents.shape[1], 1, 1, 1) for value in scale)
        memory_format = torch.channels_last_3d if channels_last_3d else torch.contiguous_format
        value = model.conv2((latents / inverse_std + offset).contiguous(memory_format=memory_format))
        step = model.decoder if decoder is None else decoder
        decoded = []
        for index in range(value.shape[2]):
            frame = value[:, :, index : index + 1].contiguous(memory_format=memory_format)
            updated = list(cache)
            result = step(frame, feat_cache=updated, feat_idx=[0])
            pixels = result[0] if isinstance(result, tuple) else result
            if isinstance(result, tuple):
                updated = result[1]
            commit_feature_cache(cache, updated)
            decoded.append(pixels)
        return torch.cat(decoded, dim=2).float().clamp_(-1, 1)
    except Exception:
        # Earlier latent steps may already have committed history without returning pixels.
        cache[:] = [None] * len(cache)
        raise


class StreamingWanDecoder:
    """Request-owned history around a reusable small-chunk decoder callable."""

    def __init__(self, vae: Any, *, channels_last_3d: bool = False, compile_decoder: bool = False) -> None:
        self.vae = vae
        self.channels_last_3d = channels_last_3d
        self.decoder = prepare_wan_chunk_decoder(vae.model.decoder, compile_decoder=compile_decoder)
        self.compile_requested = compile_decoder
        self.compile_effective = self.decoder is not vae.model.decoder or hasattr(self.decoder, "_orig_mod")
        self.reset()

    def reset(self) -> None:
        self.cache = new_decode_cache(self.vae.model)
        self.calls = 0

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Continue this rollout, or reset on failure so callers can restart from its first latent."""
        try:
            pixels = decode_wan_chunk(
                self.vae.model,
                self.vae.scale,
                latents,
                self.cache,
                channels_last_3d=self.channels_last_3d,
                decoder=self.decoder,
            )
        except Exception:
            self.calls = 0
            raise
        self.calls += latents.shape[2]
        return pixels

    def report(self) -> dict[str, int | bool]:
        """Count successfully returned latent steps; compilation counts wrapper calls, not kernels."""
        return {
            "latent_decode_calls": self.calls,
            "compile_requested": self.compile_requested,
            "compile_effective": self.compile_effective,
            "compiled_decode_calls": self.calls if self.compile_effective else 0,
            "channels_last_3d": self.channels_last_3d,
            "cache_bytes": sum(
                value.numel() * value.element_size() for value in self.cache if isinstance(value, torch.Tensor)
            ),
        }
