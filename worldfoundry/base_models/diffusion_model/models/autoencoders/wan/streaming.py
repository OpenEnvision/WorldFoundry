# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Shared one-latent Wan decoding with request-owned feature history."""

import torch

from worldfoundry.core.execution.realtime.feature_cache import commit_feature_cache


def new_decode_cache(model):
    return [None] * sum(isinstance(layer, torch.nn.Conv3d) for layer in model.decoder.modules())


@torch.no_grad()
def decode_wan_chunk(model, scale, latents, cache, *, channels_last_3d=False, decoder=None):
    """Decode BCTHW latent chunks; first frame is causal seed, later frames continue."""
    dtype = model.conv2.weight.dtype
    latents = latents.to(device=model.conv2.weight.device, dtype=dtype)
    offset, inverse_std = (value.to(latents).view(1, latents.shape[1], 1, 1, 1) for value in scale)
    memory_format = torch.channels_last_3d if channels_last_3d else torch.contiguous_format
    value = model.conv2((latents / inverse_std + offset).contiguous(memory_format=memory_format))
    step = model.decoder if decoder is None else decoder
    decoded = []
    for index in range(value.shape[2]):
        frame = value[:, :, index:index + 1].contiguous(memory_format=memory_format)
        updated = list(cache)
        result = step(frame, feat_cache=updated, feat_idx=[0])
        pixels = result[0] if isinstance(result, tuple) else result
        if isinstance(result, tuple):
            updated = result[1]
        commit_feature_cache(cache, updated)
        decoded.append(pixels)
    return torch.cat(decoded, dim=2).float().clamp_(-1, 1)


class StreamingWanDecoder:
    """Separate small-chunk decoder callable, without replacing checkpoint modules."""

    def __init__(self, vae, *, channels_last_3d=False, compile_decoder=False):
        self.vae = vae
        self.channels_last_3d = channels_last_3d
        self.decoder = vae.model.decoder
        if compile_decoder:
            self.decoder = torch.compile(self.decoder, dynamic=False, fullgraph=True, options={"triton.cudagraphs": False})
        self.compile_requested = compile_decoder
        self.calls = 0
        self.reset()

    def reset(self):
        self.cache = new_decode_cache(self.vae.model)
        self.calls = 0

    def decode(self, latents):
        pixels = decode_wan_chunk(self.vae.model, self.vae.scale, latents, self.cache,
                                  channels_last_3d=self.channels_last_3d, decoder=self.decoder)
        self.calls += latents.shape[2]
        return pixels

    def report(self):
        return {"latent_decode_calls": self.calls, "compile_requested": self.compile_requested,
                "compiled_decode_calls": self.calls if self.compile_requested else 0,
                "channels_last_3d": self.channels_last_3d,
                "cache_bytes": sum(value.numel() * value.element_size() for value in self.cache
                                   if isinstance(value, torch.Tensor))}
