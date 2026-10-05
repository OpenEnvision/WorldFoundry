# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import torch
from einops import rearrange

from worldfoundry.base_models.diffusion_model.models.autoencoders.cosmos1 import (
    Cosmos1VideoCodec,
    load_cosmos1_video_codec,
)


_TOKENIZER_CONFIG = {
    "name": "CV",
    "channels": 128,
    "latent_channels": 16,
    "spatial_compression": 8,
    "temporal_compression": 8,
}


class _LyraCosmos1Codec:
    """Preserve Lyra's raw Tokenize1 call shape over the shared native codec."""

    def __init__(self, codec: Cosmos1VideoCodec) -> None:
        self.codec = codec

    def to(self, *, device: torch.device | str, dtype: torch.dtype) -> "_LyraCosmos1Codec":
        self.codec.encoder.to(device=device, dtype=dtype)
        self.codec.decoder.to(device=device, dtype=dtype)
        self.codec.dtype = dtype
        return self

    def eval(self) -> "_LyraCosmos1Codec":
        self.codec.encoder.eval()
        self.codec.decoder.eval()
        return self

    def encode(self, video: torch.Tensor) -> tuple[torch.Tensor]:
        return (self.codec.encode_raw(video),)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.codec.decode_raw(latents)

def load_cosmos_1_decoder(vae_path: str, decoder_cosmos_kwargs):
    del vae_path, decoder_cosmos_kwargs
    raise NotImplementedError(
        "Lyra's configurable Cosmos decoder is a training-only path and is not "
        "part of the WorldFoundry inference runtime. Set use_cosmos_decoder=false."
    )

def get_tokenizer_config(checkpoint_path: str):
    model_name = os.path.basename(os.path.normpath(checkpoint_path))
    if "CV8x8x8" not in model_name and model_name not in {"tokenizer", "vae"}:
        raise ValueError(
            "Lyra-1 inference requires Cosmos-Tokenize1-CV8x8x8-720p; "
            f"got {model_name!r}"
        )
    return dict(_TOKENIZER_CONFIG)

def load_cosmos_1_tokenizer(checkpoint_path: str, load_encoder: bool = True, load_decoder: bool = False, load_jit: bool = True, return_tokenizer_config: bool = False, add_tokenizer_kwargs = None):
    del load_encoder, load_decoder
    if not load_jit or add_tokenizer_kwargs:
        raise NotImplementedError(
            "WorldFoundry exposes the released Tokenize1 JIT artifact for inference only"
        )
    tokenizer = _LyraCosmos1Codec(load_cosmos1_video_codec(checkpoint_path))
    if return_tokenizer_config:
        return tokenizer, get_tokenizer_config(checkpoint_path)
    return tokenizer

def load_cosmos_latent_statistics(vae_path: str, pixel_chunk_duration: int = 121, device: torch.device = 'cpu', weight_dtype: torch.dtype = None):
    tokenizer_config = get_tokenizer_config(vae_path)
    latent_chunk_duration = (pixel_chunk_duration - 1) // tokenizer_config['temporal_compression'] + 1
    latent_mean, latent_std = get_cosmos_diffusion_mean_std(vae_path, weight_dtype, tokenizer_config['latent_channels'], latent_chunk_duration)
    latent_mean = latent_mean.to(device)
    latent_std = latent_std.to(device)
    return latent_mean, latent_std

def get_cosmos_diffusion_mean_std(vae_dir: str, dtype: torch.dtype, latent_ch: int, latent_chunk_duration: int):
    latent_mean, latent_std = torch.load(os.path.join(vae_dir, "mean_std.pt"), weights_only=True)
    if dtype is None:
        dtype = latent_mean.dtype
    target_shape = [1, latent_ch, latent_chunk_duration, 1, 1]
    latent_mean = latent_mean.view(latent_ch, -1)
    latent_std = latent_std.view(latent_ch, -1)
    latent_mean = latent_mean.to(dtype).reshape(*target_shape)
    latent_std = latent_std.to(dtype).reshape(*target_shape)
    return latent_mean, latent_std

def denormalize_latents(model_input: torch.Tensor, latent_std: torch.Tensor, latent_mean: torch.Tensor, num_input_multi_views: int = 1, sigma_data: float = 0.5):
    # Add batch dimension
    if len(model_input.shape) == 4:
        model_input = model_input.unsqueeze(0)
        unsqueeze = True
    else:
        unsqueeze = False
    # Use same statistics across views
    model_input = rearrange(model_input, 'b (v t) c h w -> (b v) t c h w', v=num_input_multi_views)
    model_input = model_input / sigma_data
    model_input = model_input * latent_std + latent_mean
    # Convert from generated internal cosmos (B T C H W) to cosmos-predict (B C T H W)
    model_input = model_input.transpose(1, 2)
    # Reshape frames and views again in one dimension
    model_input = rearrange(model_input, '(b v) t c h w -> b (v t) c h w', v=num_input_multi_views)
    # Remove batch dimension
    if unsqueeze:
        model_input = model_input.squeeze(0)
    return model_input
