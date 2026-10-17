# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""CUDA student cache/precision contracts without distributed checkpoints."""

from __future__ import annotations

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from worldfoundry.base_models.diffusion_model.contracts import DenoiserInput
from worldfoundry.base_models.diffusion_model.models.denoisers.dreamdojo_student import DreamDojoStudentDenoiser
from worldfoundry.base_models.diffusion_model.models.networks.cosmos2p5.action_causal import (
    CausalActionCosmosTransformer,
)
from worldfoundry.base_models.diffusion_model.models.networks.cosmos2p5.model import Cosmos25Attention
from worldfoundry.core.attention import scaled_dot_product_attention
from worldfoundry.core.attention.cache.kv_cache_policy import SlidingWindowPolicy
from worldfoundry.core.attention.cache.kvcache import CompactingKVCache

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cuda_cached_attention_equals_explicit_history_on_same_backend(dtype: torch.dtype) -> None:
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(71)
        attention = Cosmos25Attention(32, None, 2, 16).to(device="cuda", dtype=dtype).eval()
        cache = CompactingKVCache(frame_tokens=4, policy=SlidingWindowPolicy(window_blocks=3), seq_dim=2)
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            for _ in range(5):
                clean = torch.randn(1, 4, 32, device="cuda", dtype=dtype)
                attention(clean, kv_cache=cache, store_kv=True)
            assert cache.length == 12
            history_key, history_value = (value.clone() for value in cache.cached())
            noisy = torch.randn(1, 4, 32, device="cuda", dtype=dtype)
            query = attention.norm_q(attention.to_q(noisy).unflatten(-1, (2, 16)).transpose(1, 2))
            key = attention.norm_k(attention.to_k(noisy).unflatten(-1, (2, 16)).transpose(1, 2))
            value = attention.to_v(noisy).unflatten(-1, (2, 16)).transpose(1, 2)
            reference = scaled_dot_product_attention(
                query, torch.cat((history_key, key), dim=2), torch.cat((history_value, value), dim=2)
            )
            reference = attention.to_out[0](reference.transpose(1, 2).flatten(2, 3))
            actual = attention(noisy, kv_cache=cache)
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            for actual_cache, expected in zip(cache.cached(), (history_key, history_value), strict=True):
                torch.testing.assert_close(actual_cache, expected, rtol=0, atol=0)


def test_cuda_bf16_student_denoising_is_finite_action_conditioned_and_repeatable() -> None:
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(17)
        model = (
            CausalActionCosmosTransformer(
                in_channels=3,
                out_channels=2,
                num_attention_heads=2,
                attention_head_dim=16,
                num_layers=2,
                text_in_channels=8,
                text_embed_dim=8,
                adaln_lora_dim=8,
            )
            .to(device="cuda", dtype=torch.bfloat16)
            .eval()
        )
        denoiser = DreamDojoStudentDenoiser(model)
        latent = torch.randn(1, 2, 1, 4, 4, device="cuda", dtype=torch.bfloat16)
        context = torch.randn(1, 3, 8, device="cuda", dtype=torch.bfloat16)
        caches: dict[int, CompactingKVCache] = {}
        mask = torch.zeros(1, 1, 1, 4, 4, device="cuda", dtype=torch.bfloat16)
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            model.forward_frame(
                latent,
                latent.new_zeros((1,)),
                context,
                action=None,
                frame_index=0,
                kv_cache=caches,
                store_kv=True,
                condition_mask=mask,
            )

            def predict(value: float) -> torch.Tensor:
                return denoiser(
                    DenoiserInput(
                        latents=latent,
                        timestep=latent.new_tensor(1.5703125),
                        next_timestep=latent.new_zeros(()),
                        step_index=0,
                        total_steps=1,
                        conditioning={
                            "context": context,
                            "condition_mask": mask,
                            "kv_cache": caches,
                            "frame_index": 1,
                            "action": torch.full((1, 4, 384), value, device="cuda", dtype=torch.bfloat16),
                        },
                    )
                ).sample

            first, changed, repeated = predict(0), predict(0.5), predict(0)
            assert first.dtype == torch.bfloat16 and torch.isfinite(first).all()
            torch.testing.assert_close(first, repeated, rtol=0, atol=0)
            assert not torch.equal(first, changed)
            assert all(len(cache.blocks) == 1 for cache in caches.values())
