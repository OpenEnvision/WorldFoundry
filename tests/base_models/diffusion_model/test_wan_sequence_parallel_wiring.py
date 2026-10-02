from __future__ import annotations

import copy
import os
from types import SimpleNamespace

import pytest
import torch

from worldfoundry.base_models.diffusion_model.models.networks.wan.model import WanModel
from worldfoundry.base_models.diffusion_model.models.networks.wan.vace import VaceWanModel


class _IdentityBlock(torch.nn.Module):
    def forward(self, x, context, t_mod, freqs, **kwargs):
        del context, t_mod, freqs, kwargs
        return x


class _IdentityVaceBlock(torch.nn.Module):
    def forward(self, condition, hidden_states, context, t_mod, freqs):
        del context, t_mod
        assert condition.shape == hidden_states.shape
        assert freqs.shape[0] == 8
        return condition, condition


class _IdentityVaceMainBlock(torch.nn.Module):
    def forward(self, hidden_states, context, t_mod, freqs, *, hints, context_scale):
        del context, t_mod, context_scale
        assert hints[0].shape == hidden_states.shape
        assert freqs.shape[0] == 8
        return hidden_states


def test_wan_sequence_parallel_pads_shards_per_token_modulation_and_gathers(
    monkeypatch,
) -> None:
    from worldfoundry.core.distributed import sequence_parallel_runtime

    model = WanModel(
        dim=32,
        in_dim=4,
        ffn_dim=64,
        out_dim=4,
        freq_dim=16,
        text_dim=16,
        num_heads=4,
        num_layers=1,
        patch_size=(1, 2, 2),
        eps=1e-6,
        has_image_input=False,
        per_token_timestep=True,
    ).eval()
    model.blocks = torch.nn.ModuleList([_IdentityBlock()])
    model._worldfoundry_sequence_parallel = SimpleNamespace(sp_degree=4)

    chunk_shapes: list[tuple[int, ...]] = []

    def fake_chunk(value: torch.Tensor, dim: int = 1) -> torch.Tensor:
        chunk_shapes.append(tuple(value.shape))
        return torch.chunk(value, 4, dim=dim)[0]

    def fake_gather(value: torch.Tensor, dim: int = 1) -> torch.Tensor:
        return torch.cat([value, value, value, value], dim=dim)

    monkeypatch.setattr(sequence_parallel_runtime, "sequence_parallel_chunk", fake_chunk)
    monkeypatch.setattr(
        sequence_parallel_runtime,
        "sequence_parallel_all_gather",
        fake_gather,
    )

    # Patch grid is 1x2x3 = 6 tokens, so degree=4 pads it to 8 before both
    # hidden-state and per-token timestep-modulation sharding.
    latents = torch.randn(1, 4, 1, 4, 6)
    timesteps = torch.full((1, 6), 500.0)
    context = torch.randn(1, 3, 16)
    with torch.no_grad():
        output = model(latents, timesteps, context)
    assert output.shape == latents.shape
    assert chunk_shapes[0][1] == 8
    assert chunk_shapes[1][1] == 8


def test_wan_vace_sequence_parallel_shards_control_and_main_streams(monkeypatch) -> None:
    from worldfoundry.core.distributed import sequence_parallel_runtime

    model = VaceWanModel(
        vace_layers=(0,),
        vace_in_dim=6,
        dim=32,
        in_dim=4,
        ffn_dim=64,
        out_dim=4,
        freq_dim=16,
        text_dim=16,
        num_heads=4,
        num_layers=1,
        patch_size=(1, 2, 2),
        eps=1e-6,
    ).eval()
    model.vace_blocks = torch.nn.ModuleList([_IdentityVaceBlock()])
    model.blocks = torch.nn.ModuleList([_IdentityVaceMainBlock()])
    model._worldfoundry_sequence_parallel = SimpleNamespace(sp_degree=4)

    chunk_shapes: list[tuple[int, ...]] = []

    def fake_chunk(value: torch.Tensor, dim: int = 1) -> torch.Tensor:
        chunk_shapes.append(tuple(value.shape))
        return torch.chunk(value, 4, dim=dim)[0]

    def fake_gather(value: torch.Tensor, dim: int = 1) -> torch.Tensor:
        return torch.cat([value, value, value, value], dim=dim)

    monkeypatch.setattr(sequence_parallel_runtime, "sequence_parallel_chunk", fake_chunk)
    monkeypatch.setattr(
        sequence_parallel_runtime,
        "sequence_parallel_all_gather",
        fake_gather,
    )

    latents = torch.randn(1, 4, 1, 4, 6)
    vace_context = torch.randn(1, 6, 1, 4, 6)
    timestep = torch.tensor([500.0])
    context = torch.randn(1, 3, 16)
    with torch.no_grad():
        output = model(latents, timestep, context, vace_context)

    assert output.shape == latents.shape
    assert [shape[1] for shape in chunk_shapes] == [8, 8]


@pytest.mark.skipif(
    int(os.getenv("WORLD_SIZE", "1")) != 4 or not torch.cuda.is_available(),
    reason="requires a four-rank CUDA torchrun",
)
def test_wan_vace_real_four_rank_sequence_parallel_matches_full_attention() -> None:
    import torch.distributed as dist

    from worldfoundry.base_models.diffusion_model.optimizations.sequence_parallel import (
        enable_sequence_parallel,
    )
    from worldfoundry.core.distributed.sequence_parallel_runtime import (
        ensure_parallel_runtime,
    )

    device = ensure_parallel_runtime(4, 1)
    torch.manual_seed(1234)
    reference = VaceWanModel(
        vace_layers=(0,),
        vace_in_dim=6,
        dim=32,
        in_dim=4,
        ffn_dim=64,
        out_dim=4,
        freq_dim=16,
        text_dim=16,
        num_heads=4,
        num_layers=1,
        patch_size=(1, 2, 2),
        eps=1e-6,
    ).to(device=device, dtype=torch.float32).eval()
    parallel = copy.deepcopy(reference)
    state = enable_sequence_parallel(parallel, 4)
    parallel._worldfoundry_sequence_parallel = state

    generator = torch.Generator(device=device).manual_seed(5678)
    latents = torch.randn(1, 4, 1, 4, 8, generator=generator, device=device)
    vace_context = torch.randn(1, 6, 1, 4, 8, generator=generator, device=device)
    timestep = torch.tensor([500.0], device=device)
    context = torch.randn(1, 3, 16, generator=generator, device=device)
    with torch.no_grad():
        expected = reference(latents, timestep, context, vace_context)
        actual = parallel(latents, timestep, context, vace_context)

    assert state.wrapped_blocks == 2
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
    dist.barrier(device_ids=[device.index])
    dist.destroy_process_group()


@pytest.mark.skipif(
    int(os.getenv("WORLD_SIZE", "1")) != 4 or not torch.cuda.is_available(),
    reason="requires a four-rank CUDA torchrun",
)
def test_wan_real_four_rank_bfloat16_flash_sequence_parallel_matches_full_attention() -> None:
    import torch.distributed as dist

    from worldfoundry.base_models.diffusion_model.optimizations.sequence_parallel import (
        enable_sequence_parallel,
    )
    from worldfoundry.core.distributed.sequence_parallel_runtime import (
        ensure_parallel_runtime,
    )
    from worldfoundry.core.model_loading.optimize import apply_attention_policy

    device = ensure_parallel_runtime(4, 1)
    results: list[tuple[str, int, float, float, float]] = []
    for backend in ("torch", "flash"):
        for layers in (1, 30):
            torch.manual_seed(1234)
            reference = WanModel(
                dim=128,
                in_dim=4,
                ffn_dim=256,
                out_dim=4,
                freq_dim=64,
                text_dim=64,
                num_heads=4,
                num_layers=layers,
                patch_size=(1, 2, 2),
                eps=1e-6,
                has_image_input=False,
            ).to(device=device, dtype=torch.bfloat16).eval()
            parallel = copy.deepcopy(reference)
            apply_attention_policy(reference, backend, device=device)
            apply_attention_policy(parallel, backend, device=device)
            state = enable_sequence_parallel(parallel, 4)
            parallel._worldfoundry_sequence_parallel = state

            generator = torch.Generator(device=device).manual_seed(5678)
            latents = torch.randn(
                1,
                4,
                1,
                16,
                16,
                generator=generator,
                device=device,
                dtype=torch.bfloat16,
            )
            timestep = torch.tensor([500.0], device=device)
            context = torch.randn(
                1,
                32,
                64,
                generator=generator,
                device=device,
                dtype=torch.bfloat16,
            )
            with torch.no_grad():
                expected = reference(latents, timestep, context)
                actual = parallel(latents, timestep, context)

            expected_float = expected.float()
            actual_float = actual.float()
            difference = actual_float - expected_float
            cosine = float(
                torch.nn.functional.cosine_similarity(
                    expected_float.flatten(), actual_float.flatten(), dim=0
                ).item()
            )
            mse = float(difference.square().mean().item())
            max_error = float(difference.abs().amax().item())
            results.append((backend, layers, cosine, mse, max_error))
            assert state.wrapped_blocks == layers
            assert torch.isfinite(actual).all()

    if dist.get_rank() == 0:
        print(f"Wan BF16 live-NCCL parity: {results}")
    for backend, layers, cosine, mse, max_error in results:
        assert cosine >= 0.999, (backend, layers, cosine, mse, max_error)
    dist.barrier(device_ids=[device.index])
    dist.destroy_process_group()
