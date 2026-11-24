"""Self-Forcing ``sink5-window7-rerope`` long-rollout cache, denoiser, runner and pipeline contracts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from worldfoundry.base_models.diffusion_model.components import (
    ComponentBuildContext,
    ComponentKey,
    ComponentKind,
)
from worldfoundry.base_models.diffusion_model.contracts import (
    Conditioning,
    DenoiserInput,
    DiffusionRequest,
    SamplingConfig,
)
from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec
from worldfoundry.base_models.diffusion_model.models.denoisers import wan as wan_denoisers
from worldfoundry.base_models.diffusion_model.models.denoisers.wan import (
    SELF_FORCING_LONG_ROLLOUT_KV_WINDOW,
    SelfForcingWanDenoiser,
    build_self_forcing_wan21_denoiser,
)
from worldfoundry.base_models.diffusion_model.models.initializers.wan.component import (
    WanTextToVideoLatentInitializer,
)
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.forcing import self_forcing
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.forcing.model import rope_params
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.forcing.self_forcing import (
    CausalWanModel,
    CausalWanSelfAttention,
)
from worldfoundry.base_models.diffusion_model.optimizations import RuntimePolicy
from worldfoundry.base_models.diffusion_model.recipes.wan import (
    SELF_FORCING_MODEL_ID,
    self_forcing_wan21_t2v_1p3b_recipe,
)
from worldfoundry.base_models.diffusion_model.runners.autoregressive import AutoregressiveWindowRunner
from worldfoundry.base_models.diffusion_model.runners.strategies import (
    ExecutionBuildContext,
    build_autoregressive_window_strategy,
)
from worldfoundry.base_models.diffusion_model.schedulers.wan import WanFlowMatchEulerScheduler
from worldfoundry.pipelines.forcing.pipeline_native_forcing import NativeSelfForcingPipeline

TINY_CONFIG: dict[str, Any] = {
    "model_type": "t2v",
    "patch_size": (1, 2, 2),
    "text_len": 8,
    "in_dim": 16,
    "dim": 24,
    "ffn_dim": 32,
    "freq_dim": 8,
    "text_dim": 16,
    "out_dim": 16,
    "num_heads": 2,
    "num_layers": 2,
    "local_attn_size": -1,
    "sink_size": 0,
    "qk_norm": True,
    "cross_attn_norm": True,
    "eps": 1e-6,
}
LONG_ROLLOUT_CONFIG = {**TINY_CONFIG, "local_attn_size": 12, "sink_size": 5, "kv_cache_relative_rope": True}
FRAME_TOKENS = 4  # a 4x4 latent with 2x2 patches


def _tiny_model(config: dict[str, Any], seed: int = 0) -> CausalWanModel:
    torch.manual_seed(seed)
    model = CausalWanModel(**config).eval()
    # The released head is zero-initialised; random weights make x0 depend on the flow.
    torch.nn.init.normal_(model.head.head.weight, std=0.02)
    return model


def _long_rollout_denoiser(seed: int = 0) -> SelfForcingWanDenoiser:
    return SelfForcingWanDenoiser(
        _tiny_model(LONG_ROLLOUT_CONFIG, seed),
        block_size=3,
        compute_dtype=torch.float32,
        kv_window=SELF_FORCING_LONG_ROLLOUT_KV_WINDOW,
    )


def _cache(denoiser: SelfForcingWanDenoiser, latent_frames: int) -> dict[str, Any]:
    return denoiser.create_kv_cache(
        batch_size=1,
        n_views=1,
        latent_frames_per_view=latent_frames,
        frame_sequence_length=FRAME_TOKENS,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def _frame_tagging_attention() -> CausalWanSelfAttention:
    """Self-attention whose keys equal its input, so cache slots reveal which frame they hold."""
    attention = CausalWanSelfAttention(
        dim=4,
        num_heads=1,
        local_attn_size=12,
        sink_size=5,
        qk_norm=False,
        kv_cache_relative_rope=True,
    )
    with torch.no_grad():
        for projection in (attention.q, attention.k, attention.v, attention.o):
            projection.weight.copy_(torch.eye(4))
            projection.bias.zero_()
    return attention


def test_long_rollout_cache_keeps_sink_and_rolls_window_in_order() -> None:
    attention = _frame_tagging_attention()
    capacity = 12 * FRAME_TOKENS
    kv_cache = {
        "k": torch.zeros(1, capacity, 1, 4),
        "v": torch.zeros(1, capacity, 1, 4),
        "global_end_index": torch.zeros(1, dtype=torch.long),
        "local_end_index": torch.zeros(1, dtype=torch.long),
    }
    freqs = rope_params(1024, 4)
    grid_sizes = torch.tensor([[3, 2, 2]])
    expected_slots = {
        0: [0, 1, 2],
        1: [0, 1, 2, 3, 4, 5],
        2: list(range(9)),
        3: list(range(12)),
        4: [0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14],
        5: [0, 1, 2, 3, 4, 11, 12, 13, 14, 15, 16, 17],
        6: [0, 1, 2, 3, 4, 14, 15, 16, 17, 18, 19, 20],
    }
    for block, slots in expected_slots.items():
        frames = torch.arange(3 * block, 3 * block + 3, dtype=torch.float32)
        x = frames.repeat_interleave(FRAME_TOKENS).view(1, -1, 1).expand(1, -1, 4).contiguous()
        # Four denoising calls and the clean commit rewrite the same block slots.
        for _ in range(5):
            attention(
                x,
                seq_lens=torch.tensor([x.shape[1]]),
                grid_sizes=grid_sizes,
                freqs=freqs,
                block_mask=None,
                kv_cache=kv_cache,
                current_start=3 * block * FRAME_TOKENS,
            )
        local_end = int(kv_cache["local_end_index"])
        assert int(kv_cache["global_end_index"]) == 3 * (block + 1) * FRAME_TOKENS
        assert local_end == len(slots) * FRAME_TOKENS
        cached = kv_cache["k"][0, :local_end, 0, 0].view(-1, FRAME_TOKENS)
        assert (cached == cached[:, :1]).all()
        assert cached[:, 0].tolist() == [float(frame) for frame in slots]


def test_long_rollout_rope_positions_are_cache_slots(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, int]] = []
    rope = self_forcing.causal_rope_apply

    def record(x: torch.Tensor, grid_sizes: torch.Tensor, freqs: torch.Tensor, start_frame: int = 0) -> torch.Tensor:
        calls.append((int(start_frame), int(grid_sizes[0, 0])))
        return rope(x, grid_sizes, freqs, start_frame=start_frame)

    monkeypatch.setattr(self_forcing, "causal_rope_apply", record)
    denoiser = _long_rollout_denoiser()
    cache = _cache(denoiser, 30)
    query_starts = []
    for block in range(6):
        calls.clear()
        denoiser(_block_input(cache, block))
        # Every layer rotates the query at its cache slots, then the cache read from slot 0.
        assert len(calls) == 2 * len(denoiser.model.blocks)
        assert set(calls[1::2]) == {(0, min(3 * block + 3, 12))}
        (query_call,) = set(calls[0::2])
        assert query_call[1] == 3
        query_starts.append(query_call[0])
    assert query_starts == [0, 3, 6, 9, 9, 9]


def _block_input(cache: dict[str, Any], block: int) -> DenoiserInput:
    latents = torch.randn(1, 16, 3, 4, 4, generator=torch.Generator().manual_seed(7 + block))
    return DenoiserInput(
        latents=latents,
        timestep=torch.tensor(750.0),
        next_timestep=torch.tensor(0.0),
        conditioning={
            "context": torch.randn(1, 8, 16, generator=torch.Generator().manual_seed(9)),
            "kv_cache": cache,
            "current_start": 3 * block * FRAME_TOKENS,
        },
        step_index=1,
        total_steps=4,
    )


def test_long_rollout_cache_is_bounded_and_default_cache_rejects_long_rollouts() -> None:
    long_rollout = _long_rollout_denoiser()
    cache = _cache(long_rollout, 120)
    assert cache["kv_cache"][0]["k"].shape == (1, 12 * FRAME_TOKENS, 2, 12)
    assert (cache["local_attention_frames"], cache["sink_frames"]) == (12, 5)

    default = SelfForcingWanDenoiser(_tiny_model(TINY_CONFIG), compute_dtype=torch.float32)
    assert default.kv_window is None
    with pytest.raises(ValueError, match="kv_window='sink5-window7-rerope'"):
        _cache(default, 24)
    with pytest.raises(ValueError, match="unsupported Self-Forcing kv_window"):
        SelfForcingWanDenoiser(_tiny_model(TINY_CONFIG), kv_window="sink4-window8")


class _Conditioner:
    def encode(self, request: DiffusionRequest, **_: object) -> Conditioning:
        seed = sum(map(ord, str(request.prompt)))
        context = torch.randn(1, 8, 16, generator=torch.Generator().manual_seed(seed))
        return Conditioning(positive={"context": context})


class _Decoder:
    def decode(self, latents: torch.Tensor, request: DiffusionRequest) -> torch.Tensor:
        del request
        return latents


def _runner(denoiser: SelfForcingWanDenoiser, decoder: Any = None) -> AutoregressiveWindowRunner:
    recipe = self_forcing_wan21_t2v_1p3b_recipe()
    by_role = {
        "denoiser": denoiser,
        "conditioner": _Conditioner(),
        "latent_initializer": WanTextToVideoLatentInitializer(),
        "scheduler": WanFlowMatchEulerScheduler(shift=5.0),
        "decoder": decoder or _Decoder(),
    }
    runner = build_autoregressive_window_strategy(
        ExecutionBuildContext(
            recipe=recipe,
            components={recipe.execution.bindings[role]: value for role, value in by_role.items()},
            policy=RuntimePolicy(device="cpu", dtype=torch.float32),
            extensions=(),
        )
    )
    assert isinstance(runner, AutoregressiveWindowRunner)
    return runner


def _request(latent_frames: int, *, seed: int = 42) -> DiffusionRequest:
    return DiffusionRequest(
        prompt="a red fox running through snow",
        height=32,
        width=32,
        num_frames=4 * latent_frames - 3,
        sampling=SamplingConfig(
            num_inference_steps=4,
            guidance_scale=1.0,
            seed=seed,
            scheduler_options={"shift": 5.0},
        ),
    )


def test_long_rollout_runner_matches_global_cache_until_the_window_rolls() -> None:
    long_rollout = _runner(_long_rollout_denoiser()).run(_request(21))
    reference = _runner(
        SelfForcingWanDenoiser(_tiny_model(TINY_CONFIG), cache_frames=21, compute_dtype=torch.float32)
    ).run(_request(21))

    assert long_rollout.latents.shape == (1, 16, 21, 4, 4)
    assert bool(torch.isfinite(long_rollout.latents).all())
    # Blocks 0-3 fill the twelve-frame cache with the same keys and positions.
    torch.testing.assert_close(long_rollout.latents[:, :, :12], reference.latents[:, :, :12], rtol=0, atol=0)
    assert not torch.equal(long_rollout.latents[:, :, 12:15], reference.latents[:, :, 12:15])


def test_unbounded_stream_continues_past_the_request_with_flat_cache() -> None:
    runner = _runner(_long_rollout_denoiser())
    request = _request(9)
    expected = runner.run(request).latents

    def rollout() -> list[torch.Tensor]:
        stream = runner.open_stream(request, unbounded=True)
        assert stream.total_blocks is None
        blocks = []
        for index in range(8):
            blocks.append(stream.step().clone())
            assert not stream.done
            assert stream.last_block_frames == (3 * index, 3 * index + 3)
        state = stream._state
        assert state is not None
        assert state.output_by_view.shape[3] == 9
        layer = state.positive_cache["kv_cache"][0]
        assert int(layer["local_end_index"]) == 12 * FRAME_TOKENS
        assert int(layer["global_end_index"]) == 24 * FRAME_TOKENS
        stream.close()
        assert runner._active_stream is None
        return blocks

    blocks = rollout()
    assert all(block.shape == (1, 16, 3, 4, 4) for block in blocks)
    torch.testing.assert_close(torch.cat(blocks[:3], dim=2), expected, rtol=0, atol=0)
    assert all(bool(torch.isfinite(block).all()) for block in blocks)
    for first, second in zip(blocks, rollout(), strict=True):
        torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_long_rollout_builder_reads_kv_window_from_the_runtime_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(wan_denoisers, "SELF_FORCING_WAN21_T2V_1P3B_CONFIG", TINY_CONFIG)
    reference = _tiny_model(TINY_CONFIG, seed=3)
    checkpoint = tmp_path / "self_forcing_dmd.pt"
    torch.save({"generator_ema": {"model." + k: v for k, v in reference.state_dict().items()}}, checkpoint)

    def build(**options: object) -> SelfForcingWanDenoiser:
        return build_self_forcing_wan21_denoiser(
            ComponentBuildContext(
                model_id=SELF_FORCING_MODEL_ID,
                key=ComponentKey(ComponentKind.DENOISER),
                policy=RuntimePolicy(device="cpu", dtype=torch.float32, options=options),
                checkpoints={"weights": CheckpointSpec(source=checkpoint)},
            )
        )

    denoiser = build(kv_window=SELF_FORCING_LONG_ROLLOUT_KV_WINDOW)
    assert denoiser.kv_window == SELF_FORCING_LONG_ROLLOUT_KV_WINDOW
    attention = denoiser.model.blocks[0].self_attn
    assert (attention.local_attn_size, attention.sink_size, attention.kv_cache_relative_rope) == (12, 5, True)
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(denoiser.model.state_dict()[name], value, rtol=0, atol=0)

    default = build()
    assert default.kv_window is None
    attention = default.model.blocks[0].self_attn
    assert (attention.local_attn_size, attention.sink_size, attention.kv_cache_relative_rope) == (-1, 0, False)

    with pytest.raises(ValueError, match="unsupported Self-Forcing kv_window"):
        build(kv_window="window21")

    causal = ComponentBuildContext(
        model_id="causal-forcing-chunkwise",
        key=ComponentKey(ComponentKind.DENOISER),
        policy=RuntimePolicy(
            device="cpu", dtype=torch.float32, options={"kv_window": SELF_FORCING_LONG_ROLLOUT_KV_WINDOW}
        ),
        checkpoints={"weights": CheckpointSpec(source=checkpoint)},
    )
    with pytest.raises(ValueError, match="kv_window is the Self-Forcing long-rollout preset"):
        build_self_forcing_wan21_denoiser(causal)


def test_native_self_forcing_pipeline_forwards_kv_window(monkeypatch: pytest.MonkeyPatch) -> None:
    from worldfoundry.pipelines import native_diffusion

    captured: dict[str, Any] = {}

    def load(model_id: str, *, policy: RuntimePolicy, **kwargs: Any) -> Any:
        captured.update(model_id=model_id, policy=policy, **kwargs)
        return SimpleNamespace(model_id=model_id)

    monkeypatch.setattr(native_diffusion.NativeDiffusionPipeline, "from_pretrained", load)
    NativeSelfForcingPipeline.from_pretrained(device="cpu", kv_window=SELF_FORCING_LONG_ROLLOUT_KV_WINDOW)
    assert captured["policy"].options["kv_window"] == SELF_FORCING_LONG_ROLLOUT_KV_WINDOW

    NativeSelfForcingPipeline.from_pretrained(device="cpu")
    assert "kv_window" not in captured["policy"].options


def test_long_rollout_realtime_session_is_unbounded() -> None:
    from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.component import WanVideoDecoder
    from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.model import WanVideoVAE
    from worldfoundry.base_models.diffusion_model.pipeline import NativeDiffusionPipeline

    torch.manual_seed(11)
    codec = WanVideoDecoder(WanVideoVAE(z_dim=16, base_dim=8), device=torch.device("cpu"))
    runner = _runner(_long_rollout_denoiser(), decoder=codec)
    pipeline = NativeSelfForcingPipeline(native_pipeline=NativeDiffusionPipeline(runner), device="cpu")

    prepared = pipeline.prepare_realtime()["realtime_spec"]
    assert "max_chunks" not in prepared
    assert (prepared["first_chunk_frames"], prepared["steady_chunk_frames"]) == (9, 12)
    configured = pipeline.configure_realtime(prompt="fox", seed=4, height=32, width=32, num_frames=33)
    assert configured["realtime_spec"] == prepared

    chunks = [pipeline.stream_realtime() for _ in range(6)]
    assert [chunk["video"].shape[0] for chunk in chunks] == [9, 12, 12, 12, 12, 12]
    assert [chunk["realtime_metrics"]["chunk_index"] for chunk in chunks] == [1, 2, 3, 4, 5, 6]
    assert {chunk["realtime_metrics"]["chunks_total"] for chunk in chunks} == {None}
    assert all("max_chunks" not in chunk["realtime_spec"] for chunk in chunks)
    assert runner._active_stream is not None
    pipeline.reset_realtime()
    assert runner._active_stream is None
