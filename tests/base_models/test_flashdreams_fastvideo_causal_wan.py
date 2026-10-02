from __future__ import annotations

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
    DenoiserOutput,
    DiffusionRequest,
    LatentInitialization,
    SamplingConfig,
)
from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec
from worldfoundry.base_models.diffusion_model.models.denoisers.wan import (
    CausalWanCacheBundle,
    FastVideoCausalWanDenoiser,
    build_fastvideo_causal_wan22_denoiser,
    convert_diffusers_wan_transformer_state_dict,
)
from worldfoundry.base_models.diffusion_model.optimizations import RuntimePolicy
from worldfoundry.base_models.diffusion_model.recipes.registry import (
    default_native_diffusion_registry,
)
from worldfoundry.base_models.diffusion_model.recipes.wan import (
    FASTVIDEO_CAUSAL_WAN22_REPO_ID,
    FASTVIDEO_CAUSAL_WAN22_REVISION,
    fastvideo_causal_wan22_i2v_14b_recipe,
)
from worldfoundry.base_models.diffusion_model.runners.autoregressive import (
    AutoregressiveWindowRunner,
)
from worldfoundry.base_models.diffusion_model.runners.base import RunnerComponents
from worldfoundry.base_models.diffusion_model.runners.strategies import (
    ExecutionBuildContext,
    build_autoregressive_window_strategy,
)
from worldfoundry.base_models.diffusion_model.schedulers.wan import (
    FastVideoCausalWanSelfForcingScheduler,
    add_flow_noise,
    flow_prediction_to_x0,
    shift_flow_sigmas,
)
from worldfoundry.pipelines.fastvideo_causal_wan import FastVideoCausalWanPipeline
from worldfoundry.pipelines.fastvideo_causal_wan.pipeline_fastvideo_causal_wan import (
    _best_output_size,
)
from worldfoundry.pipelines.native_diffusion import NativeVisualDiffusionPipeline


class _Expert(torch.nn.Module):
    num_layers = 1
    num_heads = 1
    dim = 2

    def __init__(self, marker: float) -> None:
        super().__init__()
        self.marker = float(marker)
        self.calls: list[dict[str, object]] = []

    def forward(self, **kwargs: object) -> torch.Tensor:
        x = kwargs["x"]
        assert isinstance(x, torch.Tensor)
        kv_cache = kwargs["kv_cache"]
        cross_cache = kwargs["crossattn_cache"]
        assert isinstance(kv_cache, list)
        assert isinstance(cross_cache, list)
        kv_cache[0]["k"][0, 0, 0, 0] = self.marker
        cross_cache[0]["is_init"] = True
        self.calls.append(
            {
                "kv_cache": kv_cache,
                "cross_cache": cross_cache,
                "t": kwargs["t"],
            }
        )
        return torch.full_like(x, self.marker)


def _causal_denoiser() -> tuple[FastVideoCausalWanDenoiser, _Expert, _Expert]:
    high = _Expert(2.0)
    low = _Expert(1.0)
    return (
        FastVideoCausalWanDenoiser(
            high,
            low,
            boundary_ratio=0.875,
            block_size=3,
            cache_window=3,
            text_length=4,
            compute_dtype=torch.float32,
        ),
        high,
        low,
    )


def test_fastvideo_fixed_schedule_is_shifted_once() -> None:
    scheduler = FastVideoCausalWanSelfForcingScheduler()
    sampling = SamplingConfig(
        num_inference_steps=8,
        guidance_scale=1.0,
        scheduler_options={"shift": 5.0},
    )
    schedule = scheduler.schedule(sampling, device=torch.device("cpu"), dtype=torch.float32)
    raw = torch.tensor((*scheduler.DEFAULT_RAW_TIMESTEPS, 0), dtype=torch.float32)
    expected = shift_flow_sigmas(raw / 1000.0, 5.0) * 1000.0

    torch.testing.assert_close(
        torch.stack([step.timestep for step in schedule]),
        expected[:-1],
    )
    torch.testing.assert_close(
        torch.stack([step.next_timestep for step in schedule]),
        expected[1:],
    )


def test_fastvideo_self_forcing_renoise_is_seeded_and_exact() -> None:
    scheduler = FastVideoCausalWanSelfForcingScheduler()
    schedule = scheduler.schedule(
        SamplingConfig(num_inference_steps=8),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    noisy = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 2, 2)
    flow = torch.full_like(noisy, 0.25)
    first = scheduler.step(
        flow,
        schedule[0],
        noisy,
        generator=torch.Generator().manual_seed(123),
    )
    second = scheduler.step(
        flow,
        schedule[0],
        noisy,
        generator=torch.Generator().manual_seed(123),
    )
    training_sigmas = shift_flow_sigmas(torch.linspace(1.0, 0.0, 1001)[:-1], 12.0)
    boundary_index = (training_sigmas * 1000.0 - 875.0).abs().argmin()
    assert int(boundary_index) == 632  # Official 1000-step, extra_one_step grid.
    boundary = training_sigmas[boundary_index]
    assert float(boundary) == pytest.approx(0.8748019337654114)
    torch.testing.assert_close(schedule[0].boundary_sigma, boundary)
    x_bound = flow_prediction_to_x0(flow, noisy, schedule[0].timestep / 1000.0 - boundary)
    expected_noise = torch.randn(
        x_bound.shape,
        dtype=x_bound.dtype,
        generator=torch.Generator().manual_seed(123),
    )
    next_sigma = schedule[0].next_timestep / 1000.0
    alpha = (1 - next_sigma) / (1 - boundary)
    beta = torch.sqrt(next_sigma.square() - (alpha * boundary).square())
    expected = alpha * x_bound + beta * expected_noise

    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first, expected)
    old_x0 = flow_prediction_to_x0(flow, noisy, schedule[0].timestep / 1000.0)
    assert not torch.allclose(first, add_flow_noise(old_x0, expected_noise, next_sigma))


def test_fastvideo_boundary_switch_discards_noise_but_advances_rng() -> None:
    scheduler = FastVideoCausalWanSelfForcingScheduler()
    steps = scheduler.schedule(
        SamplingConfig(num_inference_steps=8), device=torch.device("cpu"), dtype=torch.float32
    )
    switch = next(
        step for step in steps if step.timestep >= 875.0 and step.next_timestep < 875.0
    )
    noisy = torch.full((1, 2, 1, 2, 2), 0.75)
    flow = torch.full_like(noisy, 0.25)
    generator = torch.Generator().manual_seed(81)
    actual = scheduler.step(flow, switch, noisy, generator=generator)
    expected = flow_prediction_to_x0(
        flow, noisy, switch.timestep / 1000.0 - switch.boundary_sigma
    )
    torch.testing.assert_close(actual, expected)
    reference = torch.Generator().manual_seed(81)
    torch.randn(noisy.shape, generator=reference)
    torch.testing.assert_close(torch.randn(4, generator=generator), torch.randn(4, generator=reference))


def test_fastvideo_low_noise_uses_x0_with_float32_sigma() -> None:
    scheduler = FastVideoCausalWanSelfForcingScheduler()
    steps = scheduler.schedule(
        SamplingConfig(num_inference_steps=8), device=torch.device("cpu"), dtype=torch.float32
    )
    low = next(step for step in steps if step.timestep < 875.0 and step.next_timestep > 0)
    noisy = torch.full((1, 2, 1, 2, 2), 0.75, dtype=torch.bfloat16)
    flow = torch.full_like(noisy, 0.25)
    actual = scheduler.step(flow, low, noisy, generator=torch.Generator().manual_seed(82))
    x0 = flow_prediction_to_x0(flow, noisy, low.timestep / 1000.0)
    noise = torch.randn(noisy.shape, generator=torch.Generator().manual_seed(82), dtype=noisy.dtype)
    next_sigma = (low.next_timestep / 1000.0).float()
    expected = ((1 - next_sigma) * x0 + next_sigma * noise).to(noisy.dtype)
    torch.testing.assert_close(actual, expected)


def test_fastvideo_expert_boundary_commit_and_cache_independence() -> None:
    denoiser, high, low = _causal_denoiser()
    positive_cache = denoiser.create_kv_cache(
        batch_size=1,
        n_views=1,
        latent_frames_per_view=6,
        frame_sequence_length=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    negative_cache = denoiser.create_kv_cache(
        batch_size=1,
        n_views=1,
        latent_frames_per_view=6,
        frame_sequence_length=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert isinstance(positive_cache, CausalWanCacheBundle)
    assert positive_cache.high_noise is not positive_cache.low_noise
    assert positive_cache.high_noise.kv_cache[0]["k"].data_ptr() != (
        positive_cache.low_noise.kv_cache[0]["k"].data_ptr()
    )
    assert positive_cache.high_noise.kv_cache[0]["k"].data_ptr() != (
        negative_cache.high_noise.kv_cache[0]["k"].data_ptr()
    )
    assert denoiser.expert_for_timestep(torch.tensor(875.0)) == "high-noise"
    assert denoiser.expert_for_timestep(torch.tensor(874.99)) == "low-noise"
    with pytest.raises(ValueError, match="cannot mix experts"):
        denoiser.expert_for_timestep(torch.tensor([900.0, 800.0]))

    latents = torch.zeros(1, 1, 3, 2, 2)
    output = denoiser(
        DenoiserInput(
            latents=latents,
            timestep=torch.tensor(0.0),
            next_timestep=torch.tensor(0.0),
            conditioning={
                "context": torch.zeros(1, 1, 2),
                "kv_cache": positive_cache,
            },
            step_index=7,
            total_steps=8,
            branch="positive-cache-commit",
        )
    )
    assert output.extras["committed_experts"] == ("high-noise", "low-noise")
    assert len(high.calls) == len(low.calls) == 1
    assert high.calls[0]["kv_cache"] is positive_cache.high_noise.kv_cache
    assert low.calls[0]["kv_cache"] is positive_cache.low_noise.kv_cache
    assert bool(positive_cache.high_noise.cross_attention_cache[0]["is_init"])
    assert bool(positive_cache.low_noise.cross_attention_cache[0]["is_init"])


def test_fastvideo_converter_accepts_fc_in_and_fc_out() -> None:
    converted = convert_diffusers_wan_transformer_state_dict(
        {
            "blocks.0.ffn.fc_in.weight": torch.ones(1),
            "blocks.0.ffn.fc_out.bias": torch.zeros(1),
        }
    )
    assert set(converted) == {
        "blocks.0.ffn.0.weight",
        "blocks.0.ffn.2.bias",
    }


def test_fastvideo_builder_inherits_runtime_weight_dtype(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_dtypes: list[torch.dtype] = []

    class _LoadedExpert(torch.nn.Module):
        pass

    def fake_load(self, spec, checkpoint, policy):
        del self, spec, checkpoint
        observed_dtypes.append(policy.dtype)
        return _LoadedExpert()

    from worldfoundry.base_models.diffusion_model.models.denoisers import wan

    monkeypatch.setattr(wan.NativeModuleLoader, "load", fake_load)
    checkpoint = CheckpointSpec(source=tmp_path / "unused.safetensors")
    denoiser = build_fastvideo_causal_wan22_denoiser(
        ComponentBuildContext(
            model_id="fastvideo-causal-wan2.2-i2v-14b",
            key=ComponentKey(ComponentKind.DENOISER),
            policy=RuntimePolicy(device="cpu", dtype=torch.bfloat16),
            checkpoints={"high_weights": checkpoint, "low_weights": checkpoint},
        )
    )

    assert isinstance(denoiser, FastVideoCausalWanDenoiser)
    assert observed_dtypes == [torch.bfloat16, torch.bfloat16]


def test_fastvideo_vae_checkpoint_exposes_only_weight_files_to_loader() -> None:
    checkpoint = fastvideo_causal_wan22_i2v_14b_recipe().checkpoints["vae"]

    assert checkpoint.files == ("vae/diffusion_pytorch_model.safetensors",)
    assert "vae/*" in checkpoint.allow_patterns


def test_fastvideo_recipe_registry_pipeline_and_pinned_files() -> None:
    recipe = fastvideo_causal_wan22_i2v_14b_recipe()

    assert recipe.execution.strategy == "autoregressive-window"
    scheduler_component = next(
        component
        for component in recipe.components
        if component.key == recipe.execution.bindings["scheduler"]
    )
    assert scheduler_component.options["shift"] == 12.0
    assert recipe.checkpoints["high-dit"].files == (
        "transformer/diffusion_pytorch_model.safetensors",
    )
    assert recipe.checkpoints["low-dit"].files == (
        "transformer_2/diffusion_pytorch_model.safetensors",
    )
    assert recipe.checkpoints["high-dit"].revision == FASTVIDEO_CAUSAL_WAN22_REVISION
    registry = default_native_diffusion_registry()
    assert registry.resolve(FASTVIDEO_CAUSAL_WAN22_REPO_ID).model_id == (
        "fastvideo-causal-wan2.2-i2v-14b"
    )
    assert FastVideoCausalWanPipeline.DEFAULT_NUM_FRAMES == 81
    assert FastVideoCausalWanPipeline.DEFAULT_NUM_INFERENCE_STEPS == 8
    assert FastVideoCausalWanPipeline.DEFAULT_GUIDANCE_SCALE == 1.0
    assert FastVideoCausalWanPipeline.DEFAULT_SCHEDULER_OPTIONS == {"shift": 12.0}
    assert FastVideoCausalWanPipeline.GENERATION_TYPE == "i2v"
    assert FastVideoCausalWanPipeline.REQUIRES_IMAGES
    assert "image-to-video" in recipe.capabilities
    assert "text-to-video" not in recipe.capabilities


def test_fastvideo_causal_i2v_preserves_source_aspect_ratio(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, object] = {}

    def capture(self: object, **kwargs: object) -> dict[str, object]:
        del self
        observed.update(kwargs)
        return observed

    monkeypatch.setattr(NativeVisualDiffusionPipeline, "__call__", capture)
    pipeline = object.__new__(FastVideoCausalWanPipeline)
    pipeline(
        prompt="room",
        image_path="testcase/012_abandoned-room.png",
        width=832,
        height=480,
    )
    assert (observed["width"], observed["height"]) == (848, 464)
    assert observed["images"].size == (848, 464)
    assert "image_path" not in observed
    pipeline(prompt="room", images=["testcase/012_abandoned-room.png"])
    assert (observed["width"], observed["height"]) == (848, 464)
    with pytest.raises(ValueError, match="too small"):
        _best_output_size(1, 4096, 256)


class _WindowedDenoiser:
    block_size = 3
    variant = "fake-causal-wan"

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []
        self.cache_count = 0
        self.end_calls: list[tuple[str, BaseException | None]] = []

    def create_kv_cache(self, **_: object) -> dict[str, int]:
        self.cache_count += 1
        return {"cache": self.cache_count}

    def __call__(self, model_input: DenoiserInput) -> DenoiserOutput:
        self.calls.append(
            (
                model_input.branch,
                int(model_input.conditioning["current_start"]),
            )
        )
        return DenoiserOutput(torch.zeros_like(model_input.latents))

    def end_request(
        self,
        request_id: str,
        *,
        error: BaseException | None = None,
    ) -> None:
        self.end_calls.append((request_id, error))


class _Conditioner:
    def encode(self, request: DiffusionRequest, **_: object) -> Conditioning:
        del request
        return Conditioning(positive={"context": torch.zeros(1, 1, 2)})


class _Initializer:
    def initialize(self, request: DiffusionRequest, **kwargs: object) -> torch.Tensor:
        del request
        generator = kwargs["generator"]
        assert isinstance(generator, torch.Generator)
        return torch.randn(1, 1, 6, 2, 2, generator=generator)


class _Codec:
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return images

    def decode(self, latents: torch.Tensor, request: DiffusionRequest) -> torch.Tensor:
        del request
        return latents


def _fake_components(denoiser: _WindowedDenoiser) -> RunnerComponents:
    codec = _Codec()
    return RunnerComponents(
        denoiser=denoiser,
        conditioner=_Conditioner(),
        latent_initializer=_Initializer(),
        scheduler=FastVideoCausalWanSelfForcingScheduler(),
        decoder=codec,
    )


def test_fastvideo_strategy_binds_vae_encoder_for_required_image() -> None:
    recipe = fastvideo_causal_wan22_i2v_14b_recipe()
    assert "latent_encoder" in recipe.execution.bindings
    denoiser = _WindowedDenoiser()
    fake = _fake_components(denoiser)
    by_role = {
        "denoiser": fake.denoiser,
        "conditioner": fake.conditioner,
        "latent_initializer": fake.latent_initializer,
        "scheduler": fake.scheduler,
        "decoder": fake.decoder,
        "latent_encoder": fake.decoder,
    }
    components = {
        recipe.execution.bindings[role]: component for role, component in by_role.items()
    }
    runner = build_autoregressive_window_strategy(
        ExecutionBuildContext(
            recipe=recipe,
            components=components,
            policy=RuntimePolicy(),
            extensions=(),
        )
    )
    assert isinstance(runner, AutoregressiveWindowRunner)
    assert runner.components.latent_encoder is fake.decoder


class _FirstFrameInitializer:
    def initialize_with_encoder(self, request, *, latent_encoder, generator, **_):
        image = request.inputs["images"]
        assert isinstance(image, torch.Tensor)
        first = latent_encoder.encode(image)
        noise = torch.randn(1, 1, 6, 2, 2, generator=generator)
        return LatentInitialization(noise, {"first_frame_latent": first})


def test_fastvideo_i2v_seed_enters_cache_and_output() -> None:
    denoiser = _WindowedDenoiser()
    codec = _Codec()
    runner = AutoregressiveWindowRunner(
        model_id="fastvideo-fake-i2v",
        components=RunnerComponents(
            denoiser=denoiser,
            conditioner=_Conditioner(),
            latent_initializer=_FirstFrameInitializer(),
            scheduler=FastVideoCausalWanSelfForcingScheduler(),
            decoder=codec,
            latent_encoder=codec,
        ),
        prediction_mode="flow",
        device="cpu",
        dtype=torch.float32,
    )
    image = torch.full((1, 1, 1, 2, 2), 0.5)
    output = runner.run(
        DiffusionRequest(
            prompt="a test",
            height=16,
            width=16,
            num_frames=21,
            sampling=SamplingConfig(num_inference_steps=8, guidance_scale=1.0, seed=7),
            inputs={"images": image},
        )
    )
    torch.testing.assert_close(output.latents[:, :, :1], image)
    assert output.metadata["first_frame_conditioned"] is True
    sampling_calls = [call for call in denoiser.calls if not call[0].endswith("cache-commit")]
    commit_calls = [call for call in denoiser.calls if call[0].endswith("cache-commit")]
    assert {start for _, start in sampling_calls} == {1, 4}
    assert commit_calls == [
        ("positive-cache-commit", 0),
        ("positive-cache-commit", 1),
        ("positive-cache-commit", 4),
    ]


def test_fastvideo_fake_runner_traverses_three_frame_blocks_end_to_end() -> None:
    denoiser = _WindowedDenoiser()
    runner = AutoregressiveWindowRunner(
        model_id="fastvideo-fake",
        components=_fake_components(denoiser),
        prediction_mode="flow",
        device="cpu",
        dtype=torch.float32,
    )
    output = runner.run(
        DiffusionRequest(
            prompt="a test",
            height=16,
            width=16,
            num_frames=21,
            sampling=SamplingConfig(
                num_inference_steps=8,
                guidance_scale=1.0,
                seed=7,
                scheduler_options={"shift": 5.0},
            ),
        )
    )

    sampling_calls = [call for call in denoiser.calls if not call[0].endswith("cache-commit")]
    commit_calls = [call for call in denoiser.calls if call[0].endswith("cache-commit")]
    assert output.latents.shape == (1, 1, 6, 2, 2)
    assert len(sampling_calls) == 2 * 8
    assert commit_calls == [("positive-cache-commit", 0), ("positive-cache-commit", 3)]
    assert {start for _, start in sampling_calls} == {0, 3}
    assert denoiser.cache_count == 1
    assert len(denoiser.end_calls) == 1
    assert denoiser.end_calls[0][0]
    assert denoiser.end_calls[0][1] is None
