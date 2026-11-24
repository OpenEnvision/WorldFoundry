"""Native Causal-Forcing recipes, checkpoint layouts, first-block schedule, image seeding and realtime cadence."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
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
    DiffusionOutput,
    DiffusionRequest,
    LatentInitialization,
    SamplingConfig,
)
from worldfoundry.base_models.diffusion_model.extensions import DiffusionExtension
from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec
from worldfoundry.base_models.diffusion_model.models.denoisers import wan as wan_denoisers
from worldfoundry.base_models.diffusion_model.models.denoisers.wan import (
    SelfForcingWanDenoiser,
    build_self_forcing_wan21_denoiser,
)
from worldfoundry.base_models.diffusion_model.models.initializers.wan.component import (
    WanCausalForcingLatentInitializer,
    WanTextToVideoLatentInitializer,
    build_wan_causal_forcing_latent_initializer,
    build_wan_t2v_latent_initializer,
)
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.forcing.self_forcing import (
    CausalWanModel,
)
from worldfoundry.base_models.diffusion_model.optimizations import RuntimePolicy
from worldfoundry.base_models.diffusion_model.recipes.registry import default_native_diffusion_registry
from worldfoundry.base_models.diffusion_model.recipes.spec import NativeDiffusionRecipe
from worldfoundry.base_models.diffusion_model.recipes.wan import (
    CAUSAL_FORCING_CHUNKWISE_MODEL_ID,
    CAUSAL_FORCING_FRAMEWISE_1STEP_MODEL_ID,
    CAUSAL_FORCING_FRAMEWISE_2STEP_MODEL_ID,
    CAUSAL_FORCING_FRAMEWISE_MODEL_ID,
    CAUSAL_FORCING_REPO_ID,
    CAUSAL_FORCING_REVISION,
    WAN21_T2V_1P3B_REPO_ID,
    causal_forcing_chunkwise_recipe,
    causal_forcing_framewise_1step_recipe,
    causal_forcing_framewise_2step_recipe,
    causal_forcing_framewise_recipe,
)
from worldfoundry.base_models.diffusion_model.runners.autoregressive import AutoregressiveWindowRunner
from worldfoundry.base_models.diffusion_model.runners.strategies import (
    ExecutionBuildContext,
    build_autoregressive_window_strategy,
)
from worldfoundry.base_models.diffusion_model.schedulers.wan import WanFlowMatchEulerScheduler
from worldfoundry.core.nn import FlowMatchScheduler
from worldfoundry.pipelines.forcing.pipeline_native_forcing import NativeCausalForcingPipeline

TINY_CONFIG = {
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

FOUR_STEPS = (1000, 750, 500, 250)

# model_id → (recipe, checkpoint file, block size, step list, first-block step list, image input)
VARIANTS: dict[str, tuple[Any, str, int, tuple[int, ...], tuple[int, ...], bool]] = {
    CAUSAL_FORCING_CHUNKWISE_MODEL_ID: (
        causal_forcing_chunkwise_recipe,
        "chunkwise/causal_forcing.pt",
        3,
        FOUR_STEPS,
        (),
        False,
    ),
    CAUSAL_FORCING_FRAMEWISE_MODEL_ID: (
        causal_forcing_framewise_recipe,
        "framewise/causal_forcing.pt",
        1,
        FOUR_STEPS,
        (),
        True,
    ),
    CAUSAL_FORCING_FRAMEWISE_1STEP_MODEL_ID: (
        causal_forcing_framewise_1step_recipe,
        "causal-forcing++/framewise-1step.pt",
        1,
        (1000,),
        FOUR_STEPS,
        False,
    ),
    CAUSAL_FORCING_FRAMEWISE_2STEP_MODEL_ID: (
        causal_forcing_framewise_2step_recipe,
        "causal-forcing++/framewise-2step.pt",
        1,
        (1000, 500),
        FOUR_STEPS,
        False,
    ),
}


def _official_warped(steps: tuple[int, ...]) -> list[float]:
    """``CausalInferencePipeline`` step warp: ``timesteps[1000 - step]`` on the shift-5 training calendar."""

    scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(1000, training=True)
    timesteps = torch.cat((scheduler.timesteps, torch.tensor([0.0])))
    return timesteps[1000 - torch.tensor(steps)].tolist()


def _tiny_model(seed: int = 0) -> CausalWanModel:
    torch.manual_seed(seed)
    model = CausalWanModel(**TINY_CONFIG).eval()
    torch.nn.init.normal_(model.head.head.weight, std=0.02)
    return model


@pytest.mark.parametrize("model_id", sorted(VARIANTS))
def test_causal_forcing_recipe_declares_released_variant(model_id: str) -> None:
    factory, checkpoint_file, block_size, steps, first_steps, image_input = VARIANTS[model_id]
    recipe = factory()
    assert default_native_diffusion_registry().resolve(model_id).model_id == model_id
    assert recipe.model_id == model_id
    assert recipe.aliases == (("causal-forcing",) if model_id == CAUSAL_FORCING_CHUNKWISE_MODEL_ID else ())

    expected_options: dict[str, object] = {"prediction_mode": "distilled-x0", "fixed_timesteps": steps}
    if first_steps:
        expected_options["first_block_timesteps"] = first_steps
    expected_options["context_timestep"] = 0
    if block_size == 1:
        expected_options["renoise_dtype"] = "bfloat16"
    assert recipe.execution.strategy == "autoregressive-window"
    assert dict(recipe.execution.options) == expected_options

    components = {component.key: component for component in recipe.components}
    denoiser = components[recipe.execution.bindings["denoiser"]]
    assert denoiser.factory is build_self_forcing_wan21_denoiser
    assert dict(denoiser.options) == {"block_size": block_size, "cache_frames": 21, "timestep_shift": 5.0}
    initializer = components[recipe.execution.bindings["latent_initializer"]]
    assert initializer.factory is (
        build_wan_causal_forcing_latent_initializer if image_input else build_wan_t2v_latent_initializer
    )
    assert ("latent_encoder" in recipe.execution.bindings) is image_input
    assert ("image-to-video" in recipe.capabilities) is image_input
    assert {"text-to-video", "causal-generation", "causal-forcing"} <= recipe.capabilities
    assert dict(components[recipe.execution.bindings["decoder"]].options) == {
        "dtype": torch.bfloat16,
        "std_reciprocal_in_codec_dtype": True,
    }

    dit = recipe.checkpoints["dit"]
    assert (dit.repo_id, dit.revision, dit.files) == (
        CAUSAL_FORCING_REPO_ID,
        CAUSAL_FORCING_REVISION,
        (checkpoint_file,),
    )
    assert {recipe.checkpoints[name].repo_id for name in ("text-encoder", "tokenizer", "vae")} == {
        WAN21_T2V_1P3B_REPO_ID
    }
    assert recipe.options["default_num_inference_steps"] == len(steps)
    assert recipe.options["default_num_frames"] == 81
    assert recipe.metadata["latent_block_size"] == block_size


def test_public_causal_forcing_id_resolves_to_the_chunkwise_recipe() -> None:
    from worldfoundry.base_models.diffusion_model.recipes.registry import UnknownNativeDiffusionRecipeError

    registry = default_native_diffusion_registry()
    for name in ("causal-forcing", "Causal-Forcing", "causal_forcing"):
        assert registry.resolve(name).model_id == CAUSAL_FORCING_CHUNKWISE_MODEL_ID
    with pytest.raises(UnknownNativeDiffusionRecipeError):
        registry.resolve("zhuhz22/Causal-Forcing")


@pytest.mark.parametrize(
    ("role", "prefix", "block_size"),
    [
        # chunkwise/causal_forcing.pt: ``generator`` only, ``model.`` prefix.
        ("generator", "model.", 3),
        # framewise and Causal Forcing++ files: ``generator_ema`` only, FSDP-wrapped prefix.
        ("generator_ema", "model._fsdp_wrapped_module.", 1),
    ],
)
def test_causal_forcing_checkpoint_layouts_load_strictly(
    role: str,
    prefix: str,
    block_size: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = _tiny_model(seed=4)
    checkpoint = tmp_path / "causal_forcing.pt"
    torch.save({role: {prefix + name: value for name, value in reference.state_dict().items()}}, checkpoint)
    monkeypatch.setattr(wan_denoisers, "SELF_FORCING_WAN21_T2V_1P3B_CONFIG", TINY_CONFIG)

    denoiser = build_self_forcing_wan21_denoiser(
        ComponentBuildContext(
            model_id=CAUSAL_FORCING_FRAMEWISE_MODEL_ID,
            key=ComponentKey(ComponentKind.DENOISER),
            policy=RuntimePolicy(device="cpu", dtype=torch.float32),
            checkpoints={"weights": CheckpointSpec(source=checkpoint)},
            component_options={"block_size": block_size, "cache_frames": 21, "timestep_shift": 5.0},
        )
    )

    assert denoiser.block_size == block_size
    assert denoiser.model.num_frame_per_block == block_size
    loaded = denoiser.model.state_dict()
    assert loaded.keys() == reference.state_dict().keys()
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(loaded[name], value, rtol=0, atol=0)

    missing = tmp_path / "missing.pt"
    state = {prefix + name: value for name, value in reference.state_dict().items()}
    state.pop(prefix + "head.head.weight")
    torch.save({role: state}, missing)
    with pytest.raises(RuntimeError, match="Missing key"):
        build_self_forcing_wan21_denoiser(
            ComponentBuildContext(
                model_id=CAUSAL_FORCING_FRAMEWISE_MODEL_ID,
                key=ComponentKey(ComponentKind.DENOISER),
                policy=RuntimePolicy(device="cpu", dtype=torch.float32),
                checkpoints={"weights": CheckpointSpec(source=missing)},
                component_options={"block_size": block_size},
            )
        )


def test_framewise_cache_matches_official_global_cache_size() -> None:
    model = CausalWanModel(**{**TINY_CONFIG, "num_layers": 3})
    denoiser = SelfForcingWanDenoiser(model, block_size=1, compute_dtype=torch.bfloat16)
    cache = denoiser.create_kv_cache(
        batch_size=1,
        n_views=1,
        latent_frames_per_view=21,
        frame_sequence_length=1560,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    # CausalInferencePipeline._initialize_kv_cache: 32760 tokens (21 frames x 1560) per layer for any block size.
    assert [layer["k"].shape for layer in cache["kv_cache"]] == [(1, 32760, 2, 12)] * 3
    assert cache["frames_per_block"] == 1


class _Conditioner:
    def encode(self, request: DiffusionRequest, **_: object) -> Conditioning:
        seed = sum(map(ord, str(request.prompt)))
        context = torch.randn(1, 8, 16, generator=torch.Generator().manual_seed(seed))
        return Conditioning(positive={"context": context})


class _Decoder:
    def decode(self, latents: torch.Tensor, request: DiffusionRequest) -> torch.Tensor:
        del request
        return latents


class _CallRecorder(DiffusionExtension):
    extension_id = "causal-forcing-call-recorder"

    def __init__(self) -> None:
        self.calls: list[tuple[str, float, int]] = []
        self.inputs: list[torch.Tensor] = []

    def before_denoiser(self, context, model_input: DenoiserInput) -> DenoiserInput:
        del context
        current_start = model_input.conditioning["current_start"]
        assert isinstance(current_start, int)
        self.calls.append((model_input.branch, float(model_input.timestep), current_start))
        self.inputs.append(model_input.latents.clone())
        return model_input


def _runner(
    recipe: NativeDiffusionRecipe,
    *,
    block_size: int,
    cache_frames: int,
    decoder: Any = None,
    initializer: Any = None,
    extensions: tuple[DiffusionExtension, ...] = (),
) -> AutoregressiveWindowRunner:
    by_role = {
        "denoiser": SelfForcingWanDenoiser(
            _tiny_model(), block_size=block_size, cache_frames=cache_frames, compute_dtype=torch.float32
        ),
        "conditioner": _Conditioner(),
        "latent_initializer": initializer or WanTextToVideoLatentInitializer(),
        "scheduler": WanFlowMatchEulerScheduler(shift=5.0),
        "decoder": decoder or _Decoder(),
    }
    if "latent_encoder" in recipe.execution.bindings:
        by_role["latent_encoder"] = by_role["decoder"]
    runner = build_autoregressive_window_strategy(
        ExecutionBuildContext(
            recipe=recipe,
            components={recipe.execution.bindings[role]: value for role, value in by_role.items()},
            policy=RuntimePolicy(device="cpu", dtype=torch.float32),
            extensions=extensions,
        )
    )
    assert isinstance(runner, AutoregressiveWindowRunner)
    return runner


def _request(num_frames: int, steps: int, *, inputs: dict[str, Any] | None = None) -> DiffusionRequest:
    return DiffusionRequest(
        prompt="a red fox running through snow",
        height=32,
        width=32,
        num_frames=num_frames,
        sampling=SamplingConfig(
            num_inference_steps=steps,
            guidance_scale=1.0,
            seed=42,
            scheduler_options={"shift": 5.0},
        ),
        inputs=inputs or {},
    )


@pytest.mark.parametrize("model_id", sorted(VARIANTS))
def test_causal_forcing_rollout_follows_official_block_schedule(model_id: str) -> None:
    factory, _, block_size, steps, first_steps, _ = VARIANTS[model_id]
    recorder = _CallRecorder()
    runner = _runner(factory(), block_size=block_size, cache_frames=6, extensions=(recorder,))
    assert runner.fixed_timesteps == steps
    assert runner.first_block_timesteps == first_steps
    assert runner.renoise_dtype == (torch.bfloat16 if block_size == 1 else None)

    # 21 pixel frames → 6 latent frames: two chunk-wise blocks or six frame-wise blocks.
    output = runner.run(_request(21, len(steps)))
    assert output.latents.shape == (1, 16, 6, 4, 4)
    assert bool(torch.isfinite(output.latents).all())
    assert output.metadata["block_size"] == block_size

    frame_tokens = 4 * 4 // 4
    expected = []
    for block_index, block_start in enumerate(range(0, 6, block_size)):
        block_steps = first_steps if block_index == 0 and first_steps else steps
        current_start = block_start * frame_tokens
        expected.extend(("positive", step, current_start) for step in _official_warped(block_steps))
        expected.append(("positive-cache-commit", 0.0, current_start))
    assert recorder.calls == expected

    rerun = runner.run(_request(21, len(steps)))
    torch.testing.assert_close(rerun.latents, output.latents, rtol=0, atol=0)


def test_first_block_steps_need_the_released_step_count() -> None:
    runner = _runner(causal_forcing_framewise_1step_recipe(), block_size=1, cache_frames=3)
    with pytest.raises(ValueError, match="requires 1 inference steps"):
        runner.run(_request(9, 4))


def test_framewise_initializer_without_image_draws_t2v_noise() -> None:
    request = _request(9, 4, inputs={"images": None})
    kwargs = {"device": torch.device("cpu"), "dtype": torch.float32}
    expected = WanTextToVideoLatentInitializer().initialize(
        request, generator=torch.Generator().manual_seed(3), **kwargs
    )
    initializer = WanCausalForcingLatentInitializer()
    plain = initializer.initialize(request, generator=torch.Generator().manual_seed(3), **kwargs)
    encoded = initializer.initialize_with_encoder(
        request,
        latent_encoder=cast(Any, None),
        generator=torch.Generator().manual_seed(3),
        **kwargs,
    )
    assert isinstance(encoded, torch.Tensor)
    torch.testing.assert_close(plain, expected, rtol=0, atol=0)
    torch.testing.assert_close(encoded, expected, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="latent_encoder"):
        initializer.initialize(
            _request(9, 4, inputs={"images": np.zeros((8, 8, 3), np.uint8)}),
            generator=torch.Generator(),
            **kwargs,
        )


def test_framewise_initializer_prepares_the_image_as_official_i2v() -> None:
    from PIL import Image
    from torchvision import transforms

    image = Image.fromarray(
        np.random.default_rng(0).integers(0, 256, size=(23, 41, 3), dtype=np.uint8),
    )
    seen: list[torch.Tensor] = []

    class _Encoder:
        dtype = torch.bfloat16

        def encode(self, images: torch.Tensor) -> torch.Tensor:
            seen.append(images)
            return torch.full((images.shape[0], 16, 1, 4, 4), 0.5, dtype=images.dtype)

    initialized = WanCausalForcingLatentInitializer().initialize_with_encoder(
        _request(9, 4, inputs={"images": [image]}),
        latent_encoder=_Encoder(),
        generator=torch.Generator().manual_seed(1),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    # inference.py --i2v: Resize((480, 832)), ToTensor, Normalize([0.5], [0.5]), then [1, C, 1, H, W] in bf16.
    official = transforms.Compose(
        [transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    )(image.convert("RGB"))
    official = official[None, :, None].to(dtype=torch.bfloat16)
    torch.testing.assert_close(seen[0], official, rtol=0, atol=0)
    assert isinstance(initialized, LatentInitialization)
    assert initialized.latents.shape == (1, 16, 3, 4, 4)
    first = initialized.conditioning["first_frame_latent"]
    assert isinstance(first, torch.Tensor)
    assert first.dtype == torch.float32
    torch.testing.assert_close(first, torch.full((1, 16, 1, 4, 4), 0.5), rtol=0, atol=0)


def _tiny_codec() -> Any:
    from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.component import WanVideoDecoder
    from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.model import WanVideoVAE

    torch.manual_seed(11)
    return WanVideoDecoder(WanVideoVAE(z_dim=16, base_dim=8), device=torch.device("cpu"))


def _image() -> np.ndarray:
    return np.random.default_rng(5).integers(0, 256, size=(32, 32, 3), dtype=np.uint8)


def test_framewise_image_seeds_the_cache_then_rolls_out_remaining_frames() -> None:
    recorder = _CallRecorder()
    codec = _tiny_codec()
    runner = _runner(
        causal_forcing_framewise_recipe(),
        block_size=1,
        cache_frames=3,
        decoder=codec,
        initializer=WanCausalForcingLatentInitializer(),
        extensions=(recorder,),
    )
    output = runner.run(_request(9, 4, inputs={"images": _image()}))

    assert output.latents.shape == (1, 16, 3, 4, 4)
    assert output.metadata["first_frame_conditioned"] is True
    frame_tokens = 4
    steps = _official_warped(FOUR_STEPS)
    expected = [("positive-cache-commit", 0.0, 0)]
    for frame in (1, 2):
        expected.extend(("positive", step, frame * frame_tokens) for step in steps)
        expected.append(("positive-cache-commit", 0.0, frame * frame_tokens))
    assert recorder.calls == expected
    torch.testing.assert_close(recorder.inputs[0], output.latents[:, :, :1], rtol=0, atol=0)


def _pipeline(model_id: str) -> tuple[NativeCausalForcingPipeline, AutoregressiveWindowRunner]:
    from worldfoundry.base_models.diffusion_model.pipeline import NativeDiffusionPipeline

    factory, _, block_size, _, _, image_input = VARIANTS[model_id]
    runner = _runner(
        factory(),
        block_size=block_size,
        cache_frames=21,
        decoder=_tiny_codec(),
        initializer=WanCausalForcingLatentInitializer() if image_input else WanTextToVideoLatentInitializer(),
    )
    pipeline = NativeCausalForcingPipeline(native_pipeline=NativeDiffusionPipeline(runner), device="cpu")
    return pipeline, runner


REALTIME_CALL: dict[str, Any] = {"height": 32, "width": 32, "num_frames": 13}


def _stream_all(pipeline: NativeCausalForcingPipeline) -> list[dict[str, Any]]:
    chunks = []
    while True:
        chunks.append(pipeline.stream_realtime())
        metrics = chunks[-1]["realtime_metrics"]
        if metrics["chunk_index"] == metrics["chunks_total"]:
            return chunks


@pytest.mark.parametrize(
    ("model_id", "first", "steady", "max_chunks"),
    [
        (CAUSAL_FORCING_CHUNKWISE_MODEL_ID, 9, 12, 7),
        (CAUSAL_FORCING_FRAMEWISE_MODEL_ID, 1, 4, 21),
        (CAUSAL_FORCING_FRAMEWISE_1STEP_MODEL_ID, 1, 4, 21),
        (CAUSAL_FORCING_FRAMEWISE_2STEP_MODEL_ID, 1, 4, 21),
    ],
)
def test_causal_forcing_realtime_spec_follows_the_codec(
    model_id: str, first: int, steady: int, max_chunks: int
) -> None:
    pipeline, _ = _pipeline(model_id)
    spec = pipeline.prepare_realtime()["realtime_spec"]
    assert (spec["first_chunk_frames"], spec["steady_chunk_frames"], spec["max_chunks"]) == (first, steady, max_chunks)
    assert (spec["fps"], spec["advance"], spec["controls"], spec["keys"]) == (16, "continuous", ["prompt_update"], [])


@pytest.mark.parametrize(
    ("model_id", "num_frames", "chunk_frames"),
    [
        (CAUSAL_FORCING_FRAMEWISE_1STEP_MODEL_ID, 13, [1, 4, 4, 4]),
        (CAUSAL_FORCING_FRAMEWISE_2STEP_MODEL_ID, 13, [1, 4, 4, 4]),
        (CAUSAL_FORCING_CHUNKWISE_MODEL_ID, 21, [9, 12]),
    ],
)
def test_causal_forcing_realtime_stream_matches_offline_call_bitwise(
    model_id: str, num_frames: int, chunk_frames: list[int]
) -> None:
    from worldfoundry.core.media.codecs.video import video_tensor_to_uint8_frames

    pipeline, runner = _pipeline(model_id)
    call: dict[str, Any] = {**REALTIME_CALL, "num_frames": num_frames}
    offline = pipeline(prompt="fox", seed=7, return_dict=True, **call)
    expected = video_tensor_to_uint8_frames(offline["video"], value_range="-1,1", layout="bcthw")

    configured = pipeline.configure_realtime(prompt="fox", seed=7, **call)
    assert configured["realtime_spec"]["max_chunks"] == len(chunk_frames)
    chunks = _stream_all(pipeline)

    assert [chunk["video"].shape[0] for chunk in chunks] == chunk_frames
    np.testing.assert_array_equal(np.concatenate([chunk["video"] for chunk in chunks]), expected)
    assert runner._active_stream is None


def test_framewise_image_realtime_emits_the_image_frame_with_the_first_block() -> None:
    from worldfoundry.core.media.codecs.video import video_tensor_to_uint8_frames

    pipeline, runner = _pipeline(CAUSAL_FORCING_FRAMEWISE_MODEL_ID)
    image = _image()
    offline = pipeline(prompt="fox", images=image, seed=3, return_dict=True, **REALTIME_CALL)
    expected = video_tensor_to_uint8_frames(offline["video"], value_range="-1,1", layout="bcthw")

    configured = pipeline.configure_realtime(images=image, prompt="fox", seed=3, **REALTIME_CALL)
    spec = configured["realtime_spec"]
    assert (spec["first_chunk_frames"], spec["steady_chunk_frames"], spec["max_chunks"]) == (5, 4, 3)
    chunks = _stream_all(pipeline)
    assert [chunk["video"].shape[0] for chunk in chunks] == [5, 4, 4]
    np.testing.assert_array_equal(np.concatenate([chunk["video"] for chunk in chunks]), expected)

    # A prompt update reopens the image-seeded rollout with the configured image and seed.
    pipeline.configure_realtime(images=image, prompt="fox", seed=3, **REALTIME_CALL)
    pipeline.stream_realtime()
    switched = pipeline.stream_realtime(prompt="owl")
    assert switched["video"].shape[0] == 5
    assert switched["realtime_metrics"]["chunks_total"] == 3
    pipeline.reset_realtime()
    assert runner._active_stream is None


def test_native_causal_forcing_pipeline_selects_variant_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    from worldfoundry.pipelines import native_diffusion

    requests: list[DiffusionRequest] = []

    class _NativePipeline:
        def __init__(self, model_id: str) -> None:
            self.model_id = model_id

        def __call__(self, request: DiffusionRequest) -> DiffusionOutput:
            requests.append(request)
            return DiffusionOutput(sample=torch.zeros(1, 3, 81, 8, 8), latents=torch.zeros(1, 16, 21, 1, 1))

    for model_id, (_, _, _, steps, _, image_input) in VARIANTS.items():
        pipeline = NativeCausalForcingPipeline(native_pipeline=cast(Any, _NativePipeline(model_id)), device="cpu")
        result = pipeline(prompt="a red fox", seed=1, return_dict=True)
        request = requests[-1]
        assert request.sampling.num_inference_steps == len(steps)
        assert (request.height, request.width, request.num_frames) == (480, 832, 81)
        assert request.sampling.guidance_scale == 1.0
        assert pipeline.ACCEPTS_IMAGES is image_input
        assert result["generation_type"] == ("ti2v" if image_input else "t2v")
        assert ("images" in request.inputs) is image_input
        if not image_input:
            with pytest.raises(ValueError, match="does not accept image inputs"):
                pipeline(prompt="a red fox", images=np.zeros((8, 8, 3), np.uint8))

    with pytest.raises(ValueError, match="not a native Causal-Forcing recipe"):
        NativeCausalForcingPipeline(native_pipeline=cast(Any, _NativePipeline("self-forcing")), device="cpu")

    captured: dict[str, Any] = {}

    def load(model_id: str, **kwargs: Any) -> Any:
        captured.update(model_id=model_id, **kwargs)
        return SimpleNamespace(model_id=default_native_diffusion_registry().resolve(model_id).model_id)

    monkeypatch.setattr(native_diffusion.NativeDiffusionPipeline, "from_pretrained", load)
    loaded = NativeCausalForcingPipeline.from_pretrained(
        device="cpu",
        model_id=CAUSAL_FORCING_FRAMEWISE_2STEP_MODEL_ID,
        checkpoint_path="/ckpt/framewise-2step.pt",
    )
    assert captured["model_id"] == CAUSAL_FORCING_FRAMEWISE_2STEP_MODEL_ID
    assert captured["checkpoint_overrides"] == {"dit": "/ckpt/framewise-2step.pt"}
    assert loaded.DEFAULT_NUM_INFERENCE_STEPS == 2
    NativeCausalForcingPipeline.from_pretrained(device="cpu", variant=CAUSAL_FORCING_FRAMEWISE_MODEL_ID)
    assert captured["model_id"] == CAUSAL_FORCING_FRAMEWISE_MODEL_ID
    default = NativeCausalForcingPipeline.from_pretrained(device="cpu")
    assert captured["model_id"] == "causal-forcing"
    assert default.model_id == CAUSAL_FORCING_CHUNKWISE_MODEL_ID
