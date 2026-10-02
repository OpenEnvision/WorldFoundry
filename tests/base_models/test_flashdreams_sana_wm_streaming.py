from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

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
from worldfoundry.base_models.diffusion_model.models.denoisers.sana import SanaDenoiser
from worldfoundry.base_models.diffusion_model.models.denoisers.sana_refiner import (
    SanaWMLTX2RefinerProcessor,
    build_sana_wm_ltx2_refiner_processor,
)
from worldfoundry.base_models.diffusion_model.models.encoders.sana.refiner import (
    SanaWMRefinerConditioner,
    build_sana_wm_refiner_conditioner,
)
from worldfoundry.base_models.diffusion_model.models.initializers.sana import (
    SanaWorldInitializer,
)
from worldfoundry.base_models.diffusion_model.optimizations import (
    OffloadMode,
    OffloadPolicy,
    RuntimePolicy,
)
from worldfoundry.base_models.diffusion_model.recipes.registry import (
    default_native_diffusion_registry,
)
from worldfoundry.base_models.diffusion_model.recipes.sana import (
    SANA_WM_STREAMING_REVISION,
    sana_wm_streaming_recipe,
)
from worldfoundry.base_models.diffusion_model.runners.base import RunnerComponents
from worldfoundry.base_models.diffusion_model.runners.prefix_recompute import (
    PrefixRecomputeRunner,
    prefix_chunk_boundaries,
    prefix_recompute_window,
    remap_condition_frame_info,
    slice_prefix_conditioning,
)
from worldfoundry.base_models.diffusion_model.schedulers.sana import (
    SanaWMStreamingEulerScheduler,
)
from worldfoundry.pipelines.sana_wm import SanaWMStreamingPipeline


def _request(*, num_frames: int = 241) -> DiffusionRequest:
    return DiffusionRequest(
        prompt="a moving world",
        height=704,
        width=1280,
        num_frames=num_frames,
        sampling=SamplingConfig(
            num_inference_steps=4,
            guidance_scale=1.0,
            seed=3,
            scheduler_options={"shift": 8.0},
        ),
        inputs={"fps": 16, "refiner_seed": 5},
    )


def test_sana_streaming_241_pixels_map_to_31_latents_and_ten_chunks() -> None:
    initializer = SanaWorldInitializer(
        channels=128,
        spatial_compression=32,
        temporal_compression=8,
    )
    assert initializer.latent_shape(_request()) == (1, 128, 31, 22, 40)
    assert prefix_chunk_boundaries(31, 3) == (
        0,
        4,
        7,
        10,
        13,
        16,
        19,
        22,
        25,
        28,
        31,
    )


def test_sana_streaming_windows_keep_anchor_and_six_recent_frames() -> None:
    first = prefix_recompute_window(start=0, end=4, history_frames=6, sink_frames=1)
    assert first.frame_indices == (0, 1, 2, 3)
    assert (first.active_start, first.active_end) == (1, 4)

    later = prefix_recompute_window(start=10, end=13, history_frames=6, sink_frames=1)
    assert later.frame_indices == (0, 4, 5, 6, 7, 8, 9, 10, 11, 12)
    assert (later.active_start, later.active_end) == (7, 10)


def test_sana_streaming_slices_camera_pluecker_and_remaps_frame_info() -> None:
    frames = 31
    camera = torch.arange(frames * 2).reshape(1, frames, 2)
    pluecker = torch.arange(frames).reshape(1, 1, frames, 1, 1)
    window = prefix_recompute_window(start=28, end=31, history_frames=6, sink_frames=1)
    assert window.frame_indices == (0, 22, 23, 24, 25, 26, 27, 28, 29, 30)
    assert remap_condition_frame_info(
        {"0": 0.0, 22: 0.25, 30: 1.0, "bad": 9.0},
        window.frame_indices,
    ) == {0: 0.0, 1: 0.25, 9: 1.0}

    sliced = slice_prefix_conditioning(
        {
            "camera_conditions": camera,
            "chunk_plucker": pluecker,
            "condition_frame_info": {0: 0.0},
        },
        frame_indices=window.frame_indices,
        active_start=window.active_start,
        active_end=window.active_end,
        total_frames=frames,
        chunk_size=3,
    )
    assert sliced["camera_conditions"].flatten().tolist() == (
        camera[:, window.frame_indices].flatten().tolist()
    )
    assert sliced["chunk_plucker"].flatten().tolist() == list(window.frame_indices)
    assert sliced["condition_frame_info"] == {index: 0.0 for index in range(7)}
    assert sliced["chunk_index"] == [0, 4, 7]


def test_sana_streaming_scheduler_uses_released_values_without_second_shift() -> None:
    scheduler = SanaWMStreamingEulerScheduler()
    schedule = scheduler.schedule(
        _request().sampling,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert [float(step.timestep) for step in schedule] == [1000.0, 960.0, 889.0, 727.0]
    assert [float(step.next_timestep) for step in schedule] == [960.0, 889.0, 727.0, 0.0]
    updated = scheduler.step(
        torch.ones(1),
        schedule[0],
        torch.zeros(1),
        generator=torch.Generator().manual_seed(0),
    )
    torch.testing.assert_close(updated, torch.tensor([-0.04]))
    with pytest.raises(ValueError, match="exactly 4"):
        scheduler.schedule(
            SamplingConfig(num_inference_steps=5),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


class _SanaModel:
    def __init__(self) -> None:
        self.timestep: torch.Tensor | None = None

    def __call__(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        **_: object,
    ) -> torch.Tensor:
        del context
        self.timestep = timestep.detach().clone()
        return torch.zeros_like(latents)


def test_sana_denoiser_preserves_temporal_clean_context_timestep() -> None:
    model = _SanaModel()
    denoiser = SanaDenoiser(model)  # type: ignore[arg-type]
    latents = torch.zeros(1, 1, 4, 1, 1)
    denoiser(
        DenoiserInput(
            latents=latents,
            timestep=torch.tensor([[[0.0, 1000.0, 1000.0, 1000.0]]]),
            next_timestep=torch.zeros(1, 1, 4),
            conditioning={
                "context": torch.zeros(1, 1, 2),
                "context_mask": torch.ones(1, 1),
                "condition_frame_info": {0: 0.0},
            },
            step_index=0,
            total_steps=4,
        )
    )
    assert model.timestep is not None
    torch.testing.assert_close(
        model.timestep,
        torch.tensor([[[0.0, 1000.0, 1000.0, 1000.0]]]),
    )


class _Conditioner:
    def encode(self, request: DiffusionRequest, **_: object) -> Conditioning:
        total = (request.num_frames - 1) // 8 + 1
        return Conditioning(
            positive={
                "context": torch.zeros(1, 1, 2),
                "context_mask": torch.ones(1, 1),
            },
            shared={
                "camera_conditions": torch.arange(total * 2).reshape(1, total, 2),
                "chunk_plucker": torch.arange(total).reshape(1, 1, total, 1, 1),
            },
        )


class _Initializer:
    def initialize(self, *_: object, **__: object) -> torch.Tensor:
        raise RuntimeError("encoded initialization is required")

    def initialize_with_encoder(
        self,
        request: DiffusionRequest,
        **_: object,
    ) -> LatentInitialization:
        total = (request.num_frames - 1) // 8 + 1
        latents = torch.arange(total, dtype=torch.float32).reshape(1, 1, total, 1, 1)
        latents[:, :, 0] = 99.0
        return LatentInitialization(
            latents,
            conditioning={"condition_frame_info": {0: 0.0}},
        )


class _Codec:
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return images

    def decode(self, latents: torch.Tensor, request: DiffusionRequest) -> torch.Tensor:
        del request
        return latents


class _Denoiser:
    def __init__(self) -> None:
        self.calls: list[DenoiserInput] = []
        self.end_calls: list[tuple[str, BaseException | None]] = []

    def __call__(self, model_input: DenoiserInput) -> DenoiserOutput:
        self.calls.append(model_input)
        return DenoiserOutput(torch.ones_like(model_input.latents))

    def end_request(
        self,
        request_id: str,
        *,
        error: BaseException | None = None,
    ) -> None:
        self.end_calls.append((request_id, error))


class _RefinerConditioner:
    def encode(self, request: DiffusionRequest, **_: object) -> Conditioning:
        del request
        return Conditioning(positive={"refiner_context": torch.ones(1)})


class _Refiner:
    def __init__(self) -> None:
        self.contexts: list[torch.Tensor] = []
        self.active: list[torch.Tensor] = []

    def refine_active_latents(self, **kwargs: object) -> torch.Tensor:
        context = kwargs["context_latents"]
        active = kwargs["active_latents"]
        assert isinstance(context, torch.Tensor)
        assert isinstance(active, torch.Tensor)
        self.contexts.append(context.detach().clone())
        self.active.append(active.detach().clone())
        return active + 100.0


def test_sana_prefix_runner_executes_stage1_and_refiner_end_to_end() -> None:
    denoiser = _Denoiser()
    refiner = _Refiner()
    codec = _Codec()
    runner = PrefixRecomputeRunner(
        model_id="sana-wm-streaming-fake",
        components=RunnerComponents(
            denoiser=denoiser,
            conditioner=_Conditioner(),
            latent_initializer=_Initializer(),
            scheduler=SanaWMStreamingEulerScheduler(),
            decoder=codec,
            latent_encoder=codec,
        ),
        refiner=refiner,
        refiner_conditioner=_RefinerConditioner(),
        chunk_size=3,
        history_frames=6,
        sink_frames=1,
        refiner_max_frames=11,
        device="cpu",
        dtype=torch.float32,
    )
    output = runner.run(_request())

    assert len(denoiser.calls) == 10 * 4
    assert len(refiner.contexts) == 10
    assert output.metadata["num_chunks"] == 10
    assert output.latents.shape == (1, 1, 31, 1, 1)
    torch.testing.assert_close(
        denoiser.calls[0].timestep.flatten(),
        torch.tensor([0.0, 1000.0, 1000.0, 1000.0]),
    )
    # The next chunk sees the fixed anchor, completed first chunk, and untouched active noise.
    torch.testing.assert_close(
        denoiser.calls[4].latents.flatten(),
        torch.tensor([99.0, 0.0, 1.0, 2.0, 4.0, 5.0, 6.0]),
    )
    assert denoiser.calls[-4].conditioning["camera_conditions"].flatten().tolist() == [
        0,
        1,
        *list(range(44, 62)),
    ]
    assert denoiser.calls[-4].conditioning["condition_frame_info"] == {
        index: 0.0 for index in range(7)
    }
    assert refiner.contexts[0].flatten().tolist() == [99.0]
    assert refiner.contexts[1].flatten().tolist() == [99.0, 100.0, 101.0, 102.0]
    assert refiner.contexts[-1].shape[2] == 8
    assert output.latents.flatten()[0].item() == 99.0
    assert output.latents.flatten()[1].item() == 100.0
    assert output.latents.flatten()[-1].item() == 129.0
    assert len(denoiser.end_calls) == 1
    assert denoiser.end_calls[0][0] == denoiser.calls[0].request_id
    assert denoiser.end_calls[0][1] is None


class _RefinerTransformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            patch_size=1,
            patch_size_t=1,
            audio_in_channels=2,
            timestep_scale_multiplier=1000.0,
        )
        self.kwargs: dict[str, object] = {}
        self.moves: list[torch.device] = []

    def to(self, *args, **kwargs):
        raw_device = kwargs.get("device", args[0] if args else None)
        if raw_device is not None:
            self.moves.append(torch.device(raw_device))
        return super().to(*args, **kwargs)

    def forward(self, **kwargs: object) -> tuple[torch.Tensor]:
        self.kwargs = kwargs
        hidden = kwargs["hidden_states"]
        assert isinstance(hidden, torch.Tensor)
        return (torch.zeros_like(hidden),)


def test_sana_refiner_masks_context_queries_and_uses_dummy_audio() -> None:
    transformer = _RefinerTransformer()
    processor = SanaWMLTX2RefinerProcessor(
        transformer,
        device="cpu",
        dtype=torch.float32,
    )
    velocity = processor._velocity(
        context=torch.zeros(1, 2, 1, 1, 1),
        active=torch.zeros(1, 2, 3, 1, 1),
        sigma=0.5,
        conditioning={
            "refiner_video_context": torch.zeros(1, 1, 2),
            "refiner_audio_context": torch.zeros(1, 1, 2),
            "refiner_context_mask": torch.ones(1, 1),
        },
        fps=16.0,
    )
    assert velocity.shape == (1, 3, 2)
    timestep = transformer.kwargs["timestep"]
    assert isinstance(timestep, torch.Tensor)
    torch.testing.assert_close(timestep, torch.tensor([[0.0, 500.0, 500.0, 500.0]]))
    mask = transformer.kwargs["video_self_attention_mask"]
    assert isinstance(mask, torch.Tensor)
    assert mask[0, 0].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert mask[0, 1:].bool().all()
    assert transformer.kwargs["isolate_modalities"] is True
    audio = transformer.kwargs["audio_hidden_states"]
    assert isinstance(audio, torch.Tensor)
    assert torch.count_nonzero(audio) == 0


def test_sana_refiner_stages_transformer_around_active_chunk() -> None:
    transformer = _RefinerTransformer()
    processor = SanaWMLTX2RefinerProcessor(
        transformer,
        device="cpu",
        dtype=torch.float32,
        sigmas=(0.5, 0.0),
        offload_after_refine=True,
    )
    output = processor.refine_active_latents(
        context_latents=torch.zeros(1, 2, 1, 1, 1),
        active_latents=torch.zeros(1, 2, 1, 1, 1),
        conditioning={
            "refiner_video_context": torch.zeros(1, 1, 2),
            "refiner_audio_context": torch.zeros(1, 1, 2),
            "refiner_context_mask": torch.ones(1, 1),
        },
        fps=16.0,
        generator=torch.Generator().manual_seed(0),
    )

    assert output.shape == (1, 2, 1, 1, 1)
    assert transformer.moves == [torch.device("cpu"), torch.device("cpu")]


def test_sana_refiner_conditioner_offloads_gemma_and_connectors_after_encode() -> None:
    class _Tokenizer:
        padding_side = "right"
        pad_token = None
        eos_token = "</s>"

        def __call__(self, prompts, **kwargs):
            del prompts, kwargs
            return SimpleNamespace(
                input_ids=torch.ones(1, 2, dtype=torch.long),
                attention_mask=torch.ones(1, 2, dtype=torch.long),
            )

    class _Backbone(torch.nn.Module):
        def forward(self, **kwargs):
            del kwargs
            return SimpleNamespace(
                hidden_states=(
                    torch.zeros(1, 2, 1),
                    torch.ones(1, 2, 1),
                )
            )

    class _TextEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = _Backbone()
            self.moves: list[torch.device] = []

        def to(self, *args, **kwargs):
            raw_device = kwargs.get("device", args[0] if args else None)
            if raw_device is not None:
                self.moves.append(torch.device(raw_device))
            return super().to(*args, **kwargs)

    class _Connectors(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.moves: list[torch.device] = []

        def to(self, *args, **kwargs):
            raw_device = kwargs.get("device", args[0] if args else None)
            if raw_device is not None:
                self.moves.append(torch.device(raw_device))
            return super().to(*args, **kwargs)

        def forward(self, packed, attention_mask):
            return packed, packed, attention_mask

    text_encoder = _TextEncoder()
    connectors = _Connectors()
    conditioner = SanaWMRefinerConditioner(
        text_encoder,
        _Tokenizer(),
        connectors,
        max_length=2,
        offload_after_encode=True,
    )
    encoded = conditioner.encode(
        DiffusionRequest(prompt="test"),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert encoded.positive["refiner_video_context"].shape == (1, 2, 2)
    assert text_encoder.moves == [torch.device("cpu"), torch.device("cpu")]
    assert connectors.moves == [torch.device("cpu"), torch.device("cpu")]


def test_sana_streaming_recipe_registry_pipeline_and_pinned_files() -> None:
    recipe = sana_wm_streaming_recipe()
    assert recipe.execution.strategy == "prefix-recompute"
    assert recipe.checkpoints["dit"].files == ("sana_dit/model.pt",)
    assert recipe.checkpoints["codec"].files == (
        "ltx2_causal_vae/diffusion_pytorch_model.safetensors",
    )
    assert recipe.checkpoints["refiner"].files == (
        "refiner_diffusers/transformer/diffusion_pytorch_model.safetensors",
    )
    assert recipe.checkpoints["dit"].revision == SANA_WM_STREAMING_REVISION
    registry = default_native_diffusion_registry()
    assert registry.resolve("Efficient-Large-Model/SANA-WM_streaming").model_id == (
        "sana-wm-streaming"
    )
    assert SanaWMStreamingPipeline.DEFAULT_NUM_FRAMES == 241
    assert SanaWMStreamingPipeline.DEFAULT_NUM_INFERENCE_STEPS == 4
    assert SanaWMStreamingPipeline.DEFAULT_GUIDANCE_SCALE == 1.0


def test_sana_refiner_factories_use_local_diffusers_directories(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, tuple[str, dict[str, object]]] = {}

    class _Materialized:
        def directory(self, relative: str):
            return tmp_path / relative

    def materialize(*_: object, **__: object) -> _Materialized:
        return _Materialized()

    from worldfoundry.base_models.diffusion_model.models.denoisers import sana_refiner
    from worldfoundry.base_models.diffusion_model.models.encoders.sana import refiner as encoder_refiner

    monkeypatch.setattr(
        sana_refiner.NativeCheckpointResolver,
        "materialize",
        materialize,
    )
    monkeypatch.setattr(
        encoder_refiner.NativeCheckpointResolver,
        "materialize",
        materialize,
    )

    class _Loader:
        @staticmethod
        def from_pretrained(path: str, **kwargs: object):
            calls["transformer"] = (str(path), dict(kwargs))
            return _RefinerTransformer()

    class _Connectors(torch.nn.Module):
        @staticmethod
        def from_pretrained(path: str, **kwargs: object):
            calls["connectors"] = (str(path), dict(kwargs))
            return _Connectors()

    class _TextEncoder(torch.nn.Module):
        @staticmethod
        def from_pretrained(path: str, **kwargs: object):
            calls["text"] = (str(path), dict(kwargs))
            return _TextEncoder()

    class _Tokenizer:
        padding_side = "right"
        pad_token = None
        eos_token = "</s>"

        @staticmethod
        def from_pretrained(path: str, **kwargs: object):
            calls["tokenizer"] = (str(path), dict(kwargs))
            return _Tokenizer()

    # The factories' import boundary is exercised with fake loaders; optional
    # model packages are not prerequisites for this CPU checkpoint contract.
    diffusers = ModuleType("diffusers")
    diffusers.LTX2VideoTransformer3DModel = _Loader
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    ltx2_module = ModuleType("diffusers.pipelines.ltx2")
    ltx2_module.LTX2TextConnectors = _Connectors
    monkeypatch.setitem(sys.modules, "diffusers.pipelines.ltx2", ltx2_module)
    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = _Tokenizer
    transformers.Gemma3ForConditionalGeneration = _TextEncoder
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    checkpoint = CheckpointSpec(source=tmp_path)
    policy = RuntimePolicy(
        device="cuda",
        dtype=torch.float32,
        offload=OffloadPolicy(
            mode=OffloadMode.BLOCK,
            target="cpu",
            pin_memory=True,
        ),
    )
    processor = build_sana_wm_ltx2_refiner_processor(
        ComponentBuildContext(
            model_id="sana-wm-streaming",
            key=ComponentKey(ComponentKind.LATENT_PROCESSOR, "refiner"),
            policy=policy,
            checkpoints={"weights": checkpoint},
        )
    )
    conditioner = build_sana_wm_refiner_conditioner(
        ComponentBuildContext(
            model_id="sana-wm-streaming",
            key=ComponentKey(ComponentKind.CONDITIONER, "refiner"),
            policy=policy,
            checkpoints={"weights": checkpoint, "connectors": checkpoint},
        )
    )
    assert isinstance(processor, SanaWMLTX2RefinerProcessor)
    assert processor.offload_after_refine is True
    assert conditioner.offload_after_encode is True
    assert conditioner.tokenizer.padding_side == "left"
    assert calls["transformer"][0].endswith("refiner_diffusers/transformer")
    assert calls["connectors"][0].endswith("refiner_diffusers/connectors")
    assert calls["text"][0].endswith("gemma3_12b")
    assert calls["tokenizer"][0].endswith("gemma3_12b")
    assert calls["transformer"][1]["local_files_only"] is True
    assert calls["connectors"][1]["local_files_only"] is True
    assert calls["text"][1]["local_files_only"] is True
    assert calls["tokenizer"][1]["local_files_only"] is True
