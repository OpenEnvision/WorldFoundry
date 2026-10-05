from __future__ import annotations

import pytest

# This test module imports worldfoundry code that requires the optional
# "diffusers" dependency at import time; skip when it is unavailable.
pytest.importorskip("diffusers")

import asyncio
import inspect
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from worldfoundry.pipelines.helios.pipeline_helios import HeliosPipeline
from worldfoundry.studio.inference.catalog import find_entry
from worldfoundry.studio.inference.dispatch import _requested_torchrun_nproc
from worldfoundry.studio.inference.execution import (
    TORCHRUN_LINGBOT_FAST_ENV,
    BaseRuntimeDriver,
    PipelineContext,
    StudioManager,
)
from worldfoundry.studio.ui.launch_config import StudioLaunchConfig, launch_uses_lingbot_torchrun_rollout
from worldfoundry.studio.visualization.backends.world import world_frontend_html
from worldfoundry.studio.visualization.backends.world_realtime import (
    LatestFrameBuffer,
    RealtimeControlResampler,
    RealtimePeerManager,
    _ActivePeer,
    _ActiveSocket,
    _prefer_websocket_transport,
)
from worldfoundry.studio.visualization.backends.world_realtime_client import (
    WORLD_REALTIME_CLIENT_JS,
)
from worldfoundry.synthesis.visual_generation.helios import realtime
from worldfoundry.synthesis.visual_generation.helios import transformer_helios_diffusers as helios_transformer


def _checkpoint(tmp_path: Path, *, distilled: bool = True) -> Path:
    root = tmp_path / "Helios-Distilled"
    for relative in realtime._REQUIRED_CHECKPOINT_PATHS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    (root / "model_index.json").write_text(
        json.dumps({"is_distilled": distilled}), encoding="utf-8"
    )
    return root


def test_checkpoint_gate_accepts_only_distilled(tmp_path: Path) -> None:
    distilled = _checkpoint(tmp_path)
    assert realtime._resolve_checkpoint(distilled) == distilled.resolve()

    (distilled / "model_index.json").write_text(
        json.dumps({"is_distilled": False}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Helios-Distilled"):
        realtime._resolve_checkpoint(distilled)


def test_native_spec_is_prompt_scheduled_not_keyboard_controlled() -> None:
    session = object.__new__(realtime.HeliosRealtimeSession)
    session.fps = 12
    spec = session.realtime_spec()

    assert spec.first_chunk_frames == 33
    assert spec.steady_chunk_frames == 33
    assert spec.controls == ("prompt_update",)
    assert not ({"forward", "left", "right"} & set(spec.controls))


def test_prompt_update_resolves_only_at_segment_boundary() -> None:
    segments = [
        {"duration": 1.0, "prompt": "sunrise"},
        {"duration": 1.0, "prompt": "night rain"},
    ]
    assert realtime._prompt_from_segments(None, segments) == "night rain"
    assert realtime._prompt_from_segments("  snow  ", segments) == "snow"
    assert realtime._prompt_from_segments("", [{"keys": ["w"]}]) is None


def test_rgb_normalization_preserves_native_segment() -> None:
    source = np.linspace(0.0, 1.0, 33 * 4 * 6 * 3, dtype=np.float32).reshape(1, 33, 4, 6, 3)
    frames = realtime._as_rgb_frames(source)

    assert frames.shape == (33, 4, 6, 3)
    assert frames.dtype == np.uint8
    assert frames.flags.c_contiguous
    with pytest.raises(RuntimeError, match="native"):
        realtime._as_rgb_frames(source[:, :-1])


def test_hot_path_is_in_memory_and_keeps_resident_state() -> None:
    source = inspect.getsource(realtime.HeliosRealtimeSession.generate_next)
    forbidden = ("subprocess", "export_to_video", "empty_cache", ".cpu()", "mp4")

    assert all(token not in source for token in forbidden)
    assert "state.history_latents" in source
    assert "state.generator" in source
    assert "state.image_latents" in source


class _FakeRuntime:
    def runtime_plan(self, **_: object) -> dict[str, object]:
        return {"checkpoint_path": "/models/Helios-Distilled", "missing": []}


class _FakeSession:
    def __init__(self, checkpoint: str) -> None:
        self.checkpoint = checkpoint
        self.configured: dict[str, object] | None = None
        self.generated: dict[str, object] | None = None
        self.reset_count = 0

    def realtime_spec(self):
        return realtime.RealtimeSpec(
            fps=12,
            first_chunk_frames=33,
            steady_chunk_frames=33,
            controls=("prompt_update",),
        )

    def runtime_info(self) -> dict[str, object]:
        return {"resident": True, "context_parallel": False}

    def configure(self, image, **kwargs):
        self.configured = {"image": image, **kwargs}
        return {"status": "configured", "realtime_spec": self.realtime_spec().to_payload()}

    def generate_next(self, **kwargs):
        self.generated = dict(kwargs)
        return {"frames": np.zeros((33, 2, 2, 3), dtype=np.uint8)}

    def reset(self) -> None:
        self.reset_count += 1


def test_worldfoundry_pipeline_uses_native_resident_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(realtime, "HeliosRealtimeSession", _FakeSession)
    pipeline = HeliosPipeline(runtime=_FakeRuntime())

    prepared = pipeline.prepare_realtime()
    assert prepared["realtime_spec"]["steady_chunk_frames"] == 33
    configured = pipeline.configure_realtime(prompt="a user prompt")
    assert configured["status"] == "configured"
    streamed = pipeline.stream_realtime(
        prompt="a second prompt",
        realtime_segments=[{"prompt": "a third prompt"}],
    )
    assert streamed["frames"].shape[0] == 33
    assert pipeline._realtime_session.generated == {
        "prompt": "a second prompt",
        "prompt_segments": [{"prompt": "a third prompt"}],
    }
    with pytest.raises(ValueError, match="does not implement keyboard"):
        pipeline.stream_realtime(interactions=["forward"])
    pipeline.reset_realtime()
    assert pipeline._realtime_session.reset_count == 1


def test_torchrun_bridge_configures_prompt_only_helios_without_an_image(tmp_path: Path) -> None:
    class PromptOnlyPipeline:
        def __init__(self) -> None:
            self.configured = None

        def prepare_realtime(self):
            return {"prepared": True}

        def configure_realtime(self, **kwargs):
            self.configured = kwargs
            return {"configured": True}

        def stream_realtime(self, **_kwargs):
            return {"frames": []}

    entry = find_entry("helios")
    pipeline = PromptOnlyPipeline()
    context = PipelineContext(
        entry=entry,
        pipeline=pipeline,
        cache_key="helios-prompt-only",
        backend="from_pretrained",
        model_ref="",
        endpoint="",
        load_kwargs={},
        device="cuda",
    )
    request = SimpleNamespace(
        image=None,
        image_path=None,
        input_path="",
        prompt="a user-authored world prompt",
        call_kwargs={},
        fps=12,
        interactions=[],
    )

    handled, result = StudioManager(
        workspace_root=str(tmp_path / "studio")
    )._run_native_realtime_pipeline_action(
        driver=BaseRuntimeDriver(),
        ctx=context,
        request=request,  # type: ignore[arg-type]
        action="configure",
    )

    assert handled is True
    assert result == {"configured": True}
    assert pipeline.configured["images"] is None
    assert pipeline.configured["prompt"] == "a user-authored world prompt"
    assert context.state["studio_stream_initialized"] is True


def test_catalog_has_quality_demo_prompt_and_no_fixture_path() -> None:
    entry = find_entry("helios")

    assert "cat and a dog baking a cake" in entry.default_prompt
    assert not entry.default_input_path
    assert entry.supports_stream is True
    assert "prompt" in entry.stream_params


def test_user_can_select_true_context_parallel_gpu_count(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_kwargs = {"call_kwargs_text": json.dumps({"nproc_per_node": 8})}
    assert _requested_torchrun_nproc("helios", run_kwargs) == 8

    monkeypatch.setenv("WORLD_SIZE", "8")
    assert launch_uses_lingbot_torchrun_rollout(
        StudioLaunchConfig(model_id="helios", frontend="world")
    )
    monkeypatch.setenv(TORCHRUN_LINGBOT_FAST_ENV, "1")
    manager = StudioManager(workspace_root=str(tmp_path / "studio"))
    assert manager._should_use_torchrun_lingbot_fast(
        SimpleNamespace(model_id="helios"),
        SimpleNamespace(backend="from_pretrained"),
    )


def test_context_parallel_auto_uses_uneven_sharding_only_when_needed() -> None:
    lengths = realtime._context_parallel_sequence_lengths(
        height=384,
        width=640,
        vae_scale_factor=8,
        patch_size=(1, 2, 2),
    )

    assert lengths == (540, 2940, 2160, 4560, 8640, 11040)
    assert realtime._resolve_context_parallel_backend(
        "auto",
        4,
        attention_heads=40,
        sequence_lengths=lengths,
    ) == "ulysses"
    assert realtime._resolve_context_parallel_backend(
        "auto",
        8,
        attention_heads=40,
        sequence_lengths=lengths,
    ) == "ulysses_anything"


def test_context_parallel_rejects_incompatible_equipartition_before_generation() -> None:
    lengths = realtime._context_parallel_sequence_lengths(
        height=384,
        width=640,
        vae_scale_factor=8,
        patch_size=(1, 2, 2),
    )

    realtime._validate_context_parallel_shape(
        "ulysses",
        4,
        attention_heads=40,
        sequence_lengths=lengths,
        height=384,
        width=640,
    )
    with pytest.raises(ValueError, match="ulysses_anything"):
        realtime._validate_context_parallel_shape(
            "ulysses",
            8,
            attention_heads=40,
            sequence_lengths=lengths,
            height=384,
            width=640,
        )


def test_uneven_context_parallel_prepares_cpu_and_cuda_collectives() -> None:
    assert realtime._distributed_backend("auto") == "cpu:gloo,cuda:nccl"
    assert realtime._distributed_backend("ulysses_anything") == "cpu:gloo,cuda:nccl"
    assert realtime._distributed_backend("ulysses") == "nccl"


@pytest.mark.parametrize(
    ("world_size", "expected_main_sizes", "expected_total_sizes"),
    [
        (4, [135, 135, 135, 135], [735, 735, 735, 735]),
        (8, [68, 68, 68, 68, 67, 67, 67, 67], [368, 368, 368, 368, 367, 367, 367, 367]),
    ],
)
def test_persistent_context_shards_preserve_history_and_main_regions(
    world_size: int,
    expected_main_sizes: list[int],
    expected_total_sizes: list[int],
) -> None:
    history = torch.arange(2400)
    main = torch.arange(10_000, 10_540)
    source = torch.cat((history, main)).view(1, -1, 1)
    history_shards = []
    main_shards = []
    total_sizes = []
    main_sizes = []

    for rank in range(world_size):
        shard, local_main_length = helios_transformer._shard_history_and_main(
            source,
            global_main_length=540,
            rank=rank,
            world_size=world_size,
        )
        total_sizes.append(shard.shape[1])
        main_sizes.append(local_main_length)
        history_shards.append(shard[:, :-local_main_length])
        main_shards.append(shard[:, -local_main_length:])

    assert main_sizes == expected_main_sizes
    assert total_sizes == expected_total_sizes
    assert torch.equal(torch.cat(history_shards, dim=1), history.view(1, -1, 1))
    assert torch.equal(torch.cat(main_shards, dim=1), main.view(1, -1, 1))
    assert helios_transformer.HeliosTransformer3DModel._cp_plan == {}


def test_prompt_scheduled_ui_does_not_claim_keyboard_controls() -> None:
    entry = find_entry("helios")
    html = world_frontend_html(
        entry,
        StudioLaunchConfig(model_id="helios", frontend="world"),
    )

    assert "EDIT PROMPT + PRESS ENTER BETWEEN SEGMENTS" in html
    assert "OPTIONAL IMAGE" in html
    assert 'id="frameCanvas"' in html
    assert "stick-well is-hidden" in html
    assert "HOLD WASD / DRAG STICKS" not in html
    assert 'type: "prompt_update", prompt' in WORLD_REALTIME_CLIENT_JS
    assert "allowedKeys.clear()" in WORLD_REALTIME_CLIENT_JS
    assert "createImageBitmap" in WORLD_REALTIME_CLIENT_JS
    assert "__worldfoundryPresentedFrames" in WORLD_REALTIME_CLIENT_JS
    assert 'el.viewport.scrollIntoView({ block: "center"' in WORLD_REALTIME_CLIENT_JS


def test_prompt_scheduled_ui_bypasses_ice_even_for_remote_clients() -> None:
    assert _prefer_websocket_transport(
        prompt_scheduled=True,
        queued_segments=False,
        remote="203.0.113.7",
    )
    assert not _prefer_websocket_transport(
        prompt_scheduled=False,
        queued_segments=False,
        remote="203.0.113.7",
    )


def test_prompt_update_message_is_queued_for_next_native_boundary() -> None:
    async def exercise() -> None:
        active = _ActivePeer(
            peer=None,
            channel=None,
            frames=LatestFrameBuffer(maxsize=1),
            resampler=RealtimeControlResampler(fps=12),
            prompt_scheduled=True,
        )
        peers = RealtimePeerManager(
            runtime=None,  # type: ignore[arg-type]
            fps=12,
            chunk_frames=33,
        )
        await peers._handle_message(
            active,
            json.dumps({"type": "prompt_update", "prompt": "  a night storm  "}),
        )

        assert active.pending_prompt == "a night storm"
        assert active.first_action.is_set()
        assert len(active.action_arrivals) == 1

    asyncio.run(exercise())


def test_websocket_prompt_update_is_queued_for_next_native_boundary() -> None:
    class Socket:
        closed = False

        async def send_str(self, _payload: str) -> None:
            return None

    async def exercise() -> None:
        active = _ActiveSocket(
            socket=Socket(),
            resampler=RealtimeControlResampler(fps=12),
            frame_packets=asyncio.Queue(maxsize=1),
            prompt_scheduled=True,
        )
        peers = RealtimePeerManager(
            runtime=None,  # type: ignore[arg-type]
            fps=12,
            chunk_frames=33,
        )
        await peers._handle_socket_message(
            active,
            json.dumps({"type": "prompt_update", "prompt": "  a night storm  "}),
        )

        assert active.pending_prompt == "a night storm"
        assert active.first_action.is_set()
        assert len(active.action_arrivals) == 1

    asyncio.run(exercise())


class _FakeResidentCore:
    def __init__(self) -> None:
        self.transformer = SimpleNamespace(
            config=SimpleNamespace(in_channels=1),
            dtype=torch.float32,
        )
        self.vae = SimpleNamespace(dtype=torch.float32, decode=self._decode)
        self.video_processor = SimpleNamespace(postprocess_video=self._postprocess)
        self.stage_calls: list[bool] = []
        self._interrupt = False
        self._current_timestep = None

    def encode_prompt(self, **_kwargs):
        return torch.ones((1, 2, 1), dtype=torch.float32), None

    def prepare_latents(self, *_args, **_kwargs):
        value = float(len(self.stage_calls) + 1)
        return torch.full((1, 1, 9, 1, 1), value, dtype=torch.float32)

    def stage2_sample(self, *, latents, is_amplify_first_chunk, **_kwargs):
        self.stage_calls.append(bool(is_amplify_first_chunk))
        return latents

    def progress_bar(self, **_kwargs):
        return nullcontext(SimpleNamespace(update=lambda: None))

    @staticmethod
    def _decode(latents, **_kwargs):
        value = float(latents.mean()) / 10.0
        return (torch.full((1, 3, 33, 2, 2), value, dtype=torch.float32),)

    @staticmethod
    def _postprocess(video, **_kwargs):
        return video.permute(0, 2, 3, 4, 1).numpy()


def test_generate_next_retains_native_ar_state_across_calls() -> None:
    session = object.__new__(realtime.HeliosRealtimeSession)
    session.pipe = _FakeResidentCore()
    session.device = torch.device("cpu")
    session.checkpoint = Path("/models/Helios-Distilled")
    session.rank = 0
    session.world_size = 1
    session.cp_backend = "ulysses"
    session.fps = 12
    session._prompt_cache = {}
    generator = torch.Generator(device="cpu").manual_seed(42)
    session._state = realtime._AutoregressiveState(
        generator=generator,
        prompt="first prompt",
        prompt_embeds=torch.ones((1, 2, 1)),
        negative_prompt_embeds=None,
        history_latents=torch.zeros((1, 1, 19, 1, 1)),
        image_latents=None,
        latents_mean=torch.zeros((1, 1, 1, 1, 1)),
        latents_std=torch.ones((1, 1, 1, 1, 1)),
        indices_hidden_states=torch.arange(9).unsqueeze(0),
        indices_history_short=torch.arange(2).unsqueeze(0),
        indices_history_mid=torch.arange(2).unsqueeze(0),
        indices_history_long=torch.arange(16).unsqueeze(0),
        height=64,
        width=64,
    )

    first = session.generate_next()
    second = session.generate_next(prompt="second prompt")

    assert first["frames"].shape == (33, 2, 2, 3)
    assert second["frames"].shape == (33, 2, 2, 3)
    assert session.pipe.stage_calls == [True, False]
    assert session._state.chunk_index == 2
    assert session._state.history_latents.shape[2] == 19
    assert session._state.image_latents is not None
    assert session._state.prompt == "second prompt"
    assert session._state.generator is generator
