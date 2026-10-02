from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from worldfoundry.synthesis.visual_generation.versecrafter.versecrafter_runtime.versecrafter.pipeline import (
    WanVerseCrafterPipeline,
)


class _RecordingOffloadHook:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.is_onloaded = False

    def pre_forward(self, module) -> None:
        assert module._hf_hook is self
        self.events.append("pre_forward")
        self.is_onloaded = True


class _FakeVae:
    dtype = torch.float32

    def __init__(self, events: list[str], *, with_hook: bool = True) -> None:
        self.events = events
        if with_hook:
            self._hf_hook = _RecordingOffloadHook(events)

    def encode(self, value: torch.Tensor):
        assert self._hf_hook.is_onloaded
        self.events.append("encode")
        return SimpleNamespace(
            latent_dist=SimpleNamespace(mode=lambda: value + 1)
        )

    def decode(self, value: torch.Tensor):
        assert self._hf_hook.is_onloaded
        self.events.append("decode")
        return SimpleNamespace(sample=value)


class _RecordingUserCpuOffloadHook:
    def __init__(self, model: _FakeVae, events: list[str]) -> None:
        self.model = model
        self.events = events

    def offload(self) -> None:
        self.events.append("offload")
        self.model._hf_hook.is_onloaded = False


def test_vae_encode_onloads_component_before_non_forward_method() -> None:
    events: list[str] = []
    vae = _FakeVae(events)
    value = torch.zeros(1)

    result = WanVerseCrafterPipeline._vae_encode_mode(vae, value)

    torch.testing.assert_close(result, value + 1)
    assert events == ["pre_forward", "encode"]


def test_vae_decode_onloads_component_before_non_forward_method() -> None:
    events: list[str] = []
    vae = _FakeVae(events)
    pipeline = SimpleNamespace(
        vae=vae,
        _apply_vae_forward_hook=WanVerseCrafterPipeline._apply_vae_forward_hook,
    )
    latents = torch.zeros(1, 1, 1, 1, 1)

    result = WanVerseCrafterPipeline.decode_latents(pipeline, latents)

    np.testing.assert_array_equal(result, np.full(latents.shape, 0.5))
    assert events == ["pre_forward", "decode"]


def test_geoada_encode_releases_vae_before_transformer_phase() -> None:
    events: list[str] = []
    vae = _FakeVae(events)
    user_hook = _RecordingUserCpuOffloadHook(vae, events)
    pipeline = WanVerseCrafterPipeline.__new__(WanVerseCrafterPipeline)
    pipeline._all_hooks = [SimpleNamespace(model=object()), user_hook]
    frames = [torch.zeros(1, 1, 1, 1, 1)]

    result = pipeline.geoada_encode_multi_frames(frames, None, vae=vae)

    assert len(result) == 1
    assert events == ["pre_forward", "encode", "offload"]
    assert not vae._hf_hook.is_onloaded


def test_post_encode_release_ignores_sequential_leaf_hooks() -> None:
    events: list[str] = []
    vae = _FakeVae(events)
    pipeline = WanVerseCrafterPipeline.__new__(WanVerseCrafterPipeline)
    pipeline._all_hooks = []

    pipeline._offload_vae_after_encode(vae)

    assert events == []


def test_vae_forward_hook_is_optional() -> None:
    vae = SimpleNamespace()

    WanVerseCrafterPipeline._apply_vae_forward_hook(vae)
