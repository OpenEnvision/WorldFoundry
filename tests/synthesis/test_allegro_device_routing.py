from __future__ import annotations

import inspect
import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import torch

from worldfoundry.synthesis.visual_generation.allegro import worldfoundry_runtime


def test_allegro_initialization_uses_workspace_device(monkeypatch, tmp_path) -> None:
    moved_devices: list[torch.device] = []
    selected_devices: list[torch.device] = []

    class FakeModule:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

        def to(self, device):
            moved_devices.append(torch.device(device))
            return self

        def eval(self):
            return self

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

    class FakePipeline(FakeModule):
        def __init__(self, **kwargs):
            assert kwargs["device"] == torch.device("cuda:3")

    @contextmanager
    def fake_cuda_device(device):
        selected_devices.append(torch.device(device))
        yield

    components = SimpleNamespace(
        autoencoder_cls=FakeModule,
        transformer_cls=FakeModule,
        pipeline_cls=FakePipeline,
    )
    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoImageProcessor = object()
    fake_transformers.T5EncoderModel = FakeModule
    fake_transformers.T5Tokenizer = FakeTokenizer
    fake_diffusers = ModuleType("diffusers")
    fake_diffusers.__path__ = []
    fake_schedulers = ModuleType("diffusers.schedulers")
    fake_schedulers.EulerAncestralDiscreteScheduler = lambda: object()
    fake_diffusers.schedulers = fake_schedulers

    monkeypatch.setattr(worldfoundry_runtime, "load_allegro_components", lambda: components)
    monkeypatch.setattr(torch.cuda, "device", fake_cuda_device)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    monkeypatch.setitem(sys.modules, "diffusers.schedulers", fake_schedulers)

    runtime = worldfoundry_runtime.Allegro(
        model_name="allegro_ti2v",
        model_path=str(tmp_path),
        device="cuda:3",
    )

    assert runtime.device == torch.device("cuda:3")
    assert selected_devices == [torch.device("cuda:3")]
    assert moved_devices == [torch.device("cuda:3")] * 4


def test_allegro_generation_has_no_hard_coded_cuda_zero() -> None:
    source = inspect.getsource(worldfoundry_runtime.Allegro.generate_video)

    assert 'device="cuda:0"' not in source
    assert "device=self.device" in source
