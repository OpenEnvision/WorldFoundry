from __future__ import annotations

import torch

from worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v3.api import (
    DepthAnything3,
)
from worldfoundry.synthesis.visual_generation.alayaworld.spatial import AlayaSpatialMemory
from worldfoundry.synthesis.visual_generation.alayaworld.runtime import AlayaWorldRuntime
from worldfoundry.synthesis.visual_generation.alayaworld.ltx_compiling import (
    _SeqDynamicMarkingProcessor,
    _supported_config_options,
)


def test_alayaworld_loads_direct_da3_checkpoint_with_nested_giant_preset(
    tmp_path,
    monkeypatch,
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    calls = []

    class DummyModel:
        def to(self, device):
            calls.append(("to", device))
            return self

        def eval(self):
            calls.append(("eval",))
            return self

    def fake_from_pretrained(repo_id, **kwargs):
        calls.append(("from_pretrained", repo_id, kwargs))
        return DummyModel()

    monkeypatch.setattr(DepthAnything3, "from_pretrained", fake_from_pretrained)
    memory = AlayaSpatialMemory(
        video_encoder=None,
        video_decoder=None,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        height=32,
        width=32,
        da3_path=str(checkpoint),
    )

    assert memory._load_da3() is memory._da3
    assert calls[0] == (
        "from_pretrained",
        "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        {"model_name": "da3nested-giant-large", "weights_path": str(checkpoint)},
    )
    assert calls[1:] == [("to", torch.device("cpu")), ("eval",)]


def test_alayaworld_catalog_exposes_rollout_runtime_parameters() -> None:
    from worldfoundry.studio.inference.catalog import find_entry

    entry = find_entry("alayaworld")

    assert {
        "num_frames",
        "sampling_steps",
        "seed",
        "spatial_enabled",
        "depth_backend",
        "compile_mode",
    }.issubset(entry.call_params)


def test_alayaworld_prompt_encoder_stays_on_cpu() -> None:
    policy = AlayaWorldRuntime._prompt_runtime_policy(
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )

    assert policy.device == torch.device("cpu")


def test_alayaworld_can_offload_da3_without_discarding_it(monkeypatch) -> None:
    calls = []

    class DummyModel:
        device = torch.device("cuda")

        def to(self, device):
            calls.append(device)
            return self

    memory = AlayaSpatialMemory(
        video_encoder=None,
        video_decoder=None,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
        height=32,
        width=32,
        depth_backend="constant",
    )
    model = DummyModel()
    memory._da3 = model
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    memory.offload_depth_model()

    assert memory._da3 is model
    assert calls == ["cpu"]
    assert model.device == torch.device("cpu")


def test_da3_cpu_forward_keeps_fp32_when_cuda_supports_bfloat16(monkeypatch) -> None:
    seen = []

    class DummyNetwork(torch.nn.Module):
        def forward(self, image, *_args):
            seen.append(image.dtype)
            return {"depth": image[:, :, :1]}

    model = object.__new__(DepthAnything3)
    torch.nn.Module.__init__(model)
    model.model = DummyNetwork()
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    model.forward(torch.ones(1, 1, 3, 8, 8, dtype=torch.float32))

    assert seen == [torch.float32]


def test_alayaworld_filters_unsupported_torch_compile_options() -> None:
    class Config:
        _config = {"available": False}

    assert _supported_config_options(
        Config(), {"available": True, "newer_torch_only": True}
    ) == {"available": True}


def test_alayaworld_static_compile_does_not_mark_dynamic(monkeypatch) -> None:
    calls = []
    marker_calls = []

    def inner(*args):
        calls.append(args)
        return "processed"

    monkeypatch.setattr(torch._dynamo, "mark_dynamic", lambda *args: marker_calls.append(args))
    processor = _SeqDynamicMarkingProcessor(inner=inner, enabled=False)
    arguments = object()
    perturbations = object()
    self_attention_type = object()
    cross_attention_type = object()

    result = processor(
        arguments,
        perturbations,
        3,
        self_attention_type,
        cross_attention_type,
    )

    assert result == "processed"
    assert calls == [
        (arguments, perturbations, 3, self_attention_type, cross_attention_type)
    ]
    assert marker_calls == []
