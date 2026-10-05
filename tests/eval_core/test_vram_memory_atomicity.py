from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_move_preflights_capacity_before_touching_model(monkeypatch: pytest.MonkeyPatch) -> None:
    from worldfoundry.core.vram import memory

    model = torch.nn.Linear(4, 4)
    monkeypatch.setattr(memory, "get_cuda_free_memory_gb", lambda device=None: 1.0)

    with pytest.raises(RuntimeError, match="Insufficient memory"):
        memory.move_model_to_device_with_memory_preservation(
            model,
            "cuda:0",
            preserved_memory_gb=1.0,
        )

    assert {parameter.device.type for parameter in model.parameters()} == {"cpu"}


def test_transfer_estimate_includes_parameter_gradients() -> None:
    from worldfoundry.core.vram import memory

    model = torch.nn.Linear(4, 4, bias=False)
    model.weight.grad = torch.ones_like(model.weight)
    expected = model.weight.numel() * model.weight.element_size() * 2

    assert memory._model_transfer_bytes(model, torch.device("cuda:0")) == expected


def test_move_rolls_back_after_transfer_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    from worldfoundry.core.vram import memory

    class FailingModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.calls: list[str] = []

        def to(self, *args, **kwargs):
            device = torch.device(kwargs.get("device", args[0] if args else "cpu"))
            self.calls.append(str(device))
            if device.type == "cuda":
                raise RuntimeError("synthetic transfer failure")
            return self

    model = FailingModel()
    monkeypatch.setattr(memory, "get_cuda_free_memory_gb", lambda device=None: 80.0)

    with pytest.raises(RuntimeError, match="synthetic transfer failure"):
        memory.move_model_to_device_with_memory_preservation(model, "cuda:0")

    assert model.calls == ["cuda:0", "cpu"]


def test_offload_moves_whole_model_once_instead_of_stopping_midway(monkeypatch: pytest.MonkeyPatch) -> None:
    from worldfoundry.core.vram import memory

    class RecordingModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.calls: list[str] = []

        def to(self, *args, **kwargs):
            device = torch.device(kwargs.get("device", args[0] if args else "cpu"))
            self.calls.append(str(device))
            return self

    model = RecordingModel()
    monkeypatch.setattr(memory, "get_cuda_free_memory_gb", lambda device=None: 0.0)
    observed_devices = iter((torch.device("cuda:0"), torch.device("cpu")))
    monkeypatch.setattr(memory, "_uniform_model_device", lambda selected: next(observed_devices))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    memory.offload_model_from_device_for_memory_preservation(
        model,
        "cuda:0",
        preserved_memory_gb=8.0,
    )

    assert model.calls == ["cpu"]


def test_mixed_device_model_is_rejected_before_migration() -> None:
    from worldfoundry.core.vram import memory

    class MixedModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.cpu_weight = torch.nn.Parameter(torch.ones(1))
            self.meta_weight = torch.nn.Parameter(torch.empty(1, device="meta"))

    with pytest.raises(RuntimeError, match="mixed-device"):
        memory._uniform_model_device(MixedModel())


def test_unload_complete_models_is_safe_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    from worldfoundry.core.vram import memory

    model = torch.nn.Linear(1, 1)
    memory.gpu_complete_modules[:] = [model]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def unexpected_empty_cache() -> None:
        raise AssertionError("CPU-only unload must not call torch.cuda.empty_cache")

    monkeypatch.setattr(torch.cuda, "empty_cache", unexpected_empty_cache)
    memory.unload_complete_models()

    assert memory.gpu_complete_modules == []
