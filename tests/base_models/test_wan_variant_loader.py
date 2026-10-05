from __future__ import annotations

import json

import torch

from worldfoundry.base_models.diffusion_model.loaders import wan_variant


class _TinyWanVariant(torch.nn.Module):
    load_assign_values: list[bool] = []

    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(3, 2)

    @classmethod
    def from_config(cls, _config, **_overrides):
        return cls()

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self.load_assign_values.append(bool(assign))
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


def _checkpoint_state() -> dict[str, torch.Tensor]:
    return {
        "projection.weight": torch.arange(6, dtype=torch.bfloat16).reshape(2, 3),
        "projection.bias": torch.tensor([1.0, -1.0], dtype=torch.bfloat16),
    }


def test_low_cpu_mem_wan_variant_assigns_checkpoint_storage(tmp_path, monkeypatch) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"model": "tiny"}), encoding="utf-8")
    checkpoint = _checkpoint_state()
    monkeypatch.setattr(wan_variant, "load_state_dict", lambda *_args, **_kwargs: checkpoint)
    _TinyWanVariant.load_assign_values.clear()

    model = wan_variant.load_wan_transformer(
        _TinyWanVariant,
        tmp_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    )

    assert _TinyWanVariant.load_assign_values == [True]
    assert model.projection.weight.dtype is torch.bfloat16
    assert model.projection.weight.data_ptr() == checkpoint["projection.weight"].data_ptr()
    torch.testing.assert_close(model.projection.weight, checkpoint["projection.weight"])


def test_regular_wan_variant_loading_preserves_copy_semantics(tmp_path, monkeypatch) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"model": "tiny"}), encoding="utf-8")
    checkpoint = _checkpoint_state()
    monkeypatch.setattr(wan_variant, "load_state_dict", lambda *_args, **_kwargs: checkpoint)
    _TinyWanVariant.load_assign_values.clear()

    model = wan_variant.load_wan_transformer(
        _TinyWanVariant,
        tmp_path,
        low_cpu_mem_usage=False,
        torch_dtype=torch.bfloat16,
    )

    assert _TinyWanVariant.load_assign_values == [False]
    assert model.projection.weight.dtype is torch.bfloat16
    assert model.projection.weight.data_ptr() != checkpoint["projection.weight"].data_ptr()
    torch.testing.assert_close(model.projection.weight, checkpoint["projection.weight"])
