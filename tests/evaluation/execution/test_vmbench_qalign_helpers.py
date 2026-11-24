# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""QAlign configuration warnings and positional resizing need no checkpoints."""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3] / (
    "worldfoundry/evals/tasks/execution/runners/vmbench/runtime/official/q_align/model"
)


def _load_module(monkeypatch: pytest.MonkeyPatch, name: str) -> ModuleType:
    source = _ROOT / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_vmbench_qalign_{name}", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    monkeypatch.setitem(sys.modules, "icecream", SimpleNamespace(ic=Mock()))
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "name,parent_key",
    [("MplugOwlVisionConfig", "vision_config"), ("MplugOwlVisualAbstractorConfig", "abstractor_config")],
)
@pytest.mark.parametrize("config_kind", ["matching", "mismatched", "parent"])
def test_child_configs_preserve_loading_and_report_model_type_mismatches(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    name: str,
    parent_key: str,
    config_kind: str,
) -> None:
    module = _load_module(monkeypatch, "configuration_mplug_owl2")
    config_class = getattr(module, name)
    if config_kind == "parent":
        config_dict = {"model_type": "mplug-owl", parent_key: {"hidden_size": 24}}
    else:
        model_type = "another-model" if config_kind == "mismatched" else config_class.model_type
        config_dict = {"model_type": model_type, "hidden_size": 24}
    get_config_dict = Mock(return_value=(config_dict, {"output_attentions": True}))
    monkeypatch.setattr(config_class, "get_config_dict", get_config_dict)
    caplog.set_level(logging.WARNING, logger=module.__name__)

    config = config_class.from_pretrained("local-config", local_files_only=True)

    assert type(config) is config_class
    assert config.hidden_size == 24
    assert config.output_attentions is True
    get_config_dict.assert_called_once_with("local-config", local_files_only=True)
    messages = [record.getMessage() for record in caplog.records if record.name == module.__name__]
    if config_kind == "mismatched":
        assert messages == [
            f"You are using a model of type another-model to instantiate a model of type {config_class.model_type}. "
            "This is not supported for all configurations of models and can yield errors."
        ]
    else:
        assert messages == []


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize("target_size", [1, 4, 9])
def test_absolute_positions_preserve_bicubic_resize_and_dtype(
    monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype, target_size: int
) -> None:
    module = _load_module(monkeypatch, "visual_encoder")
    positions = torch.linspace(-3.0, 4.0, 12, dtype=dtype).reshape(4, 3)
    original = positions.clone()

    result = module.get_abs_pos(positions, target_size)

    if target_size == 4:
        assert result is positions
    else:
        side = 1 if target_size == 1 else 3
        expected = (
            F.interpolate(
                positions.float().reshape(1, 2, 2, 3).permute(0, 3, 1, 2),
                size=(side, side),
                mode="bicubic",
                align_corners=False,
            )
            .permute(0, 2, 3, 1)
            .flatten(0, 2)
            .to(dtype=dtype)
        )
        assert torch.equal(result, expected)
    assert result.shape == (target_size, 3)
    assert result.dtype == dtype
    assert result.device == positions.device
    assert torch.equal(positions, original)
