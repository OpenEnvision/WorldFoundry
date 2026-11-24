# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""VBench dimensions report missing auxiliary metadata before decoding videos."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.unit

_METRICS = (
    "appearance_style",
    "color",
    "multiple_objects",
    "object_class",
    "scene",
    "spatial_relationship",
)


def _load_metric(monkeypatch: pytest.MonkeyPatch, name: str) -> ModuleType:
    utils = ModuleType("vbench.utils")
    for attribute in (
        "clip_transform",
        "clip_transform_Image",
        "load_dimension_info",
        "load_video",
        "read_frames_decord_by_fps",
        "tag2text_transform",
    ):
        setattr(utils, attribute, Mock())
    monkeypatch.setitem(sys.modules, "vbench.utils", utils)
    monkeypatch.setitem(
        sys.modules,
        "worldfoundry.base_models.perception_core.captioning.grit.model",
        SimpleNamespace(DenseCaptioning=Mock()),
    )
    monkeypatch.setitem(
        sys.modules,
        "worldfoundry.base_models.perception_core.captioning.tag2text.tag2text",
        SimpleNamespace(tag2text_caption=Mock()),
    )
    source = Path(__file__).resolve().parents[3] / (
        f"worldfoundry/evals/tasks/execution/runners/vbench/runtime/vbench/{name}.py"
    )
    spec = importlib.util.spec_from_file_location(f"test_vbench_metadata_{name}", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "get_rank", lambda: 0)
    return module


@pytest.mark.parametrize("name", _METRICS)
def test_missing_auxiliary_info_reports_metadata_error(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    module = _load_metric(monkeypatch, name)
    model = Mock()

    with pytest.raises(ValueError) as error:
        getattr(module, name)(model, [{"video_list": ["unopened.mp4"]}], "cpu")

    assert str(error.value) == "Auxiliary info is not in json, please check your json."
    module.load_video.assert_not_called()
    assert model.mock_calls == []
