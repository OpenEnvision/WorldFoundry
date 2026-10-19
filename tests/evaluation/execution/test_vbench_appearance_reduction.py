# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""VBench appearance style has the same frame-weighted definition on every rank."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.unit


@pytest.fixture
def appearance(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    utils = ModuleType("vbench.utils")
    for name in (
        "load_video",
        "load_dimension_info",
        "clip_transform",
        "read_frames_decord_by_fps",
        "clip_transform_Image",
    ):
        setattr(utils, name, lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "vbench.utils", utils)
    source = Path(__file__).resolve().parents[3] / (
        "worldfoundry/evaluation/tasks/execution/runners/vbench/runtime/vbench/appearance_style.py"
    )
    spec = importlib.util.spec_from_file_location("test_vbench_appearance_runtime", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def logits(image: torch.Tensor, text: torch.Tensor) -> tuple[None, torch.Tensor]:
        return None, image.reshape(1, 1)

    monkeypatch.setattr(
        module,
        "clip",
        SimpleNamespace(
            load=lambda **kwargs: (logits, None),
            tokenize=lambda queries: torch.zeros(1),
        ),
    )
    monkeypatch.setattr(
        module, "clip_transform_Image", lambda size: lambda image: torch.tensor(float(image.getpixel((0, 0))[0]))
    )
    return module


@pytest.mark.parametrize("frames", [[[20, 80], [60, 40]], [[20, 80], [60]], [[20, 80]]])
def test_distributed_appearance_matches_single_process_with_empty_rank(
    appearance: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    frames: list[list[int]],
) -> None:
    infos = [
        {"video_list": [str(index)], "auxiliary_info": {"appearance_style": "style"}} for index in range(len(frames))
    ]
    monkeypatch.setattr(
        appearance,
        "load_video",
        lambda path, **kwargs: [np.full((2, 2, 3), value, dtype=np.uint8) for value in frames[int(path)]],
    )
    monkeypatch.setattr(appearance, "load_dimension_info", lambda *args, **kwargs: (None, infos))
    monkeypatch.setattr(appearance, "get_rank", lambda: 0)
    monkeypatch.setattr(appearance, "get_world_size", lambda: 1)
    monkeypatch.setattr(appearance, "distribute_list_to_rank", lambda rows: rows)
    single, global_rows = appearance.compute_appearance_style("unused.json", "cpu", {})
    expected = sum(sum(row) for row in frames) / (100 * sum(len(row) for row in frames))
    assert single == pytest.approx(expected)

    monkeypatch.setattr(appearance, "get_world_size", lambda: len(infos) + 1)
    monkeypatch.setattr(appearance, "gather_list_of_dict", lambda rows: global_rows)
    for rank in range(len(infos) + 1):
        monkeypatch.setattr(appearance, "get_rank", lambda rank=rank: rank)
        monkeypatch.setattr(appearance, "distribute_list_to_rank", lambda rows, rank=rank: rows[rank : rank + 1])
        distributed, details = appearance.compute_appearance_style("unused.json", "cpu", {})
        assert distributed == pytest.approx(single)
        assert details == global_rows
        assert sum(len(row["frame_results"]) for row in details) == sum(map(len, frames))


def test_empty_appearance_shard_has_no_fabricated_score(appearance: ModuleType) -> None:
    score, rows = appearance.appearance_style(None, [], "cpu")
    assert score is None
    assert rows == []
