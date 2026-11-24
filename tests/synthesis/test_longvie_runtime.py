# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from worldfoundry.synthesis.visual_generation.longvie import worldfoundry_runtime as runtime_module

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("variant", ["longvie-1", "longvie-2"])
def test_load_resolves_package_components_without_search_path_changes(
    monkeypatch: pytest.MonkeyPatch, variant: str
) -> None:
    def missing_weights(value: object) -> Path:
        raise FileNotFoundError("test Wan weights")

    monkeypatch.setattr(runtime_module, "resolve_wan21_i2v_dir", missing_weights)
    search_path = sys.path.copy()
    runtime = runtime_module.LongVieOfficialRuntime(
        control_weight_path=None,
        dit_weight_path=None,
        weight_dir=None,
        wan_base_dir=None,
        tokenizer_dir=None,
        device="cpu",
        variant=variant,
    )

    with pytest.raises(FileNotFoundError, match="test Wan weights"):
        runtime.load()

    assert sys.path == search_path


def test_video_to_frames_imports_decord_without_search_path_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    pixels = np.arange(2 * 3 * 4 * 3, dtype=np.uint8).reshape(2, 3, 4, 3)

    class VideoReader:
        def __init__(self, path: str) -> None:
            assert path == "control.mp4"

        def __getitem__(self, key: slice) -> SimpleNamespace:
            assert key == slice(None)
            return SimpleNamespace(asnumpy=lambda: pixels)

    decord = ModuleType("decord")
    monkeypatch.setattr(decord, "VideoReader", VideoReader, raising=False)
    monkeypatch.setitem(sys.modules, "decord", decord)
    search_path = sys.path.copy()

    frames = runtime_module.video_to_frames("control.mp4", target_size=(4, 3))

    np.testing.assert_array_equal(np.stack([np.asarray(frame) for frame in frames]), pixels)
    assert sys.path == search_path
