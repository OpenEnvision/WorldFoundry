# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""WBench SAM2 batch tracking reports a completion summary in both device modes."""

from __future__ import annotations

import importlib.util
import logging
import multiprocessing
import os
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import Mock, call

import pytest

pytestmark = pytest.mark.unit

_SOURCE = Path(__file__).resolve().parents[3] / (
    "worldfoundry/evals/tasks/execution/runners/wbench/runtime/wbench/tools/run_sam2_track.py"
)
_TASKS = [
    ("1", "m1.png", "v1.mp4", None),
    ("2", "m2.png", "v2.mp4", "i2.png"),
    ("3", "m3.png", "v3.mp4", None),
]


def _load(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    spec = importlib.util.spec_from_file_location("test_wbench_run_sam2_track", _SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _run(monkeypatch: pytest.MonkeyPatch, module: ModuleType, output: Path, *options: str) -> Mock:
    collect = Mock(return_value=(list(_TASKS), 4))
    monkeypatch.setattr(module, "_collect_tasks", collect)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_sam2_track.py",
            "--video_dir",
            "videos",
            "--case_dir",
            "cases",
            "--mask_dir",
            "masks",
            "--output_base",
            str(output),
            *options,
        ],
    )
    module.main()
    return collect


def _messages(caplog: pytest.LogCaptureFixture, module: ModuleType) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.name == module.__name__]


def test_multi_gpu_batches_count_worker_exit_codes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    module = _load(monkeypatch)
    start_methods: list[tuple[str, bool]] = []
    processes: list[Any] = []

    class Process:
        def __init__(self, target: Callable[..., None], args: tuple[Any, ...]) -> None:
            self.target = target
            self.args = args
            self.started = False
            self.exitcode: int | None = None
            processes.append(self)

        def start(self) -> None:
            self.started = True

        def join(self) -> None:
            self.exitcode = 0 if self.args[1] == 0 else 3

    def set_start_method(method: str, force: bool = False) -> None:
        start_methods.append((method, force))

    monkeypatch.setattr(multiprocessing, "Process", Process)
    monkeypatch.setattr(multiprocessing, "set_start_method", set_start_method)
    caplog.set_level(logging.INFO, logger=module.__name__)

    collect = _run(monkeypatch, module, tmp_path, "--gpus", "4,7", "--fps", "2.5")

    collect.assert_called_once_with("cases", "masks", "videos", str(tmp_path), False)
    assert start_methods == [("spawn", True)]
    assert [process.target for process in processes] == [module._gpu_worker, module._gpu_worker]
    assert [process.args for process in processes] == [
        (4, 0, 2, [_TASKS[0], _TASKS[2]], str(tmp_path), 2.5),
        (7, 1, 2, [_TASKS[1]], str(tmp_path), 2.5),
    ]
    assert all(process.started for process in processes)
    assert _messages(caplog, module)[-2:] == ["All workers done.", "Done: 1, skipped: 4, failed: 1"]


def test_single_device_batches_count_tracking_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    module = _load(monkeypatch)
    predictor = object()
    build_predictor = Mock(return_value=predictor)

    def track(*args: Any, **kwargs: Any) -> None:
        if args[1] == "v2.mp4":
            raise RuntimeError("tracking failed")

    track_video = Mock(side_effect=track)
    monkeypatch.setattr(module, "build_predictor", build_predictor)
    monkeypatch.setattr(module, "track_video", track_video)
    caplog.set_level(logging.INFO, logger=module.__name__)

    _run(monkeypatch, module, tmp_path, "--device", "cpu")

    build_predictor.assert_called_once_with("cpu")
    assert track_video.call_args_list == [
        call(
            predictor,
            video,
            mask,
            os.path.join(str(tmp_path), f"case_{case}"),
            device="cpu",
            target_fps=5.0,
            initial_image_path=image,
        )
        for case, mask, video, image in _TASKS
    ]
    assert _messages(caplog, module)[-3:] == [
        "case_2: tracking failed",
        "Done: 2, Failed: 1",
        "Done: 2, skipped: 4, failed: 1",
    ]
