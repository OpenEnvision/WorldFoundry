# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""FETV frame directories preserve preprocessing and report empty inputs."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image

pytestmark = pytest.mark.unit

_SOURCE = Path(__file__).resolve().parents[3] / (
    "worldfoundry/evals/tasks/execution/runners/fetv/runtime/fetv_eval/video_dataset.py"
)


def _dataset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    preprocessor: Callable[[Image.Image], torch.Tensor] | None = None,
) -> Any:
    spec = importlib.util.spec_from_file_location("test_fetv_video_dataset", _SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"video_id": "video", "prompt": "caption"}\n', encoding="utf-8")
    return module.VideoDataset(
        prompt_file=str(prompts),
        gen_path=str(tmp_path),
        media_size=4,
        max_num_frm=3,
        preprocessor=preprocessor,
    )


@pytest.mark.parametrize("entrypoint", ["direct", "item"])
def test_empty_frame_directory_reports_its_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    entrypoint: str,
) -> None:
    dataset = _dataset(monkeypatch, tmp_path)
    directory = tmp_path / "sent0_frames"
    directory.mkdir()

    with pytest.raises(ValueError) as error:
        if entrypoint == "direct":
            dataset.get_video_frames(str(directory))
        else:
            dataset[0]

    assert str(error.value) == f"No video frames found in {directory}"
    assert isinstance(error.value.__cause__, IndexError)
    assert capsys.readouterr().out == f"{directory}\n[]\n[]\n"


@pytest.mark.parametrize("frame_count", [1, 3, 7])
@pytest.mark.parametrize("custom_preprocessor", [False, True])
def test_frame_sorting_sampling_preprocessing_and_dtype_are_preserved(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    frame_count: int,
    custom_preprocessor: bool,
) -> None:
    calls: list[int] = []

    def preprocess(image: Image.Image) -> torch.Tensor:
        assert image.mode == "RGB"
        pixel = image.getpixel((0, 0))
        assert isinstance(pixel, tuple)
        calls.append(int(pixel[0]))
        return torch.full((3, 2, 2), float(pixel[0]), dtype=torch.float64)

    dataset = _dataset(monkeypatch, tmp_path, preprocess if custom_preprocessor else None)
    directory = tmp_path / "sent0_frames"
    directory.mkdir()
    identifiers = [10, 2, 1, 12, 3, 11, 0][:frame_count]
    for identifier in identifiers:
        Image.new("RGB", (6, 8), (identifier, identifier + 1, identifier + 2)).save(
            directory / f"frame_{identifier}.png"
        )

    result, caption, video_id = dataset[0]

    ordered = sorted(identifiers)
    selected = [ordered[index] for index in (0, 2, 4)] if frame_count == 7 else ordered
    if custom_preprocessor:
        expected = torch.stack([torch.full((3, 2, 2), float(value), dtype=torch.float64) for value in selected])
        assert calls == ordered
    else:
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711])
        colors = torch.tensor([[value, value + 1, value + 2] for value in selected], dtype=torch.float32) / 255
        expected = ((colors - mean) / std).reshape(-1, 3, 1, 1).expand(-1, -1, 4, 4)
        assert calls == []
    assert torch.equal(result, expected)
    assert result.dtype == expected.dtype
    assert result.device.type == "cpu"
    assert (caption, video_id) == ("caption", "video")
    assert dataset.video_shape == result.shape


def test_absent_frame_path_retains_its_sentinel(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assert _dataset(monkeypatch, tmp_path).get_video_frames(None) == "none"
