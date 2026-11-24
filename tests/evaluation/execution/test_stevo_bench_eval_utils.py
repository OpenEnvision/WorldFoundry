# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""STEVO-Bench evaluation helpers import and resolve generated-video links."""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.unit

_SOURCE = Path(__file__).resolve().parents[3] / (
    "worldfoundry/evals/tasks/execution/runners/stevo_bench/runtime/stevo_bench/eval/utils.py"
)


def _load(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    spec = importlib.util.spec_from_file_location("test_stevo_bench_eval_utils_module", _SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_local_video_source_resolution_does_not_require_http(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "requests", None)
    module = _load(monkeypatch)
    assert module._resolve_video_source("video.mp4", tmp_path / "outputs.json") == (
        "file",
        (tmp_path / "video.mp4").resolve(),
    )


def test_video_source_resolver_declares_a_kind_and_path(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load(monkeypatch)

    assert inspect.signature(module._resolve_video_source).return_annotation == tuple[str, Path]


@pytest.mark.parametrize(
    ("link", "kind", "relative"),
    [
        ("https://example.test/videos/a.mp4", "url", None),
        ("  http://example.test/b.mp4\n", "url", None),
        ("videos/c.mp4", "file", "videos/c.mp4"),
        ("  ../d.mp4 ", "file", "../d.mp4"),
    ],
)
def test_links_resolve_as_urls_or_paths_next_to_the_output_map(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, link: str, kind: str, relative: str | None
) -> None:
    module = _load(monkeypatch)
    outputs = tmp_path / "run" / "outputs.json"

    result = module._resolve_video_source(link, outputs)

    if relative is None:
        assert result == (kind, Path(link.strip()))
    else:
        assert result == (kind, (outputs.parent / relative).resolve())


def test_absolute_links_are_normalized(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = _load(monkeypatch)
    link = tmp_path / "videos" / ".." / "e.mp4"

    assert module._resolve_video_source(str(link), tmp_path / "outputs.json") == (
        "file",
        (tmp_path / "e.mp4").resolve(),
    )
