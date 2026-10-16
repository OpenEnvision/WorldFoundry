# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Product defaults and per-request export reuse the native batch contract."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest
import torch

from worldfoundry.base_models.diffusion_model.contracts import DiffusionOutput, DiffusionRequest
from worldfoundry.pipelines import native_diffusion
from worldfoundry.pipelines.wan.pipeline_wan_2p1_t2v import Wan2p1T2VPipeline

pytestmark = pytest.mark.unit


class Native:
    model_id = "wan2.1-t2v-1.3b"

    def __init__(self) -> None:
        self.calls: list[tuple[DiffusionRequest, ...]] = []
        self.fail = False

    def __call__(self, request: DiffusionRequest) -> DiffusionOutput:
        self.calls.append((request,))
        return self.output(request)

    @staticmethod
    def output(request: DiffusionRequest) -> DiffusionOutput:
        value = torch.full((1, 3, request.num_frames, 4, 4), float(request.sampling.seed))
        return DiffusionOutput(sample=value, latents=value.clone(), metadata={"seed": request.sampling.seed})

    def run_batch(self, requests: Iterable[DiffusionRequest]) -> tuple[DiffusionOutput, ...]:
        values = tuple(requests)
        self.calls.append(values)
        if self.fail:
            raise RuntimeError("native execution failed")
        return tuple(self.output(value) for value in values)


def test_product_batch_uses_singleton_defaults_aliases_and_separate_exports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    native: Any = Native()
    pipeline = Wan2p1T2VPipeline(native_pipeline=native, device="cpu")
    saved = []

    def save(sample: torch.Tensor, path: Path, **kwargs: object) -> str:
        saved.append((sample.clone(), path, kwargs))
        return str(path)

    monkeypatch.setattr(native_diffusion, "save_image_or_video_tensor", save)
    calls: list[dict[str, Any]] = [
        dict(prompt=prompt, seed=seed, infer_steps=2, output_dir=tmp_path / str(seed), return_dict=True)
        for prompt, seed in (("boat", 11), ("dog", 29))
    ]
    singles = [pipeline(**call) for call in calls]
    outputs = pipeline.batch_call(calls)
    assert native.calls[-1] == (native.calls[0][0], native.calls[1][0])
    assert len(saved) == 4 and saved[2][1] != saved[3][1]
    for single, output in zip(singles, outputs):
        assert torch.equal(single["sample"], output["sample"])
        assert single["metadata"] == output["metadata"]
        assert single["artifact_path"] == output["artifact_path"]
    assert saved[2][2] == saved[0][2]


def test_duplicate_paths_fail_before_native_execution(tmp_path: Path) -> None:
    native: Any = Native()
    pipeline = Wan2p1T2VPipeline(native_pipeline=native, device="cpu")
    with pytest.raises(ValueError, match="distinct output paths"):
        pipeline.batch_call([{"output_dir": tmp_path}, {"output_dir": tmp_path}])
    with pytest.raises(ValueError, match="one or more"):
        pipeline.batch_call([])
    assert not native.calls


@pytest.mark.parametrize("frames,suffix", [(1, ".jpg"), (2, ".mp4")])
def test_default_extension_collisions_fail_before_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frames: int,
    suffix: str,
) -> None:
    native: Any = Native()
    pipeline = Wan2p1T2VPipeline(native_pipeline=native, device="cpu")
    monkeypatch.setattr(
        native_diffusion, "save_image_or_video_tensor", lambda *_args, **_kwargs: pytest.fail("must not overwrite")
    )
    with pytest.raises(ValueError, match="distinct output paths"):
        pipeline.batch_call(
            [
                {"num_frames": frames, "output_path": tmp_path / "output"},
                {"num_frames": frames, "output_path": tmp_path / f"output{suffix}"},
            ]
        )


def test_file_uri_and_local_path_cannot_share_an_export(tmp_path: Path) -> None:
    native: Any = Native()
    pipeline = Wan2p1T2VPipeline(native_pipeline=native, device="cpu")
    path = tmp_path / "output.mp4"
    with pytest.raises(ValueError, match="distinct output paths"):
        pipeline.batch_call([{"output_path": path}, {"output_path": path.as_uri()}])
    assert not native.calls


def test_failed_native_batch_does_not_export_or_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    native: Any = Native()
    native.fail = True
    pipeline = Wan2p1T2VPipeline(native_pipeline=native, device="cpu")
    monkeypatch.setattr(
        native_diffusion, "save_image_or_video_tensor", lambda *_args, **_kwargs: pytest.fail("must not export")
    )
    with pytest.raises(RuntimeError, match="native execution failed"):
        pipeline.batch_call([{"prompt": "boat"}, {"prompt": "dog"}])
    assert len(native.calls) == 1 and len(native.calls[0]) == 2


def test_loader_normalizes_singletons_and_explicit_batch_warmup(monkeypatch: pytest.MonkeyPatch) -> None:
    native: Any = Native()
    warmed = []
    native.warmup = lambda requests: warmed.extend(requests) or ()
    monkeypatch.setattr(
        native_diffusion.NativeDiffusionPipeline, "from_pretrained", classmethod(lambda *_args, **_kwargs: native)
    )
    a = {"prompt": "boat", "height": 32, "width": 32, "num_frames": 5, "sampling": {"seed": 11}}
    b = {**a, "prompt": "dog", "sampling": {"seed": 29}}
    pipeline = Wan2p1T2VPipeline.from_pretrained(device="cpu", warmup_requests=[a, [a, b]])
    assert pipeline.native_pipeline is native
    assert isinstance(warmed[0], DiffusionRequest)
    assert [request.sampling.seed for request in warmed[1]] == [11, 29]


def test_empty_warmup_group_rejected_before_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        native_diffusion.NativeDiffusionPipeline,
        "from_pretrained",
        lambda *_args, **_kwargs: pytest.fail("must validate first"),
    )
    with pytest.raises(ValueError, match="nonempty"):
        Wan2p1T2VPipeline.from_pretrained(device="cpu", warmup_requests=[[]])
