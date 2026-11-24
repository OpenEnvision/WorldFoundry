# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from worldfoundry.synthesis.visual_generation.bernini.worldfoundry_runtime import (
    MODEL_CONFIG_DEFAULTS,
    VARIANTS,
    BerniniRuntime,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("model_id", VARIANTS)
@pytest.mark.parametrize("workers", [2, 8])
def test_distributed_launch_preserves_payload_topology_and_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, model_id: str, workers: int
) -> None:
    captured: dict[str, Any] = {}

    def launch(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured.update(command=command, payload=json.loads(Path(command[-1]).read_text()), **kwargs)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", launch)
    monkeypatch.setenv("PYTHONPATH", "/operator/modules")
    monkeypatch.setenv("NCCL_DEBUG", "INFO")
    (tmp_path / "config.json").write_text("{}")
    runtime = BerniniRuntime(model_id=model_id, checkpoint_path=tmp_path, model_config={})
    runtime._generate_distributed(
        prompt="A camera orbit.",
        output_path=tmp_path / "clip.mp4",
        task_type="rv2v",
        image="first.png",
        images=["last.png"],
        video="input.mp4",
        nproc_per_node=workers,
        ulysses_size=workers,
        overrides={"seed": 7, "num_inference_steps": 4},
    )
    assert captured["command"][:-1] == [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(workers),
        "-m",
        "worldfoundry.synthesis.visual_generation.bernini.distributed_infer",
        "--payload",
    ]
    assert captured["payload"] == {
        "model_id": model_id,
        "checkpoint_path": str(tmp_path),
        "model_config": MODEL_CONFIG_DEFAULTS,
        "prompt": "A camera orbit.",
        "output_path": str(tmp_path / "clip.mp4"),
        "task_type": "rv2v",
        "image": "first.png",
        "images": ["last.png"],
        "video": "input.mp4",
        "ulysses_size": workers,
        "overrides": {"seed": 7, "num_inference_steps": 4},
    }
    assert captured["env"]["PYTHONPATH"] == "/operator/modules"
    assert captured["env"]["NCCL_DEBUG"] == "INFO"
    assert captured["text"] and captured["capture_output"]
    assert captured["check"] is False
    assert not Path(captured["command"][-1]).exists()


def test_distributed_failure_keeps_the_worker_diagnostic(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        subprocess, "run", lambda command, **kwargs: subprocess.CompletedProcess(command, 1, "", "worker failed")
    )
    runtime = BerniniRuntime(model_id="bernini", checkpoint_path=tmp_path)
    with pytest.raises(RuntimeError, match="Bernini distributed inference failed: worker failed"):
        runtime._generate_distributed(
            prompt="test",
            output_path=tmp_path / "clip.mp4",
            task_type="t2v",
            image=None,
            images=None,
            video=None,
            nproc_per_node=2,
            ulysses_size=2,
            overrides={},
        )


def test_renderer_load_forwards_default_and_custom_flow_shift(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from types import SimpleNamespace

    from worldfoundry.pipelines.bernini.pipeline_bernini import BerniniWorldFoundryPipeline

    captured: list[dict[str, Any]] = []

    class _FakeRendererPipeline:
        @classmethod
        def from_pretrained(cls, config_dir: str, **kwargs: Any) -> object:
            captured.append({"config_dir": config_dir, **kwargs})
            return object()

    monkeypatch.setattr(BerniniRuntime, "requirement_report", lambda _self: {"ready": True, "missing": []})
    monkeypatch.setitem(
        sys.modules,
        "worldfoundry.synthesis.visual_generation.bernini.inference.pipeline",
        SimpleNamespace(BerniniPipeline=_FakeRendererPipeline, BerniniRendererPipeline=_FakeRendererPipeline),
    )
    (tmp_path / "config.json").write_text("{}")

    default_runtime = BerniniRuntime(model_id="bernini-r-1.3b", checkpoint_path=tmp_path, device="cuda")
    default_runtime._load_pipeline()
    custom_pipe = BerniniWorldFoundryPipeline.from_pretrained(
        model_path=str(tmp_path), model_id="bernini-r-14b", device="cuda", flow_shift=7.5
    )
    custom_pipe.runtime._load_pipeline()

    assert captured[0]["shift"] == 5.0
    assert captured[0]["load_ckpt_weights"] is False
    assert captured[1]["shift"] == 7.5
    assert captured[1]["lazy_expert_loading"] is True
