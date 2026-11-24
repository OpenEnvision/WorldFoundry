# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from worldfoundry.synthesis.visual_generation.pusa_vidgen import adapter

pytestmark = pytest.mark.unit


def _runtime(tmp_path: Path) -> adapter.PusaVidGenRuntime:
    checkpoint = tmp_path / "pusa"
    checkpoint.mkdir()
    for filename in ("high_noise_pusa.safetensors", "low_noise_pusa.safetensors"):
        (checkpoint / filename).touch()
    return adapter.PusaVidGenRuntime(
        "pusa-vidgen",
        checkpoint,
        tmp_path / "wan",
        lightx2v_root=tmp_path / "lightx2v",
        device="cuda:3",
        python_executable="/env/python",
    )


@pytest.mark.parametrize("images", [None, ["/data/start.png", "/data/end.png"]])
@pytest.mark.parametrize("lightx2v", [False, True])
def test_v1_plan_preserves_conditioning_sampling_and_device(
    tmp_path: Path, images: list[str] | None, lightx2v: bool
) -> None:
    runtime = _runtime(tmp_path)
    request = runtime.prepare(
        "A forest",
        images,
        None,
        [0, 8],
        noise_multipliers=[0.1, 0.2],
        lightx2v=lightx2v,
        num_frames=9,
        num_steps=4,
        seed=7,
        fps=12,
        high_lora_alpha=1.2,
        low_lora_alpha=1.1,
        switch_DiT_boundary=0.7,
    )
    plan = runtime.build_plan(request=request, output_dir=tmp_path / "out")
    assert plan.command[:3] == (
        "/env/python",
        "-m",
        "worldfoundry.synthesis.visual_generation.pusa_vidgen.pusa_v1_runner",
    )
    assert plan.command[plan.command.index("--mode") + 1] == ("multi" if images else "t2v")
    expected = {
        "--num-frames": "9",
        "--num-inference-steps": "4",
        "--seed": "7",
        "--fps": "12",
        "--high-lora-alpha": "1.2",
        "--low-lora-alpha": "1.1",
        "--switch-dit-boundary": "0.7",
        "--cond-position": "0,8",
        "--noise-multipliers": "0.1,0.2",
        "--cfg-scale": "1.0" if lightx2v else "3.0",
    }
    for flag, value in expected.items():
        assert plan.command[plan.command.index(flag) + 1] == value
    assert ("--lightx2v" in plan.command) is lightx2v
    assert [plan.command[i + 1] for i, value in enumerate(plan.command) if value == "--image-path"] == (images or [])
    assert plan.env == {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "CUDA_VISIBLE_DEVICES": "3"}
    assert request["runtime"]["runner_version"] == "v1"


def test_v0_public_loader_is_rejected(tmp_path: Path) -> None:
    from worldfoundry.pipelines.pusa_vidgen.pipeline_pusa_vidgen import PusaVidGenPipeline

    with pytest.raises(ValueError, match="Pusa v0 / Mochi route is not available in this in-tree build; use v1"):
        PusaVidGenPipeline.from_pretrained(
            model_path={"checkpoint_root": tmp_path, "base_model_root": tmp_path, "runner_version": "v0"}, device="cpu"
        )


def test_v0_request_plan_is_rejected(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    with pytest.raises(ValueError, match="Pusa v0 / Mochi route is not available in this in-tree build; use v1"):
        runtime.build_plan(request={"runtime": {"runner_version": "v0"}}, output_dir=tmp_path / "out")


@pytest.mark.parametrize("filename", ["high_noise_pusa.safetensors", "low_noise_pusa.safetensors"])
def test_missing_v1_lora_does_not_select_another_backend(tmp_path: Path, filename: str) -> None:
    runtime = _runtime(tmp_path)
    (runtime.checkpoint_root / filename).unlink()
    request = runtime.prepare("A forest", None, None, None)
    with pytest.raises(FileNotFoundError, match="Pusa v1 requires high- and low-noise LoRA checkpoints") as exc:
        runtime.build_plan(request=request, output_dir=tmp_path / "out")
    assert filename in str(exc.value)


def test_run_plan_inherits_operator_environment_without_source_injection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _runtime(tmp_path)
    plan = runtime.build_plan(request=runtime.prepare("A forest", None, None, None), output_dir=tmp_path / "out")
    monkeypatch.setenv("PYTHONPATH", "/operator/dependencies")

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert command == list(plan.command)
        assert kwargs["env"]["PYTHONPATH"] == "/operator/dependencies"
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "3"
        assert kwargs["env"]["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
        assert kwargs["cwd"] == plan.workdir
        assert kwargs["timeout"] == 17
        Path(plan.output_path).write_bytes(b"mp4")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(adapter.subprocess, "run", run)
    result = runtime.run_plan(plan, timeout_seconds=17)
    assert result["ok"] is True
    assert result["generated_files"] == [plan.output_path]
