# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from worldfoundry.pipelines.lingbot_video.pipeline_lingbot_video import LingBotVideoGenerationPipeline
from worldfoundry.synthesis.visual_generation.lingbot_video.runtime import (
    INFERENCE_ROOT,
    RUNNER_MODULE,
    LingBotVideoRuntime,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("mode", ["t2i", "t2v", "ti2v"])
@pytest.mark.parametrize("nproc", [1, 4])
def test_lingbot_video_launches_the_packaged_runner(tmp_path: Path, mode: str, nproc: int) -> None:
    runtime = LingBotVideoRuntime(tmp_path / "checkpoint", device="cuda:2,3,4,5")
    request = {
        "prompt": "a robot on a workbench",
        "mode": mode,
        "image": "input.png",
        "num_inference_steps": 20,
        "nproc_per_node": nproc,
        "cfg_parallel_degree": 2 if nproc > 1 else 1,
        "context_parallel_degree": 2 if nproc > 1 else 1,
        "transformer_dtype": "bf16",
        "vae_dtype": "fp32",
        "run_refiner": True,
        "refiner_steps": 8,
        "enable_fsdp_inference": nproc > 1,
    }

    plan = runtime.build_plan(request=request, output_path=tmp_path / "output.mp4")

    prefix = [str(Path(sys.executable).resolve())]
    if nproc > 1:
        prefix += ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node", str(nproc)]
    prefix += ["-m", RUNNER_MODULE]
    assert list(plan.command[: len(prefix)]) == prefix
    assert plan.command[plan.command.index("--steps") + 1] == "20"
    assert plan.command[plan.command.index("--mode") + 1] == mode
    assert plan.command[plan.command.index("--vae_dtype") + 1] == "fp32"
    assert plan.command[plan.command.index("--transformer_dtype") + 1] == "bf16"
    assert plan.command[plan.command.index("--refiner_steps") + 1] == "8"
    assert "--run_refiner" in plan.command
    assert ("--enable_fsdp_inference" in plan.command) is (nproc > 1)
    assert plan.env == {"CUDA_VISIBLE_DEVICES": "2,3,4,5"}
    assert plan.refiner_output_path == str(tmp_path / "output_refined.mp4")


def test_lingbot_video_preflight_uses_the_installed_family(tmp_path: Path) -> None:
    pipeline = LingBotVideoGenerationPipeline.from_pretrained(tmp_path / "checkpoint")

    status = pipeline.preflight()

    assert pipeline.runtime.inference_root == INFERENCE_ROOT
    assert status["inference_root"] == str(INFERENCE_ROOT)
    assert status["runtime_ready"] is True
    assert status["missing_runtime_files"] == []
    assert status["checkpoint_ready"] is False
    assert status["missing_checkpoint_files"]
