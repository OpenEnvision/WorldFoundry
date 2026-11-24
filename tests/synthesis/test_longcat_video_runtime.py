# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Weight-free LongCat-Video package-launch and I2V scheduler contracts."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import pytest

from worldfoundry.synthesis.visual_generation.longcat_video import worldfoundry_runtime as runtime

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("task_type", "entrypoint"),
    [
        ("t2v", "run_inference_text_to_video"),
        ("i2v", "run_inference_image_to_video"),
        ("vc", "run_inference_video_continuation"),
        ("refine", "run_inference_long_video"),
    ],
)
@pytest.mark.parametrize("context_parallel_size", [1, 2])
@pytest.mark.parametrize("enable_compile", [False, True])
def test_longcat_build_plan_launches_installed_task_module(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task_type: str,
    entrypoint: str,
    context_parallel_size: int,
    enable_compile: bool,
) -> None:
    monkeypatch.setenv("PYTHONPATH", "operator-owned-path")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,5")
    model = runtime.LongCatVideoRuntime(
        checkpoint_dir=tmp_path / "checkpoint", python_executable="/env/bin/python", device="cuda"
    )
    output = tmp_path / "output" / "video.mp4"
    request = {"prompt": "camera move", "seed": 71, "cpu_offload": True}

    plan = model.build_plan(
        request=request,
        task_type=task_type,
        output_dir=output.parent,
        output_path=output,
        context_parallel_size=context_parallel_size,
        enable_compile=enable_compile,
    )

    module = f"worldfoundry.synthesis.visual_generation.longcat_video.longcat_video_runtime.{entrypoint}"
    prefix = (
        ["/env/bin/python", "-m", "torch.distributed.run", "--standalone", "--nnodes", "1", "--nproc_per_node", "2"]
        if context_parallel_size > 1
        else ["/env/bin/python"]
    )
    command = [
        *prefix,
        "-m",
        module,
        "--checkpoint_dir",
        str(tmp_path / "checkpoint"),
        "--context_parallel_size",
        str(context_parallel_size),
    ]
    if enable_compile:
        command.append("--enable_compile")
    assert list(plan.command) == command
    assert plan.script == str(runtime.RUNTIME_ROOT / f"{entrypoint}.py")
    assert plan.workdir == str(output.parent if task_type == "t2v" else runtime.RUNTIME_ROOT)
    assert plan.env["WORLD_EVALS_LONGCAT_REQUEST_PATH"] == str(output.with_suffix(".json"))
    assert plan.env["WORLD_EVALS_LONGCAT_TASK"] == task_type
    assert plan.env["CUDA_VISIBLE_DEVICES"] == "2,5"
    assert "PYTHONPATH" not in plan.env
    assert json.loads(output.with_suffix(".json").read_text()) == request
    if context_parallel_size == 1:
        assert plan.env["RANK"] == plan.env["LOCAL_RANK"] == "0"
        assert plan.env["WORLD_SIZE"] == "1"
        assert plan.env["MASTER_ADDR"] == "127.0.0.1"
        assert plan.env["MASTER_PORT"] == str(29500 + os.getpid() % 1000)
    else:
        assert "MASTER_PORT" not in plan.env
    preflight = model.preflight()
    assert preflight["runtime_ready"] is True
    assert preflight["runtime_scripts"][task_type] == plan.script


def test_longcat_i2v_imports_its_family_scheduler() -> None:
    from worldfoundry.synthesis.visual_generation.longcat_video.longcat_video_runtime.longcat_video import (
        scheduling_flow_match_euler,
    )

    source = runtime.RUNTIME_ROOT / "run_inference_image_to_video.py"
    tree = ast.parse(source.read_text())
    scheduler_imports = [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and any(alias.name == "FlowMatchEulerDiscreteScheduler" for alias in node.names)
    ]

    assert scheduler_imports == [scheduling_flow_match_euler.__name__]


def test_longcat_released_scheduler_preserves_shifted_euler_steps() -> None:
    import torch

    from worldfoundry.synthesis.visual_generation.longcat_video.longcat_video_runtime.longcat_video.scheduling_flow_match_euler import (
        FlowMatchEulerDiscreteScheduler,
    )

    scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        {"shift": 12.0, "use_dynamic_shifting": False, "invert_sigmas": False, "time_shift_type": "linear"}
    )
    assert isinstance(scheduler, FlowMatchEulerDiscreteScheduler)
    sigmas = torch.linspace(1, 0.001, 50, dtype=torch.float32)
    scheduler.set_timesteps(50, sigmas=sigmas.tolist(), device="cpu")
    assert scheduler.sigmas is not None
    assert scheduler.timesteps is not None
    expected_sigmas = 12.0 * sigmas / (1 + 11.0 * sigmas)
    torch.testing.assert_close(scheduler.sigmas[:-1], expected_sigmas, rtol=0, atol=0)
    sample = torch.FloatTensor(torch.randn(1, 4, 2, 4, 4, generator=torch.Generator().manual_seed(17)))
    prediction = torch.FloatTensor(torch.ones_like(sample))

    actual = scheduler.step(prediction, float(scheduler.timesteps[0]), sample, return_dict=False)[0]

    expected = sample + (expected_sigmas[1] - expected_sigmas[0]) * prediction
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
