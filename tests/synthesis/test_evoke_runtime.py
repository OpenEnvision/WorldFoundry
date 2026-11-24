# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Weight-free contracts for Evoke's package workers and canonical depth loader."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from worldfoundry.synthesis.visual_generation.evoke import worldfoundry_runtime as runtime_module

pytestmark = pytest.mark.unit

_INFERENCE = "worldfoundry.synthesis.visual_generation.evoke.evoke_runtime.scripts.inference.infer_single"
_POSTPROCESS = "worldfoundry.synthesis.visual_generation.evoke.evoke_runtime.scripts.inference.postprocess_viz"


@pytest.fixture
def evoke_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> runtime_module.EvokeRuntime:
    checkpoint = tmp_path / "checkpoint"
    for component in ("vae", "scheduler", "text_encoder", "tokenizer"):
        (checkpoint / "evoke-base" / component).mkdir(parents=True)
    (checkpoint / "evoke" / "stage3_post_distillation" / "transformer").mkdir(parents=True)
    vigeo = tmp_path / "vigeo"
    vigeo.mkdir()
    (vigeo / "vigeo.pt").touch()
    monkeypatch.setattr(runtime_module.EvokeRuntime, "_require_runtime_compatibility", lambda self: None)
    return runtime_module.EvokeRuntime(
        checkpoint_path=checkpoint,
        vigeo_path=vigeo,
        python_executable="/model-env/bin/python",
        device="cuda:1",
        env={"CUDA_VISIBLE_DEVICES": "4,7", "EVOKE_CPU_THREADS": "2"},
    )


@pytest.mark.parametrize(
    "mode,geometric", [("t2v", False), ("i2v", False), ("i2v", True), ("v2v", False), ("v2v", True)]
)
def test_evoke_predict_launches_package_with_unchanged_recipe(
    evoke_runtime: runtime_module.EvokeRuntime,
    mode: str,
    geometric: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def execute(command: list[str], **kwargs: Any) -> dict[str, Any]:
        calls.append((command, kwargs))
        return {"video": str(kwargs["output_path"])}

    monkeypatch.setattr(runtime_module, "execute_in_tree", execute)
    monkeypatch.setenv("PYTHONPATH", "/caller-path")
    image, video, pose = (tmp_path / name for name in ("frame.png", "source.mp4", "pose.npz"))
    for path in (image, video, pose):
        path.touch()
    result = evoke_runtime.predict(
        prompt="A camera moves through a garden",
        output_path=tmp_path / "result.mp4",
        image_path=image if mode == "i2v" else None,
        video_path=video if mode == "v2v" else None,
        trajectory_npz=pose if geometric else None,
        mode=mode,
        num_chunks=2,
        seed=17,
        ref_seconds=2.5,
    )
    command, kwargs = calls[0]
    assert command[:3] == ["/model-env/bin/python", "-m", _INFERENCE]
    assert command[command.index("--sample_type") + 1] == mode
    assert command[command.index("--num_frames") + 1] == "66"
    assert command[command.index("--seed") + 1] == "17"
    assert command[command.index("--stage2_steps") + 1 : command.index("--stage2_steps") + 4] == ["1"] * 3
    assert command[command.index("--vae_decode_type") + 1] == "persistent"
    assert ("--use_geometric_state" in command) is geometric
    if geometric:
        assert command[command.index("--geo_depth_backend") + 1] == "vigeo"
        assert command[command.index("--geo_vigeo_weights") + 1] == str(tmp_path / "vigeo")
    assert "python_paths" not in kwargs
    assert kwargs["cwd"] == evoke_runtime.bundled_repo_root()
    assert kwargs["env"]["PYTHONPATH"] == "/caller-path"
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "7"
    assert kwargs["env"]["EVOKE_CPU_THREADS"] == "2"
    assert all(kwargs["env"][key] == value for key, value in runtime_module.OFFLINE_ENV.items())
    assert result["video"] == str(tmp_path / "result.mp4")
    assert result["metadata"]["mode"] == mode
    assert result["metadata"]["use_geometric_state"] is geometric


def test_evoke_preflight_reports_package_module(
    evoke_runtime: runtime_module.EvokeRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(evoke_runtime, "runtime_compatibility", lambda: {"ready": True})
    result = evoke_runtime.preflight(use_geometric_state=True)
    assert result["status"] == "ready"
    assert result["runtime_module"] == _INFERENCE
    assert "runtime_script" not in result
    assert result["paths"]["vigeo_path"]
    retired_options: dict[str, Any] = {"runtime_root": "/external/source"}
    with pytest.raises(TypeError, match="runtime_root"):
        runtime_module.EvokeRuntime(**retired_options)


@pytest.mark.parametrize("option", ["--geo_vigeo_src", "--geo_da3_src"])
def test_evoke_parser_rejects_source_overrides(option: str) -> None:
    from worldfoundry.synthesis.visual_generation.evoke.evoke_runtime.scripts.inference import infer_single

    with pytest.raises(SystemExit) as caught:
        infer_single.parse_args(
            ["--ckpt_path", "/weights", "--prompt", "garden", "--output_folder", "/output", option, "/source"]
        )
    assert caught.value.code == 2


def test_evoke_background_postprocess_runs_cpu_package_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from worldfoundry.synthesis.visual_generation.evoke.evoke_runtime.scripts.inference import infer_single

    output = tmp_path / "sample"
    output.mkdir()
    frames = np.arange(2 * 3 * 4 * 3, dtype=np.uint8).reshape(2, 3, 4, 3)
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def popen(command: list[str], **kwargs: Any) -> SimpleNamespace:
        calls.append((command, kwargs))
        kwargs["stdout"].close()
        return SimpleNamespace(pid=4321)

    monkeypatch.setattr(infer_single.subprocess, "Popen", popen)
    paths = list(sys.path)
    environment = dict(os.environ)
    infer_single._spawn_bg_postprocess(output, frames, {"fps": 12}, max_workers=2)
    command, kwargs = calls[0]
    assert command == [sys.executable, "-m", _POSTPROCESS, str(output)]
    assert kwargs["start_new_session"] is True
    assert kwargs["env"] == {**environment, "CUDA_VISIBLE_DEVICES": ""}
    assert sys.path == paths
    marker = tmp_path / ".pp_workers" / "sample.lock"
    assert marker.read_text() == "4321"
    assert json.loads((output / ".postproc_params.json").read_text()) == {"fps": 12, "_worker_marker": str(marker)}
    np.testing.assert_array_equal(np.load(output / ".pred_frames.npy"), frames)
