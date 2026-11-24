# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml

from worldfoundry.synthesis.visual_generation.inspatio_world import worldfoundry_runtime as runtime_module
from worldfoundry.synthesis.visual_generation.inspatio_world.inspatio_world_runtime.scripts import (
    run_da3_parallel,
    run_render_parallel,
)
from worldfoundry.synthesis.visual_generation.inspatio_world.inspatio_world_synthesis import InspatioWorldSynthesis
from worldfoundry.synthesis.visual_generation.inspatio_world.worldfoundry_runtime import InspatioWorldRuntime

pytestmark = pytest.mark.unit
RUNTIME_MODULE = "worldfoundry.synthesis.visual_generation.inspatio_world.inspatio_world_runtime"


def test_inspatio_plan_names_package_entrypoint_and_checks_only_assets(monkeypatch: pytest.MonkeyPatch) -> None:
    is_file = Path.is_file
    exists = Path.exists

    def asset_is_file(path: Path) -> bool:
        assert path.suffix != ".py", f"package code is not a staged asset: {path}"
        return is_file(path)

    def asset_exists(path: Path) -> bool:
        assert path != runtime_module._RUNTIME_ROOT
        return exists(path)

    monkeypatch.setattr(Path, "is_file", asset_is_file)
    monkeypatch.setattr(Path, "exists", asset_exists)
    result = InspatioWorldRuntime().plan(skip_step1=True, skip_step2=True, skip_step3=True)
    assert result["status"] == "ready"
    assert result["entrypoint"] == f"{RUNTIME_MODULE}.run_inference_pipeline"
    assert result["runtime_root"] == result["cwd"] == str(runtime_module._RUNTIME_ROOT)
    assert result["config_path"] == str(runtime_module.DEFAULT_CONFIG_PATH)


@pytest.mark.parametrize("factory", [InspatioWorldRuntime, InspatioWorldSynthesis])
def test_inspatio_constructors_do_not_accept_code_roots(factory: Any) -> None:
    assert not hasattr(factory(), "repo_root")
    with pytest.raises(TypeError, match="repo_root"):
        factory(repo_root="/external/source")


@pytest.mark.parametrize("pythonpath", [None, "/operator/packages"])
def test_inspatio_environment_preserves_operator_import_path(
    monkeypatch: pytest.MonkeyPatch, pythonpath: str | None
) -> None:
    monkeypatch.delenv("PYTHONPATH", raising=False)
    if pythonpath is not None:
        monkeypatch.setenv("PYTHONPATH", pythonpath)
    env = InspatioWorldRuntime()._build_runtime_env()
    assert env.get("PYTHONPATH") == pythonpath
    assert env["PYTHON"] == sys.executable
    assert env["WORLDFOUNDRY_INSPATIO_WORLD_CONFIG_ROOT"] == str(runtime_module._CONFIG_ROOT)
    assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == env["DIFFUSERS_OFFLINE"] == "1"


@pytest.mark.parametrize("tae_location", ["package", "checkpoints", "hfd"])
def test_inspatio_package_config_and_tae_fallbacks_are_preserved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tae_location: str
) -> None:
    installed = tmp_path / "package"
    monkeypatch.setattr(runtime_module, "_RUNTIME_ROOT", installed)
    monkeypatch.setattr(runtime_module, "_CONFIG_ROOT", tmp_path / "data")
    monkeypatch.setattr(runtime_module, "checkpoint_root_path", lambda *parts: tmp_path.joinpath("checkpoints", *parts))
    monkeypatch.setattr(runtime_module, "hfd_root_path", lambda *parts: tmp_path.joinpath("hfd", *parts))
    config = installed / "configs/custom.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("{}")
    roots = {
        "package": installed / "checkpoints/taehv",
        "checkpoints": tmp_path / "checkpoints/taehv",
        "hfd": tmp_path / "hfd/custom--taehv",
    }
    checkpoint = roots[tae_location] / "taew2_1.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    runtime = InspatioWorldRuntime()
    assert runtime._resolve_config_path("configs/custom.yaml") == str(config)
    assert runtime._resolve_optional_tae_checkpoint(None, True) == str(checkpoint)
    assert runtime._resolve_optional_tae_checkpoint(None, False) is None


def test_inspatio_predict_uses_package_worker_and_preserves_request_and_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video = tmp_path / "camera '# clip.mp4"
    video.touch()
    trajectory = tmp_path / "trajectory.txt"
    trajectory.touch()
    checkpoint = tmp_path / "InSpatio-World-1.3B.safetensors"
    checkpoint.touch()
    tae = tmp_path / "taew2_1.pth"
    tae.touch()
    config = tmp_path / "inference.yaml"
    config.write_text(yaml.safe_dump({"wan_model_folder": "old", "generator": {"weight_list": [{"path": "old"}]}}))
    monkeypatch.setenv("PYTHONPATH", "/operator/packages")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6,7")
    calls: list[tuple[list[str], dict[str, Any]]] = []
    frames = np.zeros((2, 4, 4, 3), dtype=np.uint8)

    def run(command: list[str], **options: Any) -> subprocess.CompletedProcess[str]:
        calls.append((command, options))
        output = Path(command[command.index("--output_folder") + 1])
        (output / "scene-pred_video_rank0.mp4").write_bytes(b"generated video")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runtime_module.subprocess, "run", run)
    monkeypatch.setattr(runtime_module, "load_video_frames", lambda path: frames)
    runtime = InspatioWorldRuntime(
        checkpoint_source=str(checkpoint),
        wan_model_source=str(tmp_path),
        da3_model_source=str(tmp_path),
        florence_model_source=str(tmp_path),
    )
    result = runtime.predict(
        video,
        traj_txt_path=str(trajectory),
        config_path=str(config),
        prompt="A garden",
        output_root=str(tmp_path / "output"),
        step1_gpus="2",
        step3_nproc=2,
        master_port=29587,
        relative_to_source=True,
        rotation_only=True,
        adaptive_frame=False,
        freeze_repeat=3,
        freeze_frame=0,
        use_tae=True,
        tae_checkpoint_path=str(tae),
        compile_dit=True,
        seed=1730,
        deterministic=True,
        show_progress=False,
        return_dict=True,
    )
    ((command, options),) = calls
    assert command[:3] == [sys.executable, "-m", f"{RUNTIME_MODULE}.run_inference_pipeline"]
    for name, value in {
        "checkpoint_path": str(checkpoint),
        "config_path": result["config_path"],
        "traj_txt_path": str(trajectory),
        "step1_gpus": "2",
        "step2_gpus": "6,7",
        "step3_gpus": "6,7",
        "step3_nproc": "2",
        "master_port": "29587",
        "freeze_repeat": "3",
        "freeze_frame": "0",
        "tae_checkpoint_path": str(tae),
        "seed": "1730",
    }.items():
        assert command[command.index(f"--{name}") + 1] == value
    assert {
        "--skip_step1",
        "--relative_to_source",
        "--rotation_only",
        "--disable_adaptive_frame",
        "--use_tae",
        "--compile_dit",
        "--deterministic",
    }.issubset(command)
    assert options["cwd"] == str(runtime_module._RUNTIME_ROOT)
    assert options["env"]["PYTHONPATH"] == "/operator/packages"
    assert options["env"]["WORLDFOUNDRY_INSPATIO_WORLD_SEED"] == options["env"]["PYTHONHASHSEED"] == "1730"
    assert options["check"] is True and options["stdout"] == subprocess.DEVNULL
    assert options["stderr"] == subprocess.STDOUT
    assert result["video"] is frames
    assert Path(result["generated_video_path"]).read_bytes() == b"generated video"
    staged = Path(result["input_dir"])
    assert (staged / video.name).resolve() == video
    assert json.loads((staged / "new.json").read_text())[0]["text"] == "A garden"
    materialized = yaml.safe_load(Path(result["config_path"]).read_text())
    assert materialized["wan_model_folder"] == materialized["generator"]["weight_list"][0]["path"] == str(tmp_path)
    assert yaml.safe_load(config.read_text())["wan_model_folder"] == "old"


@pytest.mark.parametrize("cached, failure", [(False, None), (True, None), (False, "depth"), (False, "convert")])
def test_run_da3_parallel_uses_package_workers_and_preserves_resume_and_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cached: bool, failure: str | None
) -> None:
    entry = {"video_path": str(tmp_path / "input.mp4"), "vggt_depth_path": str(tmp_path / "depth")}
    if cached:
        (tmp_path / "depth_da3_tmp/frames_pcd").mkdir(parents=True)
    config = json.dumps({"model_path": "/weights/DA3", "fix_resize": True})
    monkeypatch.setenv("PYTHONPATH", "/operator/packages")
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def run(command: list[str], **options: Any) -> subprocess.CompletedProcess[str]:
        calls.append((command, options))
        stage = "depth" if command[2].endswith("depth_predict_da3_cli") else "convert"
        return subprocess.CompletedProcess(command, int(stage == failure))

    monkeypatch.setattr(run_da3_parallel.subprocess, "run", run)
    assert run_da3_parallel.process_video((0, entry, "3", 1, config)) is (failure is None)
    depth_module = f"{RUNTIME_MODULE}.depth.depth_predict_da3_cli"
    convert_module = f"{RUNTIME_MODULE}.scripts.convert_da3_to_pi3"
    expected = [convert_module] if cached else [depth_module] if failure == "depth" else [depth_module, convert_module]
    assert [command[:3] for command, _ in calls] == [[sys.executable, "-m", name] for name in expected]
    assert all(options["env"]["CUDA_VISIBLE_DEVICES"] == "3" for _, options in calls)
    assert all(options["env"]["PYTHONPATH"] == "/operator/packages" for _, options in calls)
    if not cached:
        assert calls[0][0][3:] == [
            "--input",
            entry["video_path"],
            "--output",
            entry["vggt_depth_path"] + "_da3_tmp",
            "--config-json",
            config,
        ]
    if failure != "depth":
        assert calls[-1][0][3:] == [
            "--da3_dir",
            entry["vggt_depth_path"] + "_da3_tmp",
            "--output_dir",
            entry["vggt_depth_path"],
            "--video_path",
            entry["video_path"],
        ]


@pytest.mark.parametrize("with_options, returncode", [(False, 0), (True, 0), (True, 1)])
def test_run_render_parallel_uses_package_worker_and_preserves_render_options(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, with_options: bool, returncode: int
) -> None:
    entry = {"video_path": str(tmp_path / "input.mp4"), "vggt_depth_path": str(tmp_path / "depth")}
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def run(command: list[str], **options: Any) -> subprocess.CompletedProcess[str]:
        calls.append((command, options))
        return subprocess.CompletedProcess(command, returncode)

    monkeypatch.setattr(run_render_parallel.subprocess, "run", run)
    success = run_render_parallel.render_video(
        (
            0,
            entry,
            "4",
            1,
            "/data/trajectory.txt",
            832,
            480,
            with_options,
            with_options,
            3 if with_options else 0,
            0 if with_options else None,
        )
    )
    assert success is (returncode == 0)
    ((command, options),) = calls
    assert command[:3] == [sys.executable, "-m", f"{RUNTIME_MODULE}.scripts.render_point_cloud"]
    expected = [
        "--da3_dir",
        entry["vggt_depth_path"] + "_da3_tmp",
        "--traj_txt_path",
        "/data/trajectory.txt",
        "--output_dir",
        str(tmp_path / "depth/render"),
        "--width",
        "832",
        "--height",
        "480",
    ]
    if with_options:
        expected.extend(["--relative_to_source", "--rotation_only", "--freeze_repeat", "3", "--freeze_frame", "0"])
    assert command[3:] == expected
    assert options["env"]["CUDA_VISIBLE_DEVICES"] == "4"


@pytest.mark.parametrize("stage", ["depth", "render"])
def test_inspatio_parallel_entrypoints_assign_gpus_without_source_file_options(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stage: str
) -> None:
    entries = [
        {"video_path": f"/data/input{index}.mp4", "vggt_depth_path": str(tmp_path / f"depth{index}")}
        for index in range(3)
    ]
    source = tmp_path / "new.json"
    source.write_text(json.dumps(entries))
    worker = run_da3_parallel if stage == "depth" else run_render_parallel
    argv = [worker.__name__, "--json_path", str(source), "--gpu_list", "2,3"]
    argv.extend(["--da3_config", "{}"] if stage == "depth" else ["--traj_txt_path", "/data/camera.txt"])
    monkeypatch.setattr(sys, "argv", argv)
    pools: list[int] = []

    def pool(processes: int) -> Any:
        pools.append(processes)
        return nullcontext(SimpleNamespace(map=lambda fn, tasks: [fn(task) for task in tasks]))

    calls: list[dict[str, str]] = []

    def run(command: list[str], **options: Any) -> subprocess.CompletedProcess[str]:
        calls.append(options["env"])
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(worker.multiprocessing, "Pool", pool)
    monkeypatch.setattr(worker.subprocess, "run", run)
    worker.main()
    assert pools == [2]
    assert [env["CUDA_VISIBLE_DEVICES"] for env in calls] == (
        ["2", "2", "3", "3", "2", "2"] if stage == "depth" else ["2", "3", "2"]
    )
