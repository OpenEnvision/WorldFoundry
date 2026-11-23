# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Weight-free contracts for the shared official CLI and manifest launchers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from worldfoundry.core.execution.process import package_module_command
from worldfoundry.synthesis.visual_generation.official_video_runtime import OfficialVideoRuntime
from worldfoundry.synthesis.visual_generation.runtime_manifest import WorldModelRuntimeSpec, WorldModelRuntimeSynthesis

OFFICIAL_MODULES = {
    "framepack": "framepack.worldfoundry_runner",
    "helios": "helios.infer_helios",
    "krea-realtime-video": "krea_realtime.worldfoundry_runner",
    "magi-1": "magi.worldfoundry_runner",
    "mmaudio": "mmaudio.run_official_demo",
    "open-sora": "open_sora.open_sora_runtime.scripts.inference",
    "open-sora-plan": "open_sora_plan.open_sora_plan_runtime.opensora.sample.sample",
    "sama-14b": "sama.sama_runtime.infer_sh.inference",
    "unianimate-dit": "unianimate_dit.worldfoundry_runner",
}


@pytest.mark.parametrize("model_id", OFFICIAL_MODULES)
def test_official_cli_config_launches_its_package_module(model_id: str, tmp_path: Path) -> None:
    runtime = OfficialVideoRuntime.from_model_id(model_id)
    module = f"worldfoundry.synthesis.visual_generation.{OFFICIAL_MODULES[model_id]}"
    assert runtime.runtime["module"] == module
    assert runtime.runtime.get("cwd") == ("{checkpoint_path}" if model_id == "mmaudio" else None)
    assert importlib.util.find_spec(module) is not None
    assert {"command", "repo_root", "repo_root_candidates", "prepend_repo_root_to_pythonpath"}.isdisjoint(
        runtime.runtime
    )
    assert "--repo-root" not in runtime.runtime["args"]
    assert "--worldfoundry-demo-script" not in runtime.runtime["args"]
    plan = runtime.runtime_plan(output_path=tmp_path / "result.mp4", prompt="test prompt", extra={"nproc_per_node": 2})
    command = plan["command"]
    nproc = 2 if model_id in {"helios", "open-sora-plan"} else None
    prefix = package_module_command(module, nproc_per_node=nproc)
    assert command[: len(prefix)] == prefix


@pytest.mark.parametrize("source_option", ["repo_root", "repo_root_candidates"])
def test_official_cli_rejects_source_root_options(source_option: str) -> None:
    with pytest.raises(TypeError, match="source-root options"):
        OfficialVideoRuntime.from_model_id("framepack", **{source_option: "/external/code"})


@pytest.mark.unit
def test_framepack_image_conditioning_uses_the_request_seed() -> None:
    import torch
    from diffusers.models.autoencoders.autoencoder_kl_hunyuan_video import AutoencoderKLHunyuanVideo

    from worldfoundry.synthesis.visual_generation.framepack.framepack_runtime.diffusers_helper.hunyuan import (
        vae_encode,
    )

    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        vae = AutoencoderKLHunyuanVideo(
            latent_channels=2,
            down_block_types=("HunyuanVideoDownBlock3D",),
            up_block_types=("HunyuanVideoUpBlock3D",),
            block_out_channels=(4,),
            layers_per_block=1,
            norm_num_groups=1,
            spatial_compression_ratio=1,
            mid_block_add_attention=False,
        ).eval()
        image = torch.linspace(-1, 1, 192).reshape(1, 3, 1, 8, 8)

        ambient_rng = torch.get_rng_state()
        first = vae_encode(image, vae, seed=7)
        changed = vae_encode(image, vae, seed=8)
        torch.testing.assert_close(torch.get_rng_state(), ambient_rng, rtol=0, atol=0)
        torch.manual_seed(7)
        expected = vae.encode(image).latent_dist.sample() * vae.config["scaling_factor"]
        repeated = vae_encode(image, vae, seed=7)

    assert not torch.equal(first, changed)
    torch.testing.assert_close(first, expected, rtol=0, atol=0)
    torch.testing.assert_close(first, repeated, rtol=0, atol=0)


@pytest.mark.parametrize("pythonpath", [None, "/operator/site-packages"])
def test_official_cli_inherits_environment_and_runs_in_output_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pythonpath: str | None
) -> None:
    import worldfoundry.synthesis.visual_generation.official_video_runtime as official

    if pythonpath is None:
        monkeypatch.delenv("PYTHONPATH", raising=False)
    else:
        monkeypatch.setenv("PYTHONPATH", pythonpath)
    runtime = OfficialVideoRuntime.from_model_id("framepack", device="cuda:2")
    output_path = tmp_path / "run" / "result.mp4"
    calls = []

    def run(command: list[str], **kwargs: Any) -> SimpleNamespace:
        calls.append((command, kwargs))
        output_path.write_bytes(b"test artifact")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(official.subprocess, "run", run)
    result = runtime._run_official_cli(
        prompt="motion",
        output_path=output_path,
        image_path=tmp_path / "input.png",
        video_path=None,
        checkpoint_path=tmp_path / "checkpoint",
        extra={"cfg": 2.5, "visible_devices": "2"},
    )
    command, kwargs = calls[0]
    assert command[:3] == [sys.executable, "-m", runtime.runtime["module"]]
    assert command[command.index("--cfg") + 1] == "2.5"
    assert kwargs["cwd"] == str(output_path.parent)
    assert kwargs["env"].get("PYTHONPATH") == pythonpath
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "2"
    assert Path(result["artifact_path"]).read_bytes() == b"test artifact"


def test_manifest_has_no_script_fallback_or_pythonpath_hook(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PYTHONPATH", "/operator/site-packages")
    runtime = WorldModelRuntimeSynthesis(
        spec=WorldModelRuntimeSpec(model_id="fixture", display_name="fixture"),
        runtime_root=tmp_path,
        entrypoint=tmp_path / "entry.py",
        blocked_reason="",
        device="cpu",
    )
    paths = {"output_path": tmp_path / "out.mp4", "prompt": "motion", "plan_path": tmp_path / "plan.json"}
    assert runtime._command(**paths) == []
    env = runtime._subprocess_env(**paths)
    assert env["PYTHONPATH"] == "/operator/site-packages"
    assert env["WORLDFOUNDRY_OUTPUT_DIR"] == str(tmp_path)
    assert "pythonpath_builder_attr" not in WorldModelRuntimeSpec.__dataclass_fields__


def test_mira_and_vid2world_commands_preserve_inference_options(tmp_path: Path) -> None:
    from worldfoundry.synthesis.visual_generation.mira_wm import worldfoundry_runtime as mira
    from worldfoundry.synthesis.visual_generation.vid2world import worldfoundry_runtime as vid2world

    context = {"python": "/env/python", "output_path": str(tmp_path / "out.mp4"), "output_dir": str(tmp_path)}
    mira_command = mira.build_command(
        {
            **context,
            "options": {
                "checkpoint_path": "/weights/mira",
                "dataset_path": "/data/clip",
                "device": "cuda:1",
                "seed": 7,
                "compile": True,
                "overlay_actions": False,
                "generated_only": True,
            },
        }
    )
    assert mira_command[:3] == ["/env/python", "-m", "worldfoundry.synthesis.visual_generation.mira_wm.infer"]
    assert mira_command[mira_command.index("--device") + 1] == "cuda:1"
    assert mira_command[mira_command.index("--seed") + 1] == "7"
    assert mira_command[-3:] == ["--compile", "--no-overlay-actions", "--generated-only"]

    vid_command = vid2world.build_command({**context, "options": {"devices": 2, "master_port": 23456}})
    assert vid_command == package_module_command(
        "worldfoundry.synthesis.visual_generation.vid2world.main.inference",
        [
            "--base",
            str(vid2world.CONFIG_ROOT / vid2world.DEFAULT_CONFIG),
            "--val",
            "--name",
            "worldfoundry_infer",
            "--logdir",
            str(tmp_path),
            "--devices",
            "2",
            "lightning.trainer.num_nodes=1",
        ],
        python_executable="/env/python",
        nproc_per_node=2,
        launcher_options=("--nnodes", "1", "--master_addr", "127.0.0.1", "--master_port", "23456"),
    )


@pytest.mark.parametrize("existing", [False, True])
def test_magi_installs_triton_shim_before_constructing_pipeline(monkeypatch, existing: bool) -> None:
    import torch

    from worldfoundry.synthesis.visual_generation.magi.magi_runtime.inference.pipeline import entry

    args = SimpleNamespace(config_file="config.json", mode="t2v", prompt="motion", output_path="out.mp4")
    monkeypatch.setattr(entry, "parse_arguments", lambda: args)
    if existing:
        expected = Mock(side_effect=lambda kernel: kernel)
        monkeypatch.setattr(torch.library, "wrap_triton", expected, raising=False)
    else:
        monkeypatch.setattr(torch.library, "wrap_triton", None, raising=False)
        monkeypatch.delattr(torch.library, "wrap_triton")
        expected = entry._wrap_triton
    pipeline = Mock()

    def build(config_file: str) -> Mock:
        assert getattr(torch.library, "wrap_triton") is expected
        assert config_file == "config.json"
        return pipeline

    module = ModuleType(f"{entry.__package__}.pipeline")
    monkeypatch.setattr(module, "MagiPipeline", build, raising=False)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    entry.main()
    pipeline.run_text_to_video.assert_called_once_with(prompt="motion", output_path="out.mp4")


def test_mmaudio_calls_the_packaged_demo_with_original_arguments(monkeypatch) -> None:
    from worldfoundry.synthesis.visual_generation.mmaudio import run_official_demo as runner

    calls = []
    monkeypatch.setattr(runner, "_patch_torchaudio_save", lambda: calls.append("audio-save"))
    module = ModuleType(f"{runner.__package__}.official_demo")
    monkeypatch.setattr(module, "main", lambda argv: calls.append(argv), raising=False)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    arguments = ["--prompt", "footsteps", "--duration", "4", "--full_precision"]
    assert runner.main(arguments) == 0
    assert calls == ["audio-save", arguments]


@pytest.mark.parametrize("model_id", ["helios", "open-sora-plan"])
def test_studio_reads_official_cli_process_count(model_id: str) -> None:
    from worldfoundry.studio.inference.dispatch import _official_video_internal_torchrun_nproc

    assert _official_video_internal_torchrun_nproc(model_id, call_kwargs={}, load_kwargs={}) == 1
    assert (
        _official_video_internal_torchrun_nproc(
            model_id, call_kwargs={"nproc_per_node": 3}, load_kwargs={"nproc_per_node": 2}
        )
        == 3
    )
    assert (
        _official_video_internal_torchrun_nproc(model_id, call_kwargs={}, load_kwargs={"torchrun_nproc_per_node": 2})
        == 2
    )
    assert _official_video_internal_torchrun_nproc("framepack", call_kwargs={}, load_kwargs={}) == 0
