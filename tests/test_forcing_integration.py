from __future__ import annotations

import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from worldfoundry.evaluation.models import discover_model_registry
from worldfoundry.operators.forcing_operator import CausalForcingOperator, SelfForcingOperator
from worldfoundry.pipelines.forcing.pipeline_forcing import CausalForcingPipeline
from worldfoundry.synthesis.visual_generation.forcing import runtime as runtime_module
from worldfoundry.synthesis.visual_generation.forcing.runtime import RollingForcingRuntime, SelfForcingRuntime


def test_self_forcing_operator_accepts_prompt_and_optional_image() -> None:
    operator = SelfForcingOperator()
    operator.get_interaction(123)

    interaction = operator.process_interaction()
    perception = operator.process_perception(images="/tmp/input.png")
    prompt = operator.process_prompt("A detailed cinematic scene.")

    assert interaction["seed"] == 123
    assert perception["images"] == "/tmp/input.png"
    assert prompt["prompt"] == "A detailed cinematic scene."


def test_causal_forcing_operator_accepts_prompt_and_optional_image() -> None:
    operator = CausalForcingOperator()
    operator.get_interaction(7)

    interaction = operator.process_interaction()
    perception = operator.process_perception(images="/tmp/input.png")
    prompt = operator.process_prompt("A detailed causal scene.")

    assert interaction["seed"] == 7
    assert perception["images"] == "/tmp/input.png"
    assert prompt["prompt"] == "A detailed causal scene."


def test_causal_forcing_pipeline_forwards_model_specific_runtime_options() -> None:
    pipe = CausalForcingPipeline.from_pretrained(
        model_path={
            "runtime_root": "/tmp/Causal-Forcing",
            "checkpoint_path": "/tmp/causal.pt",
            "config_path": "/tmp/config.yaml",
            "wan_models_root": "/tmp/wan",
        },
        model_id="causal-forcing",
        device="cuda:2",
    )

    runtime = pipe.synthesis_model.runtime
    assert runtime.model_id == "causal-forcing"
    assert runtime.runtime_root == "/tmp/Causal-Forcing"
    assert runtime.checkpoint_path == "/tmp/causal.pt"
    assert runtime.config_path == "/tmp/config.yaml"
    assert runtime.wan_models_root == "/tmp/wan"
    assert runtime.device == "cuda:2"


def test_forcing_runtime_builds_official_command(monkeypatch, tmp_path: Path) -> None:
    runtime_root = tmp_path / "Self-Forcing"
    wan_root = tmp_path / "ckpt"
    checkpoint = tmp_path / "self_forcing_dmd.pt"
    config = runtime_root / "configs" / "self_forcing_dmd.yaml"
    (runtime_root / "configs").mkdir(parents=True)
    (wan_root / "Wan2.1-T2V-1.3B").mkdir(parents=True)
    (wan_root / "Wan2.1-T2V-14B").mkdir(parents=True)
    (runtime_root / "inference.py").write_text("print('stub')\n", encoding="utf-8")
    config.write_text("generator_ckpt: checkpoints/ode_init.pt\n", encoding="utf-8")
    checkpoint.write_bytes(b"stub")

    captured = {}

    def fake_run(command, check, cwd, env, stdout, stderr, text):
        del check, env, stdout, stderr, text
        captured["command"] = command
        captured["cwd"] = cwd
        output_dir = Path(command[command.index("--output_folder") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "0-0_ema.mp4").write_bytes(b"video")
        return subprocess.CompletedProcess(command, 0, stdout="")

    monkeypatch.setattr(runtime_module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        runtime_module,
        "_load_video_frames",
        lambda _: np.zeros((2, 8, 8, 3), dtype=np.uint8),
    )

    model = SelfForcingRuntime(
        runtime_root=runtime_root,
        checkpoint_path=checkpoint,
        config_path=config,
        wan_models_root=wan_root,
        python_executable="/tmp/python",
    )
    result = model.predict(
        prompt="demo prompt",
        output_path=tmp_path / "result.mp4",
        num_output_frames=3,
        show_progress=False,
    )

    command = captured["command"]
    assert command[:2] == ["/tmp/python", str((runtime_root / "inference.py").resolve())]
    assert command[command.index("--config_path") + 1] == str(config.resolve())
    assert command[command.index("--checkpoint_path") + 1] == str(checkpoint.resolve())
    assert command[command.index("--num_output_frames") + 1] == "3"
    assert "--save_with_index" in command
    assert "--use_ema" in command
    assert Path(captured["cwd"]).name == "runtime_cwd"
    assert result["artifact_path"] == str((tmp_path / "result.mp4").resolve())
    assert result["video"].shape == (2, 8, 8, 3)


def test_forcing_runtime_resolves_flat_hfd_wan_layout(tmp_path: Path) -> None:
    wan_root = tmp_path / "ckpts"
    hfd_model = wan_root / "Wan-AI--Wan2.1-T2V-1.3B"
    hfd_model.mkdir(parents=True)

    runtime = SelfForcingRuntime(wan_models_root=wan_root)
    plan = runtime.runtime_plan()
    runtime_cwd = runtime._runtime_cwd(tmp_path / "run")

    assert plan["wan_1_3b_exists"] is True
    assert (runtime_cwd / "wan_models" / "Wan2.1-T2V-1.3B").resolve() == hfd_model.resolve()


def test_rolling_forcing_runtime_uses_subclass_model_and_flat_checkpoint_layout(
    monkeypatch, tmp_path: Path
) -> None:
    checkpoint_root = tmp_path / "ckpts"
    checkpoint = (
        checkpoint_root
        / "TencentARC--RollingForcing"
        / "checkpoints"
        / "rolling_forcing_dmd.pt"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"stub")
    wan_model = checkpoint_root / "Wan-AI--Wan2.1-T2V-1.3B"
    wan_model.mkdir(parents=True)
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(checkpoint_root))

    runtime = RollingForcingRuntime(wan_models_root=checkpoint_root)
    plan = runtime.runtime_plan()

    assert runtime.model_id == "rolling-forcing"
    assert Path(runtime.checkpoint_path).resolve() == checkpoint.resolve()
    assert plan["checkpoint_exists"] is True
    assert plan["wan_1_3b_exists"] is True
    assert plan["license"] == (
        "RollingForcing academic-only; commercial and production use prohibited"
    )


def test_rolling_forcing_inference_contract_honors_hfd_root(tmp_path: Path) -> None:
    hfd_root = tmp_path / "hfd"
    env = dict(os.environ)
    env["WORLDFOUNDRY_HFD_ROOT"] = str(hfd_root)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json; "
                "from worldfoundry.runtime.inference_catalog import get_model_inference_spec; "
                "spec = get_model_inference_spec('rolling-forcing'); "
                "print(json.dumps({'checkpoints': spec.variants[0].checkpoint_map(), "
                "'load_kwargs': spec.variants[0].load_kwargs, "
                "'prompt': spec.tasks[0].inputs[0].default}))"
            ),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["checkpoints"] == {
        "primary": str(
            hfd_root
            / "TencentARC--RollingForcing"
            / "checkpoints"
            / "rolling_forcing_dmd.pt"
        ),
        "base": str(hfd_root / "Wan-AI--Wan2.1-T2V-1.3B"),
    }
    assert payload["load_kwargs"]["wan_models_root"] == str(hfd_root)
    assert "golden retriever" in payload["prompt"].lower()


def test_forcing_runtime_rejects_invalid_latent_frame_geometry_before_launch(
    monkeypatch, tmp_path: Path
) -> None:
    model = SelfForcingRuntime()
    launched = False

    def fake_run(*args, **kwargs):
        nonlocal launched
        launched = True
        raise AssertionError("subprocess must not launch")

    monkeypatch.setattr(runtime_module.subprocess, "run", fake_run)

    with pytest.raises(ValueError, match="must follow 3k"):
        model.predict(prompt="demo", output_path=tmp_path / "invalid.mp4", num_output_frames=5)
    assert launched is False


def test_forcing_video_compat_replaces_present_torchvision_writer() -> None:
    import torchvision.io as tv_io

    original = tv_io.write_video
    try:
        runpy.run_path(str(runtime_module.TORCHVISION_VIDEO_COMPAT_DIR / "sitecustomize.py"))
        assert tv_io.write_video is not original
    finally:
        tv_io.write_video = original


def test_model_registry_contains_forcing_models() -> None:
    registry = discover_model_registry()
    self_model = registry.get("self-forcing")
    causal_model = registry.get("causal-forcing")
    rolling_model = registry.get("rolling-forcing")

    assert self_model.has_loader is True
    assert self_model.has_infer is True
    assert self_model.pipeline_target == "worldfoundry.pipelines.forcing.pipeline_forcing:SelfForcingPipeline"
    assert causal_model.has_loader is True
    assert causal_model.has_infer is True
    assert causal_model.pipeline_target == "worldfoundry.pipelines.forcing.pipeline_forcing:CausalForcingPipeline"
    assert rolling_model.has_loader is True
    assert rolling_model.has_infer is True
    assert rolling_model.pipeline_target == (
        "worldfoundry.pipelines.forcing.pipeline_forcing:RollingForcingPipeline"
    )
