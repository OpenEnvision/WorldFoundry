from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from worldfoundry.operators.forcing_operator import RollingForcingOperator
from worldfoundry.synthesis.visual_generation.forcing import runtime as runtime_module
from worldfoundry.synthesis.visual_generation.forcing.runtime import RollingForcingRuntime
from worldfoundry.zoo.catalog.registry import discover_model_registry


def test_rolling_forcing_operator_accepts_prompt_and_rejects_images() -> None:
    operator = RollingForcingOperator()
    operator.get_interaction(7)

    assert operator.process_interaction()["seed"] == 7
    assert operator.process_prompt("A long scene.")["prompt"] == "A long scene."
    with pytest.raises(ValueError, match="text only"):
        operator.process_perception(images="/tmp/input.png")


def test_forcing_runtime_builds_official_command(monkeypatch, tmp_path: Path) -> None:
    wan_root = tmp_path / "ckpt"
    checkpoint = tmp_path / "rolling_forcing_dmd.pt"
    config = tmp_path / "configs" / "rolling_forcing_dmd.yaml"
    config.parent.mkdir(parents=True)
    (wan_root / "Wan2.1-T2V-1.3B").mkdir(parents=True)
    config.write_text("num_frame_per_block: 3\n", encoding="utf-8")
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

    model = RollingForcingRuntime(
        checkpoint_path=checkpoint,
        config_path=config,
        wan_models_root=wan_root,
        python_executable="/tmp/python",
    )
    result = model.predict(
        prompt="demo prompt",
        output_path=tmp_path / "result.mp4",
        num_output_frames=6,
        show_progress=False,
    )

    command = captured["command"]
    assert command[:3] == ["/tmp/python", "-m", "worldfoundry.synthesis.visual_generation.rolling_forcing.inference"]
    assert command[command.index("--config_path") + 1] == str(config.resolve())
    assert command[command.index("--checkpoint_path") + 1] == str(checkpoint.resolve())
    assert command[command.index("--num_output_frames") + 1] == "6"
    assert command[command.index("--wan_models_root") + 1] == str(wan_root.resolve())
    assert "--use_ema" in command
    assert "--save_with_index" not in command
    assert Path(captured["cwd"]).name == "runtime_cwd"
    assert result["artifact_path"] == str((tmp_path / "result.mp4").resolve())
    assert result["video"].shape == (2, 8, 8, 3)


def test_forcing_runtime_resolves_flat_hfd_wan_layout(tmp_path: Path) -> None:
    wan_root = tmp_path / "ckpts"
    hfd_model = wan_root / "Wan-AI--Wan2.1-T2V-1.3B"
    hfd_model.mkdir(parents=True)

    runtime = RollingForcingRuntime(wan_models_root=wan_root)
    plan = runtime.runtime_plan()
    runtime_cwd = runtime._runtime_cwd(tmp_path / "run")

    assert plan["wan_1_3b_exists"] is True
    assert (runtime_cwd / "wan_models" / "Wan2.1-T2V-1.3B").resolve() == hfd_model.resolve()


def test_rolling_forcing_runtime_uses_subclass_model_and_flat_checkpoint_layout(monkeypatch, tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "ckpts"
    checkpoint = checkpoint_root / "TencentARC--RollingForcing" / "checkpoints" / "rolling_forcing_dmd.pt"
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
    assert plan["license"] == ("RollingForcing academic-only; commercial and production use prohibited")


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
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["checkpoints"] == {
        "primary": str(hfd_root / "TencentARC--RollingForcing" / "checkpoints" / "rolling_forcing_dmd.pt"),
        "base": str(hfd_root / "Wan-AI--Wan2.1-T2V-1.3B"),
    }
    assert payload["load_kwargs"]["wan_models_root"] == str(hfd_root)
    assert "golden retriever" in payload["prompt"].lower()


@pytest.mark.unit
def test_rolling_forcing_inference_preserves_video_encoding_options(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import torch
    import torchvision.io as tv_io

    from worldfoundry.core.media.codecs import video as video_io
    from worldfoundry.core.model_loading import checkpoints
    from worldfoundry.synthesis.visual_generation.rolling_forcing import inference

    original_writer = getattr(tv_io, "write_video", None)
    config = tmp_path / "config.yaml"
    config.write_text("num_frame_per_block: 3\n", encoding="utf-8")
    config.with_name("default_config.yaml").write_text("{}\n", encoding="utf-8")
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("camera move\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rolling-forcing",
            "--config_path",
            str(config),
            "--checkpoint_path",
            str(tmp_path / "model.pt"),
            "--data_path",
            str(prompts),
            "--output_folder",
            str(tmp_path),
            "--num_output_frames",
            "3",
            "--use_ema",
        ],
    )
    monkeypatch.setattr(inference, "_distributed_device", lambda _seed: (torch.device("cpu"), 0, 1))
    checkpoint = {"generator_ema": {"weight": torch.zeros(1)}}
    monkeypatch.setattr(checkpoints, "load_weights_only", lambda *_args, **_kwargs: checkpoint)
    video = torch.linspace(-0.1, 1.1, 2 * 3 * 16 * 32).reshape(1, 2, 3, 16, 32)

    def sample(**kwargs: Any) -> tuple[torch.Tensor, None]:
        assert kwargs["noise"].shape == (1, 3, 16, 60, 104)
        assert kwargs["noise"].dtype == torch.bfloat16
        assert kwargs["text_prompts"] == ["camera move"]
        return video.clone(), None

    pipeline = SimpleNamespace(
        generator=SimpleNamespace(load_state_dict=lambda *_args, **_kwargs: None),
        vae=SimpleNamespace(model=SimpleNamespace(clear_cache=lambda: None)),
        inference_rolling_forcing=sample,
    )
    pipeline.to = lambda **_kwargs: pipeline
    pipeline.eval = lambda: pipeline
    module_name = (
        "worldfoundry.synthesis.visual_generation.forcing.causal_forcing_runtime.long_video.pipeline."
        "rolling_forcing_inference"
    )
    module = ModuleType(module_name)
    monkeypatch.setattr(module, "CausalInferencePipeline", lambda *_args, **_kwargs: pipeline, raising=False)
    monkeypatch.setitem(sys.modules, module_name, module)
    captured: dict[str, Any] = {}

    def write_video(frames: torch.Tensor, path: Path, **kwargs: Any) -> None:
        captured.update(frames=frames, path=path, options=kwargs)

    monkeypatch.setattr(video_io, "write_video", write_video)
    with torch.enable_grad():
        inference.main()

    expected = (255.0 * video[0].permute(0, 2, 3, 1)).clamp(0, 255).to(torch.uint8)
    torch.testing.assert_close(captured["frames"], expected, rtol=0, atol=0)
    assert captured["path"] == tmp_path / "camera_move-0.mp4"
    assert captured["options"] == {"fps": 16, "codec": "libx264", "macro_block_size": 16}
    assert getattr(tv_io, "write_video", None) is original_writer


@pytest.mark.unit
@pytest.mark.parametrize(
    "model_id, pipeline_name",
    [("self-forcing", "NativeSelfForcingPipeline"), ("causal-forcing", "NativeCausalForcingPipeline")],
)
def test_native_forcing_routes_do_not_use_subprocess_bridge(model_id: str, pipeline_name: str) -> None:
    model = discover_model_registry().get(model_id)

    assert model.pipeline_target == f"worldfoundry.pipelines.forcing.pipeline_native_forcing:{pipeline_name}"


def test_model_registry_contains_rolling_forcing() -> None:
    rolling_model = discover_model_registry().get("rolling-forcing")

    assert rolling_model.has_loader is True
    assert rolling_model.has_infer is True
    assert rolling_model.pipeline_target == ("worldfoundry.pipelines.forcing.pipeline_forcing:RollingForcingPipeline")
