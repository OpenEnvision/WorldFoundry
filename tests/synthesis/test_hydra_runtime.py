# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from worldfoundry.synthesis.visual_generation.hydra import worldfoundry_runtime as runtime_module

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("return_dict", [False, True])
def test_predict_launches_package_module_with_unchanged_inputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, return_dict: bool
) -> None:
    checkpoint, video, camera = (tmp_path / name for name in ("hydra.ckpt", "condition.mp4", "camera.json"))
    for path in (checkpoint, video, camera):
        path.touch()
    base = tmp_path / "wan"
    base.mkdir()
    output = tmp_path / "generated.mp4"
    captured = {}

    def execute(command: list[str], **kwargs: Any) -> dict[str, Any]:
        captured.update(command=command, **kwargs)
        return {"video": str(output), "status": "succeeded"}

    monkeypatch.setattr(runtime_module, "execute_in_tree", execute)
    runtime = runtime_module.HydraRuntime(
        checkpoint_path=checkpoint,
        base_model_path=base,
        python_executable=sys.executable,
        device="cuda:2",
        env={"HF_HUB_OFFLINE": "1"},
    )

    result = runtime.predict(
        prompt="  a moving camera  ", video_path=video, camera_json=camera, output_path=output, return_dict=return_dict
    )

    command = captured["command"]
    assert command[:3] == [
        sys.executable,
        "-m",
        "worldfoundry.synthesis.visual_generation.hydra.hydra_runtime.infer_hydra",
    ]
    args = dict(zip(command[3::2], command[4::2], strict=True))
    assert args == {
        "--cond_video": str(video),
        "--cond_json": str(camera),
        "--caption_txt": str(tmp_path / ".generated_hydra_inputs" / "caption.txt"),
        "--ckpt_path": str(checkpoint),
        "--base_dit_path": str(base / "diffusion_pytorch_model.safetensors"),
        "--base_text_encoder_path": str(base / "models_t5_umt5-xxl-enc-bf16.pth"),
        "--base_vae_path": str(base / "Wan2.1_VAE.pth"),
        "--output_path": str(output),
        "--device": "cuda:2",
        "--height": "480",
        "--width": "832",
        "--cond_frames": "77",
        "--fps": "15",
        "--num_inference_steps": "50",
        "--cfg_scale": "5.0",
        "--seed": "42",
    }
    assert Path(args["--caption_txt"]).read_text() == "a moving camera\n"
    assert captured["cwd"] == runtime.repo_root
    assert captured["env"] == {"HF_HUB_OFFLINE": "1"}
    assert "python_paths" not in captured
    assert result == ({"video": str(output), "status": "succeeded"} if return_dict else str(output))


def test_default_runtime_uses_the_prepared_pinned_snapshots(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from worldfoundry.studio.inference.catalog.models.video_generation import (
        _hydra_default_load_kwargs,
        _hydra_default_ref,
    )

    hub = tmp_path / "hub"
    hydra = hub / "models--H-EmbodVis--HyDRA/snapshots/c2b83e1bc452974a9e599b026e0e09f084a0165f"
    wan = hub / "models--Wan-AI--Wan2.1-T2V-1.3B/snapshots/37ec512624d61f7aa208f7ea8140a131f93afc9a"
    files = [
        hydra / "hydra.ckpt",
        wan / "diffusion_pytorch_model.safetensors",
        wan / "models_t5_umt5-xxl-enc-bf16.pth",
        wan / "Wan2.1_VAE.pth",
        wan / "google/umt5-xxl/tokenizer.json",
        wan / "google/umt5-xxl/tokenizer_config.json",
        wan / "google/umt5-xxl/special_tokens_map.json",
        wan / "google/umt5-xxl/spiece.model",
    ]
    for file in files:
        file.parent.mkdir(parents=True, exist_ok=True)
        file.touch()
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    monkeypatch.setenv("WORLDFOUNDRY_HFD_ROOT", str(tmp_path / "hfd"))
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path / "checkpoints"))
    monkeypatch.delenv("WORLDFOUNDRY_HYDRA_CHECKPOINT", raising=False)
    monkeypatch.delenv("WORLDFOUNDRY_WAN21_1P3B_ROOT", raising=False)

    runtime = runtime_module.HydraRuntime(device="cpu")

    assert runtime.checkpoint_path == hydra / "hydra.ckpt"
    assert runtime.base_model_path == wan
    assert _hydra_default_ref() == str(hydra / "hydra.ckpt")
    assert _hydra_default_load_kwargs() == {"base_model_path": str(wan)}


def test_build_pipeline_uses_tokenizer_under_the_snapshot_for_symlinked_t5(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from worldfoundry.synthesis.visual_generation.hydra.hydra_runtime import infer_hydra

    tokenizer = tmp_path / "snapshots/revision/google/umt5-xxl"
    tokenizer.mkdir(parents=True)
    blob = tmp_path / "blobs/text-weights"
    blob.parent.mkdir()
    blob.touch()
    text_weights = tokenizer.parents[1] / "models_t5_umt5-xxl-enc-bf16.pth"
    text_weights.symlink_to(blob)
    calls = {}

    class Pipeline:
        @classmethod
        def from_components(cls, **kwargs):
            calls.update(kwargs)
            return cls()

        def load_hydra_checkpoint(self, *args, **kwargs):
            return self

        def to(self, *args, **kwargs):
            return self

    monkeypatch.setattr(infer_hydra, "HyDRAPipeline", Pipeline)
    monkeypatch.setattr(infer_hydra, "load_model", lambda *args, **kwargs: object())
    infer_hydra.build_pipeline("cpu", "hydra.ckpt", "dit.safetensors", str(text_weights), "vae.pth", True)

    assert calls["tokenizer_path"] == str(tokenizer)
