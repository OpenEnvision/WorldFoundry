from __future__ import annotations

import json
from pathlib import Path

from worldfoundry.pipelines.world_model.pipeline_runtime_manifest import HMAPipeline
from worldfoundry.synthesis.visual_generation import runtime_manifest
from worldfoundry.synthesis.visual_generation.hma import infer
from worldfoundry.synthesis.visual_generation.hma import worldfoundry_runtime as runtime


def _stage_runtime(tmp_path: Path) -> tuple[Path, Path]:
    checkpoint = tmp_path / "ckpts" / "liruiw--hma-base-cont"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text(json.dumps({"Diffusion": True}), encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"fixture")

    base = tmp_path / "ckpts" / "stabilityai--stable-video-diffusion-img2vid"
    for relative in ("vae/config.json", "vae/diffusion_pytorch_model.fp16.safetensors"):
        path = base / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    return checkpoint, base


def test_runtime_spec_uses_the_in_tree_hma_adapter() -> None:
    spec = runtime_manifest.runtime_spec("hma")
    root, entrypoint, _ = runtime_manifest.resolve_runtime_manifest(spec)

    assert spec.runtime_module == "worldfoundry.synthesis.visual_generation.hma.worldfoundry_runtime"
    assert spec.blocked_reason == ""
    assert root == runtime.RUNTIME_DIR
    assert entrypoint == runtime.OFFICIAL_ENTRYPOINT
    assert runtime.DEFAULT_PROMPT_IMAGE.is_file()


def test_missing_requirements_is_empty_when_checkpoints_are_staged(tmp_path, monkeypatch) -> None:
    checkpoint, base = _stage_runtime(tmp_path)
    monkeypatch.setattr(runtime.importlib.util, "find_spec", lambda _name: object())

    missing = runtime.missing_requirements(
        options={"checkpoint_dir": str(checkpoint), "base_model_dir": str(base)},
        runtime_root=runtime.RUNTIME_DIR,
        entrypoint=runtime.OFFICIAL_ENTRYPOINT,
        profile=None,
    )

    assert missing == []


def test_build_command_launches_the_package_module(tmp_path, monkeypatch) -> None:
    checkpoint, base = _stage_runtime(tmp_path)
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path / "ckpts"))
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    plan_path = output_dir / "hma.json"
    plan_path.write_text(
        json.dumps({"extra": {"generated_frames": 3, "maskgit_steps": 1, "direction": "left", "seed": 7}}),
        encoding="utf-8",
    )

    command = runtime.build_command(
        {
            "python": "python",
            "entrypoint": str(runtime.OFFICIAL_ENTRYPOINT),
            "runtime_root": str(runtime.RUNTIME_DIR),
            "output_path": str(output_dir / "hma.mp4"),
            "plan_path": str(plan_path),
            "device": "cuda",
            "options": {
                "checkpoint_dir": "${WORLDFOUNDRY_CKPT_DIR}/liruiw--hma-base-cont",
                "base_model_dir": "${WORLDFOUNDRY_CKPT_DIR}/stabilityai--stable-video-diffusion-img2vid",
            },
        }
    )

    assert command[:3] == ["python", "-m", "worldfoundry.synthesis.visual_generation.hma.infer"]
    assert "--source-dir" not in command
    assert command[command.index("--checkpoint-dir") + 1] == str(checkpoint)
    assert command[command.index("--base-model-dir") + 1] == str(base)
    assert command[command.index("--input-image") + 1] == str(runtime.DEFAULT_PROMPT_IMAGE)
    assert command[command.index("--generated-frames") + 1] == "3"
    assert command[command.index("--maskgit-steps") + 1] == "1"
    assert command[command.index("--direction") + 1] == "left"
    assert command[command.index("--seed") + 1] == "7"
    assert "${WORLDFOUNDRY_" not in " ".join(command)


def test_pipeline_promotes_image_path_for_runtime_gate(tmp_path) -> None:
    pipeline = object.__new__(HMAPipeline)

    promoted = pipeline._promote_call_options({"images": tmp_path / "input.png"})

    assert promoted["image_path"] == str(tmp_path / "input.png")


def test_launcher_rejects_cpu_execution(tmp_path) -> None:
    args = infer._parser().parse_args(
        [
            "--checkpoint-dir",
            str(tmp_path),
            "--base-model-dir",
            str(tmp_path),
            "--input-image",
            str(tmp_path),
            "--output-path",
            str(tmp_path / "out.mp4"),
            "--device",
            "cpu",
        ]
    )

    try:
        infer._validate_args(args)
    except ValueError as exc:
        assert "CUDA" in str(exc)
    else:
        raise AssertionError("CPU execution was accepted")
