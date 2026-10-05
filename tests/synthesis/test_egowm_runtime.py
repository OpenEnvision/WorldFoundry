from __future__ import annotations

import json
from pathlib import Path

import pytest

from worldfoundry.pipelines.world_model.pipeline_runtime_manifest import EgoWMPipeline
from worldfoundry.synthesis.visual_generation import runtime_manifest
from worldfoundry.synthesis.visual_generation.egowm import infer
from worldfoundry.synthesis.visual_generation.egowm import worldfoundry_runtime as runtime


def _stage_runtime(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "sources" / "egowm"
    (source / "models").mkdir(parents=True)
    (source / "models" / "svd_wrapper.py").write_text("", encoding="utf-8")
    image = source / "data" / "cmu_clicks" / "realw_0.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"fixture")

    checkpoint = tmp_path / "ckpts" / "anuragba--egowm" / "svd_25dof_nav.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fixture")

    base = tmp_path / "ckpts" / "stabilityai--stable-video-diffusion-img2vid"
    for relative in (
        "model_index.json",
        "unet/config.json",
        "unet/diffusion_pytorch_model.fp16.safetensors",
        "vae/diffusion_pytorch_model.fp16.safetensors",
        "image_encoder/model.fp16.safetensors",
    ):
        path = base / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    return source, checkpoint, base


def test_runtime_spec_uses_dynamic_egowm_adapter() -> None:
    spec = runtime_manifest.runtime_spec("egowm")

    assert spec.runtime_module == "worldfoundry.synthesis.visual_generation.egowm.worldfoundry_runtime"
    assert spec.runtime_root_func == "runtime_root"
    assert spec.blocked_reason == ""


def test_missing_requirements_is_empty_when_runtime_is_staged(tmp_path, monkeypatch) -> None:
    source, checkpoint, base = _stage_runtime(tmp_path)
    conditions = tmp_path / "conditions.json"
    conditions.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(runtime.importlib.util, "find_spec", lambda _name: object())

    missing = runtime.missing_requirements(
        options={"checkpoint_path": str(checkpoint), "base_model_dir": str(base),
                 "image_path": str(source / "data/cmu_clicks/realw_0.png"), "conditions_path": str(conditions)},
        runtime_root=source,
        entrypoint=runtime.OFFICIAL_ENTRYPOINT,
        profile=None,
    )

    assert missing == []


def test_25dof_requires_paired_conditions(tmp_path, monkeypatch) -> None:
    source, checkpoint, base = _stage_runtime(tmp_path)
    monkeypatch.setattr(runtime.importlib.util, "find_spec", lambda _name: object())
    kwargs = dict(runtime_root=source, entrypoint=runtime.OFFICIAL_ENTRYPOINT, profile=None)
    options = {"checkpoint_path": str(checkpoint), "base_model_dir": str(base),
               "image_path": str(source / "data/cmu_clicks/realw_0.png")}

    missing = runtime.missing_requirements(options=options, **kwargs)
    assert any(item["kind"] == "condition" for item in missing)
    assert not any(item["kind"] == "condition" for item in runtime.missing_requirements(
        options={**options, "smoke_synthetic": True}, **kwargs
    ))
    # A custom checkpoint path needs an explicit 3-DoF variant, never filename guessing.
    arbitrary_name = checkpoint.with_name("renamed_weights.pth")
    checkpoint.rename(arbitrary_name)
    assert not any(item["kind"] == "condition" for item in runtime.missing_requirements(
        options={**options, "checkpoint_path": str(arbitrary_name), "variant": "3dof"}, **kwargs
    ))
    assert any(item["kind"] == "condition" for item in runtime.missing_requirements(
        options={**options, "checkpoint_path": str(arbitrary_name)}, **kwargs
    ))
    conditions = tmp_path / "conditions.json"
    conditions.write_text("{}", encoding="utf-8")
    conflicts = runtime.missing_requirements(
        options={**options, "checkpoint_path": str(arbitrary_name), "conditions_path": str(conditions),
                 "action_scale": 0.1, "smoke_synthetic": True}, **kwargs
    )
    assert {item["path"] for item in conflicts if item["kind"] == "condition"} == {
        "action_scale", "smoke_synthetic"
    }


def test_build_command_expands_paths_and_call_options(tmp_path, monkeypatch) -> None:
    source, checkpoint, base = _stage_runtime(tmp_path)
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path / "ckpts"))
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    plan_path = output_dir / "egowm.json"
    plan_path.write_text(
        json.dumps({"extra": {"num_frames": 4, "num_inference_steps": 3, "seed": 7}}),
        encoding="utf-8",
    )

    command = runtime.build_command(
        {
            "python": "python",
            "entrypoint": str(runtime.OFFICIAL_ENTRYPOINT),
            "runtime_root": str(source),
            "output_path": str(output_dir / "egowm.mp4"),
            "plan_path": str(plan_path),
            "device": "cuda",
            "options": {
                "checkpoint_path": "${WORLDFOUNDRY_CKPT_DIR}/anuragba--egowm/svd_25dof_nav.pth",
                "base_model_dir": "${WORLDFOUNDRY_CKPT_DIR}/stabilityai--stable-video-diffusion-img2vid",
                "image_path": str(source / "data/cmu_clicks/realw_0.png"),
            },
        }
    )

    assert command[command.index("--checkpoint-path") + 1] == str(checkpoint)
    assert command[command.index("--base-model-dir") + 1] == str(base)
    assert command[command.index("--input-image") + 1] == str(source / "data/cmu_clicks/realw_0.png")
    assert command[command.index("--num-frames") + 1] == "4"
    assert command[command.index("--num-inference-steps") + 1] == "3"
    assert command[command.index("--seed") + 1] == "7"
    assert "${WORLDFOUNDRY_" not in " ".join(command)


def test_build_command_passes_physical_conditions(tmp_path) -> None:
    source, checkpoint, base = _stage_runtime(tmp_path)
    conditions = tmp_path / "conditions.json"
    conditions.write_text("{}", encoding="utf-8")
    command = runtime.build_command({
        "python": "python", "entrypoint": runtime.OFFICIAL_ENTRYPOINT, "runtime_root": source,
        "output_path": tmp_path / "out.mp4", "device": "cuda",
        "options": {"checkpoint_path": checkpoint, "base_model_dir": base,
                    "image_path": source / "data/cmu_clicks/realw_0.png", "conditions_path": conditions},
    })
    assert command[command.index("--conditions-path") + 1] == str(conditions)
    assert command[command.index("--variant") + 1] == "25dof"
    assert "--smoke-synthetic" not in command


def test_25dof_rejects_missing_image_even_with_conditions(tmp_path, monkeypatch) -> None:
    source, checkpoint, base = _stage_runtime(tmp_path)
    conditions = tmp_path / "conditions.json"
    conditions.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(runtime.importlib.util, "find_spec", lambda _name: object())
    options = {"checkpoint_path": str(checkpoint), "base_model_dir": str(base),
               "conditions_path": str(conditions)}
    missing = runtime.missing_requirements(options=options, runtime_root=source,
                                           entrypoint=runtime.OFFICIAL_ENTRYPOINT, profile=None)
    assert any(item["path"] == "image_path" for item in missing)
    with pytest.raises(ValueError, match="explicit image_path"):
        runtime.build_command({"python": "python", "entrypoint": runtime.OFFICIAL_ENTRYPOINT,
                               "runtime_root": source, "output_path": tmp_path / "out.mp4",
                               "device": "cuda", "options": options})
    command = runtime.build_command({"python": "python", "entrypoint": runtime.OFFICIAL_ENTRYPOINT,
                                     "runtime_root": source, "output_path": tmp_path / "out.mp4",
                                     "device": "cuda", "options": {"checkpoint_path": checkpoint,
                                                                    "base_model_dir": base, "variant": "3dof"}})
    assert command[command.index("--input-image") + 1] == str(source / "data/cmu_clicks/realw_0.png")


def test_25dof_condition_shape_and_values(tmp_path) -> None:
    stats = json.loads((Path(infer.__file__).parent / "nav_25dof_stats.json").read_text())
    conditions = tmp_path / "conditions.json"
    conditions.write_text(json.dumps({
        "physical_initial_state": stats["state_stats"]["min"],
        "physical_actions": [stats["action_stats_6"]["max"]],
    }), encoding="utf-8")
    initial_state, actions = infer.load_25dof_conditions(conditions, 1)
    assert initial_state == pytest.approx([-1.0] * 25)
    assert actions[0] == pytest.approx([1.0] * 25)
    with pytest.raises(ValueError, match="exactly 2"):
        infer.load_25dof_conditions(conditions, 2)


def test_25dof_rejects_action_scale_with_paired_conditions(tmp_path) -> None:
    args = infer._parser().parse_args([
        "--source-dir", str(tmp_path), "--checkpoint-path", str(tmp_path),
        "--base-model-dir", str(tmp_path), "--input-image", str(tmp_path),
        "--output-path", str(tmp_path / "out.mp4"),
        "--conditions-path", str(tmp_path), "--action-scale", "0.1",
    ])
    with pytest.raises(ValueError, match="cannot be combined"):
        infer._validate_args(args)


def test_default_25dof_rejects_missing_conditions_before_asset_load(tmp_path) -> None:
    args = infer._parser().parse_args([
        "--source-dir", str(tmp_path / "missing-source"),
        "--checkpoint-path", str(tmp_path / "missing-checkpoint"),
        "--base-model-dir", str(tmp_path / "missing-base"),
        "--input-image", str(tmp_path / "missing-image"),
        "--output-path", str(tmp_path / "out.mp4"),
    ])
    with pytest.raises(ValueError, match="requires --conditions-path"):
        infer._validate_args(args)


def test_invalid_conditions_fail_before_checkpoint_load(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.pth"
    image = tmp_path / "input.png"
    conditions = tmp_path / "conditions.json"
    checkpoint.write_bytes(b"invalid checkpoint must never be read")
    image.write_bytes(b"invalid image must never be read")
    conditions.write_text(json.dumps({"physical_initial_state": [0] * 24, "physical_actions": [[0] * 25]}))
    args = infer._parser().parse_args([
        "--source-dir", str(tmp_path), "--checkpoint-path", str(checkpoint),
        "--base-model-dir", str(tmp_path), "--input-image", str(image),
        "--conditions-path", str(conditions), "--num-frames", "1",
        "--output-path", str(tmp_path / "out.mp4"),
    ])
    with pytest.raises(ValueError, match="physical_initial_state must contain exactly 25"):
        infer.run(args)


def test_explicit_variant_matches_checkpoint_signature() -> None:
    class Embedding:
        shape = (1280, 96)

    assert infer.checkpoint_action_dims({"add_action_embedding.linear_1.weight": Embedding()}) == 3
    class StateEmbedding:
        shape = (1280, 800)

    assert infer.checkpoint_action_dims({
        "add_action_embedding.linear_1.weight": StateEmbedding(),
        "add_state_embedding.linear_1.weight": object(),
    }) == 25
    with pytest.raises(ValueError, match="conflicts with checkpoint"):
        infer._validate_checkpoint_variant(3, "25dof")
    with pytest.raises(ValueError, match="conflicts with checkpoint"):
        infer._validate_checkpoint_variant(25, "3dof")


def test_25dof_stats_reject_invalid_range(tmp_path, monkeypatch) -> None:
    stats = json.loads((Path(infer.__file__).parent / "nav_25dof_stats.json").read_text())
    stats["action_stats_6"]["max"][0] = stats["action_stats_6"]["min"][0]
    (tmp_path / "nav_25dof_stats.json").write_text(json.dumps(stats), encoding="utf-8")
    monkeypatch.setattr(infer, "__file__", str(tmp_path / "infer.py"))
    with pytest.raises(ValueError, match="max > min"):
        infer._load_25dof_stats()


def test_pipeline_promotes_image_path_for_runtime_gate(tmp_path) -> None:
    pipeline = object.__new__(EgoWMPipeline)

    promoted = pipeline._promote_call_options({"images": tmp_path / "input.png"})

    assert promoted["image_path"] == str(tmp_path / "input.png")


def test_launcher_rejects_invalid_video_geometry(tmp_path) -> None:
    args = infer._parser().parse_args(
        [
            "--source-dir",
            str(tmp_path),
            "--checkpoint-path",
            str(tmp_path),
            "--base-model-dir",
            str(tmp_path),
            "--input-image",
            str(tmp_path),
            "--output-path",
            str(tmp_path / "out.mp4"),
            "--height",
            "513",
        ]
    )

    try:
        infer._validate_args(args)
    except ValueError as exc:
        assert "divisible by 8" in str(exc)
    else:
        raise AssertionError("invalid geometry was accepted")
