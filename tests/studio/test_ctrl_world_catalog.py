from worldfoundry.runtime.inference_catalog import get_model_inference_spec
from worldfoundry.studio.inference.catalog import find_entry


def test_ctrl_world_catalog_executes_local_checkpoint_by_default(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path / "ckpts"))

    entry = find_entry("ctrl-world")

    checkpoint = tmp_path / "ckpts" / "yjguo--Ctrl-World" / "checkpoint-10000.pt"
    assert entry.display_name == "Ctrl-World"
    assert entry.supports_from_pretrained is True
    assert entry.default_backend == "from_pretrained"
    assert entry.default_model_ref == str(checkpoint)
    assert entry.default_input_path.endswith("test_vla_case1/aloha/observation_images_cam_high.png")
    assert entry.default_load_kwargs == {
        "model_id": "ctrl-world",
        "checkpoint_path": str(checkpoint),
        "base_model_dir": str(tmp_path / "ckpts" / "stabilityai--stable-video-diffusion-img2vid"),
        "clip_model_dir": str(tmp_path / "ckpts" / "openai--clip-vit-base-patch32"),
    }
    assert entry.default_call_kwargs["input_mode"] == "explicit-three-view"
    assert entry.default_call_kwargs["action_mode"] == "absolute-pose"
    assert "initial_pose" not in entry.default_call_kwargs
    assert len(entry.default_call_kwargs["input_views"]) == 3
    assert all("test_vla_case1/aloha/" in path for path in entry.default_call_kwargs["input_views"])
    assert entry.default_call_kwargs["num_frames"] == 5
    assert entry.default_call_kwargs["plan_only"] is False


def test_ctrl_world_inference_contract_uses_portable_paths() -> None:
    spec = get_model_inference_spec("ctrl-world")

    assert spec is not None
    variant = spec.variants[0]
    task = spec.tasks[0]
    assert variant.status == "requires_local_checkpoints"
    assert all(checkpoint.uri.startswith("${WORLDFOUNDRY_CKPT_DIR}/") for checkpoint in variant.checkpoints)
    assert all(
        str(variant.load_kwargs[key]).startswith("${WORLDFOUNDRY_CKPT_DIR}/")
        for key in ("checkpoint_path", "base_model_dir", "clip_model_dir")
    )
    assert task.inputs[0].default.startswith("${WORLDFOUNDRY_REPO_ROOT}/")
    assert task.default_call_kwargs["input_mode"] == "explicit-three-view"
    pose_field = next(field for field in task.inputs if field.field_id == "initial_pose")
    assert pose_field.required is False
    assert len(task.default_call_kwargs["input_views"]) == 3
    assert task.default_call_kwargs["plan_only"] is False
