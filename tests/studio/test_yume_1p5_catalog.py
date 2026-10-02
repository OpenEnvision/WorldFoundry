from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from worldfoundry.pipelines.yume.pipeline_yume_1p5 import Yume1p5Pipeline, _normalise_request
from worldfoundry.studio.inference.catalog import find_entry


def test_yume_1p5_catalog_resolves_complete_local_checkpoint() -> None:
    entry = find_entry("yume-1p5")
    checkpoint = Path(entry.default_model_ref)

    assert checkpoint.is_dir()
    assert checkpoint.name == "Yume-5B-720P"
    assert (checkpoint / "diffusion_pytorch_model.safetensors").is_file()
    assert (checkpoint / "models_t5_umt5-xxl-enc-bf16.pth").is_file()
    assert (checkpoint / "Wan2.2_VAE.pth").is_file()
    assert (checkpoint / "google" / "umt5-xxl" / "tokenizer_config.json").is_file()


def test_yume_1p5_catalog_exposes_accelerated_sampling_defaults() -> None:
    entry = find_entry("yume-1p5")

    assert entry.default_call_kwargs["size"] == "704*1280"
    assert entry.default_call_kwargs["num_euler_timesteps"] == 4
    assert entry.default_call_kwargs["interactions"] == ["forward", "left", "camera_r"]
    assert entry.default_load_kwargs["fsdp"] is False
    assert "num_euler_timesteps" in entry.call_params


def test_yume_1p5_request_normalises_single_t2v_interaction() -> None:
    request = _normalise_request(
        "forward",
        None,
        None,
        task_type="t2v",
        size="704*1280",
        num_euler_timesteps=1,
        images=None,
        videos=None,
    )

    assert request == (["forward"], [100.0], [4.0], "t2v", "704*1280", 1)


def test_yume_1p5_unified_invocation_exports_t2v_without_conditioning_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class RecordingPipeline(Yume1p5Pipeline):
        def __init__(self) -> None:
            self.call: dict[str, object] = {}

        def __call__(self, **kwargs: object) -> list[object]:
            self.call = kwargs
            return [object(), object()]

    exported: dict[str, object] = {}

    def fake_export(frames: list[object], path: str, fps: int) -> None:
        exported.update(frames=frames, path=path, fps=fps)
        Path(path).write_bytes(b"video")

    monkeypatch.setattr("diffusers.utils.export_to_video", fake_export)
    pipeline = RecordingPipeline()
    output_path = tmp_path / "sample.mp4"
    invocation = SimpleNamespace(
        prompt="coastal road",
        image="unused-image.jpg",
        video=None,
        interactions="forward",
        output_path=output_path,
        pipeline_kwargs={
            "task_type": "t2v",
            "interaction_speeds": [100],
            "interaction_distances": [4],
            "size": "704*1280",
            "seed": 0,
            "num_euler_timesteps": 1,
        },
    )

    result = pipeline.run_pipeline_invocation(invocation)

    assert pipeline.call["images"] is None
    assert pipeline.call["videos"] is None
    assert pipeline.call["interactions"] == "forward"
    assert exported["fps"] == 16
    assert output_path.is_file()
    assert result["status"] == "succeeded"
    assert result["artifact_path"] == str(output_path)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"interactions": [], "task_type": "t2v"}, "at least one interaction"),
        ({"interactions": ["forward"], "task_type": "i2v"}, "requires an image"),
        ({"interactions": ["forward"], "task_type": "v2v"}, "requires a non-empty video"),
        ({"interactions": ["forward"], "task_type": "t2v", "size": "352*640"}, "Unsupported"),
        ({"interactions": ["forward"], "task_type": "t2v", "num_euler_timesteps": 0}, "positive integer"),
    ),
)
def test_yume_1p5_request_rejects_invalid_geometry(kwargs: dict[str, object], message: str) -> None:
    request = {
        "interactions": ["forward"],
        "interaction_speeds": None,
        "interaction_distances": None,
        "task_type": "t2v",
        "size": "704*1280",
        "num_euler_timesteps": 1,
        "images": None,
        "videos": None,
    }
    request.update(kwargs)

    with pytest.raises(ValueError, match=message):
        _normalise_request(**request)
