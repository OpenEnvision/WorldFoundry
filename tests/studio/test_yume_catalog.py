from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from worldfoundry.pipelines.yume.pipeline_yume import YumePipeline, _normalise_request
from worldfoundry.studio.inference.catalog import find_entry


def test_yume_catalog_resolves_complete_local_checkpoint() -> None:
    entry = find_entry("yume")
    checkpoint = Path(entry.default_model_ref)

    assert checkpoint.is_dir()
    assert checkpoint.name == "stdstu123--Yume-I2V-540P"
    assert (checkpoint / "models_t5_umt5-xxl-enc-bf16.pth").is_file()
    assert (checkpoint / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth").is_file()
    assert (checkpoint / "Wan2.1_VAE.pth").is_file()
    assert len(tuple((checkpoint / "Yume-Dit").glob("diffusion_pytorch_model-*.safetensors"))) == 7


def test_yume_catalog_exposes_runnable_defaults() -> None:
    entry = find_entry("yume")

    assert entry.default_call_kwargs["size"] == "544*960"
    assert entry.default_call_kwargs["interactions"] == ["forward", "camera_l"]
    assert entry.default_call_kwargs["sampling_method"] == "ode"
    assert entry.default_load_kwargs["fsdp"] is False


def test_yume_request_normalises_single_t2v_interaction() -> None:
    request = _normalise_request(
        "forward",
        None,
        None,
        task_type="t2v",
        size="544*960",
        num_euler_timesteps=1,
        sampling_method="ode",
        images=None,
        videos=None,
    )

    assert request == (["forward"], [100.0], [4.0], "t2v", "544*960", 1, "ode")


def test_yume_unified_invocation_exports_t2v_without_conditioning_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class RecordingPipeline(YumePipeline):
        def __init__(self) -> None:
            self.call: dict[str, object] = {}

        def __call__(self, **kwargs: object) -> list[object]:
            self.call = kwargs
            return [object(), object()]

    def fake_export(frames: list[object], path: str, fps: int) -> None:
        Path(path).write_bytes(b"video")

    monkeypatch.setattr("diffusers.utils.export_to_video", fake_export)
    pipeline = RecordingPipeline()
    output_path = tmp_path / "sample.mp4"
    invocation = SimpleNamespace(
        prompt="mountain road",
        image="unused-image.jpg",
        video=None,
        interactions="forward",
        output_path=output_path,
        pipeline_kwargs={
            "task_type": "t2v",
            "interaction_speeds": [100],
            "interaction_distances": [4],
            "size": "544*960",
            "seed": 0,
            "num_euler_timesteps": 1,
            "sampling_method": "ode",
        },
    )

    result = pipeline.run_pipeline_invocation(invocation)

    assert pipeline.call["images"] is None
    assert pipeline.call["videos"] is None
    assert pipeline.call["interactions"] == "forward"
    assert output_path.is_file()
    assert result["status"] == "succeeded"


def test_yume_dit_casts_to_bfloat16_before_device_sharding(monkeypatch: pytest.MonkeyPatch) -> None:
    from worldfoundry.synthesis.visual_generation.yume.yume_runtime.yume import image2video

    events: list[tuple[str, object]] = []

    class FakeModel:
        patch_embedding = object()

        def eval(self) -> "FakeModel":
            return self

        def requires_grad_(self, value: bool) -> "FakeModel":
            return self

        def to(self, *args: object, **kwargs: object) -> "FakeModel":
            events.append(("to", kwargs.get("dtype") if kwargs else args))
            return self

    fake_model = FakeModel()
    monkeypatch.setattr(image2video.WanModel, "from_config", lambda config: fake_model)
    monkeypatch.setattr(image2video, "load_yume_checkpoint", lambda model, checkpoint_dir: model)
    monkeypatch.setattr(image2video, "upsample_conv3d_weights", lambda model, size: object())
    monkeypatch.setattr(
        image2video,
        "shard_model",
        lambda model, device_id: events.append(("shard", device_id)) or model,
    )
    config = SimpleNamespace(
        patch_size=(1, 2, 2),
        vae_stride=(4, 8, 8),
        sample_neg_prompt="",
        param_dtype=torch.bfloat16,
    )

    image2video.YumeI2V(config, "unused", device_id=0, dit_fsdp=True)

    assert events[0] == ("to", torch.bfloat16)
    assert events[1] == ("shard", 0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"interactions": [], "task_type": "t2v"}, "at least one interaction"),
        ({"interactions": ["forward"], "task_type": "i2v"}, "requires an image"),
        ({"interactions": ["forward"], "task_type": "t2v", "size": "352*640"}, "Unsupported"),
        ({"interactions": ["forward"], "task_type": "t2v", "num_euler_timesteps": 0}, "positive integer"),
        ({"interactions": ["forward"], "task_type": "t2v", "sampling_method": "ddim"}, "ode, sde"),
    ),
)
def test_yume_request_rejects_invalid_geometry(kwargs: dict[str, object], message: str) -> None:
    request = {
        "interactions": ["forward"],
        "interaction_speeds": None,
        "interaction_distances": None,
        "task_type": "t2v",
        "size": "544*960",
        "num_euler_timesteps": 1,
        "sampling_method": "ode",
        "images": None,
        "videos": None,
    }
    request.update(kwargs)

    with pytest.raises(ValueError, match=message):
        _normalise_request(**request)
