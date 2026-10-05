from __future__ import annotations

from types import SimpleNamespace

import torch
import worldfoundry.studio.inference.catalog as catalog
import worldfoundry.synthesis.visual_generation.kling.recammaster_runtime.models.wan_model as wan_model

from worldfoundry.pipelines.kling.pipeline_recammaster import ReCamMasterPipeline
from worldfoundry.studio.inference.catalog import (
    _recammaster_default_load_kwargs,
    _recammaster_default_ref,
    find_entry,
)


def test_recammaster_catalog_resolves_owner_prefixed_checkpoint(tmp_path, monkeypatch) -> None:
    recammaster = tmp_path / "ckpts" / "KlingTeam--ReCamMaster-Wan2.1"
    recammaster.mkdir(parents=True)
    (recammaster / "step20000.ckpt").write_bytes(b"weights")
    wan = tmp_path / "ckpts" / "Wan-AI--Wan2.1-T2V-1.3B"
    wan.mkdir(parents=True)
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path / "ckpt"))
    monkeypatch.setenv("WORLDFOUNDRY_HFD_ROOT", str(tmp_path / "hfd"))

    assert _recammaster_default_ref() == str(recammaster)
    assert _recammaster_default_load_kwargs() == {
        "wan_model_path": str(wan),
        "recammaster_ckpt_path": str(recammaster),
    }


def test_recammaster_catalog_fallbacks_and_demo_inputs_are_portable(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path / "ckpt"))
    monkeypatch.setenv("WORLDFOUNDRY_HFD_ROOT", str(tmp_path / "hfd"))
    monkeypatch.setattr(catalog, "_cache_candidates", lambda *names: [])

    assert _recammaster_default_ref() == "KlingTeam/ReCamMaster-Wan2.1"
    assert find_entry("recammaster").default_call_kwargs["video_path"] == (
        "worldfoundry/data/test_cases/longcat_video/motorcycle.mp4"
    )
    assert "motorcycle" in find_entry("recammaster").default_prompt.lower()
    assert "highway bridge" in find_entry("recammaster").default_prompt.lower()
    assert find_entry("cameractrl").default_call_kwargs["trajectory_file"] == (
        "worldfoundry/data/test_cases/cameractrl/pose_files/0f47577ab3441480.txt"
    )


def test_recammaster_pipeline_forwards_smoke_inference_options() -> None:
    captured = {}

    class Synthesis:
        def predict(self, prompt, video, camera, **kwargs):
            captured.update(kwargs)
            return "video"

    pipeline = ReCamMasterPipeline(
        operator=SimpleNamespace(), synthesis_model=Synthesis(), device="cpu"
    )
    pipeline.process = lambda *args: ("source", "camera", "prompt")

    result = pipeline(
        camera_trajectory=[0, 0, 0, 0, 0],
        video_path="input.mp4",
        prompt="prompt",
        num_frames=9,
        num_inference_steps=2,
        cfg_scale=1.0,
        size=(256, 384),
    )

    assert result == "video"
    assert captured == {
        "num_frames": 9,
        "height": 256,
        "width": 384,
        "num_inference_steps": 2,
        "cfg_scale": 1.0,
    }


def test_recammaster_flash_attention_3_accepts_tuple_result(monkeypatch) -> None:
    def flash_attn_func(query, key, value):
        del key, value
        return query, torch.zeros(query.shape[:-1])

    monkeypatch.setattr(wan_model, "FLASH_ATTN_3_AVAILABLE", True)
    monkeypatch.setattr(
        wan_model.flash_attn_interface,
        "flash_attn_func",
        flash_attn_func,
    )
    query = torch.randn(1, 3, 8)

    output = wan_model.flash_attention(query, query, query, num_heads=2)

    assert torch.equal(output, query)
