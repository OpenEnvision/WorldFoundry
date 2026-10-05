from __future__ import annotations

import types

import pytest

# worldfoundry.pipelines.wan.pipeline_wan_2p2 imports ftfy (optional
# dependency) at module load time; skip in environments without it.
pytest.importorskip("ftfy")

import worldfoundry.pipelines.wan.pipeline_wan_2p2 as wan22_module

from worldfoundry.pipelines.wan.pipeline_wan_2p2 import Wan2p2Pipeline


def _fake_config_maps():
    cfg = types.SimpleNamespace(
        frame_num=81,
        sample_steps=8,
        sample_shift=5.0,
        sample_guide_scale=4.0,
    )
    return {"ti2v-5B": cfg}, {"ti2v-5B": {"704*1280", "1280*704"}}


def test_wan22_loader_uses_pipeline_signature(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class FakeWan2p2Synthesis:
        @classmethod
        def from_pretrained(cls, **kwargs):
            calls.append(kwargs)
            return {"loaded": True}

    monkeypatch.setattr(wan22_module, "load_wan2p2_config_maps", _fake_config_maps)
    monkeypatch.setattr(wan22_module, "Wan2p2Synthesis", FakeWan2p2Synthesis)
    monkeypatch.setattr(wan22_module, "VisualFrameMemory", lambda model_id: object())

    result = Wan2p2Pipeline.from_pretrained(
        {
            "model_path": "cache/hfd/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/rev",
            "mode": "ti2v-5B",
            "rank": 3,
            "t5_cpu": True,
            "convert_model_dtype": True,
        },
        device="cuda:2",
    )

    assert result.synthesis_model == {"loaded": True}
    assert calls == [
        {
            "ckpt_dir": "cache/hfd/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/rev",
            "mode": "ti2v-5B",
            "device": 2,
            "rank": 3,
            "t5_fsdp": False,
            "dit_fsdp": False,
            "ulysses_size": 1,
            "t5_cpu": True,
            "convert_model_dtype": True,
        }
    ]


def test_wan22_call_passes_reference_image_as_images(monkeypatch) -> None:
    monkeypatch.setattr(wan22_module, "load_wan2p2_config_maps", _fake_config_maps)
    calls: list[dict[str, object]] = []

    class FakeOperator:
        def get_interaction(self, prompt):
            self.prompt = prompt

        def process_perception(self, *, input_path=None, **kwargs):
            calls.append({"input_path": input_path})
            return {"input_image": input_path}

        def process_interaction(self, **kwargs):
            calls.append(kwargs)
            return {"processed_prompt": self.prompt}

    class FakeSynthesis:
        def predict(self, **kwargs):
            calls.append(kwargs)
            return "video"

    pipeline = Wan2p2Pipeline(
        operator=FakeOperator(),
        synthesis_model=FakeSynthesis(),
        mode="ti2v-5B",
        base_seed=42,
    )
    result = pipeline(
        prompt="a white cat sits on a surfboard",
        images="ref.png",
        size="704*1280",
    )

    assert result == "video"
    assert calls[0] == {"input_path": "ref.png"}
    assert calls[1]["images"] == "ref.png"
    assert calls[2]["processed_inputs"]["image"] == "ref.png"
    assert calls[2]["size"] == "704*1280"
