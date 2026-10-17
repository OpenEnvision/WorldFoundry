from __future__ import annotations

import sys
from types import SimpleNamespace


def test_motionctrl_openclip_patch_avoids_hf_hub_download(monkeypatch):
    from worldfoundry.synthesis.visual_generation.motionctrl import worldfoundry_runtime

    calls = []

    def original_create(model_name, *args, **kwargs):
        calls.append((model_name, args, kwargs))
        return "model", "train", "val"

    fake_open_clip = SimpleNamespace(create_model_and_transforms=original_create)
    monkeypatch.setitem(sys.modules, "open_clip", fake_open_clip)

    worldfoundry_runtime._patch_motionctrl_openclip_download()

    assert getattr(fake_open_clip.create_model_and_transforms, worldfoundry_runtime._OPENCLIP_PATCH_MARKER)
    assert fake_open_clip.create_model_and_transforms(
        worldfoundry_runtime.MOTIONCTRL_OPENCLIP_HF_ID,
        device="cpu",
    ) == ("model", "train", "val")
    assert calls == [
        (
            worldfoundry_runtime.MOTIONCTRL_OPENCLIP_ARCH,
            (),
            {
                "device": "cpu",
                "pretrained": None,
                "load_weights": False,
                "pretrained_text": False,
                "pretrained_image": False,
            },
        )
    ]


def test_motionctrl_negative_seed_uses_official_default():
    from worldfoundry.synthesis.visual_generation.motionctrl import worldfoundry_runtime

    assert worldfoundry_runtime._motionctrl_seed(-1) == 20230211
    assert worldfoundry_runtime._motionctrl_seed("-1") == 20230211
    assert worldfoundry_runtime._motionctrl_seed(None) == 20230211
    assert worldfoundry_runtime._motionctrl_seed(7) == 7
