from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from worldfoundry.studio.inference import execution as execution
from worldfoundry.studio.inference.catalog import find_entry
from worldfoundry.synthesis.visual_generation.lingbot_world.realtime import LingBotRealtimeSession


@pytest.mark.parametrize("name,raw", [("taew2_1.pth", False), ("taew2_1.safetensors", False), ("lighttaew2_1.pth", True)])
def test_decoder_receives_the_checkpoint_latent_convention(monkeypatch, name, raw):
    monkeypatch.delenv("WORLDFOUNDRY_LINGBOT_TAE_LATENT_FORMAT", raising=False)
    session = object.__new__(LingBotRealtimeSession)
    session.rank = 0
    session.autoregressive_index = 0
    session._decoder_needs_scaling = session._decoder_uses_raw_latents(Path(name))
    assert session._decoder_needs_scaling is raw
    session.core = SimpleNamespace(vae=SimpleNamespace(mean=torch.arange(16), std=torch.full((16,), 2.0)))
    captured = []

    class Decoder:
        def decode(self, z=None):
            if z is None:
                return None
            captured.append(z.clone())
            return torch.full((1, 9, 3, 2, 2), 0.5)

    session._decoder = Decoder()
    latent = torch.arange(192, dtype=torch.float32).reshape(16, 3, 2, 2) / 64
    original = latent.clone()
    frames = session._decode(latent)
    expected = latent.permute(1, 0, 2, 3)[None].to(torch.bfloat16)
    if raw:
        expected = expected * 2 + torch.arange(16, dtype=torch.bfloat16).view(1, 1, 16, 1, 1)
    torch.testing.assert_close(captured[0], expected, rtol=0, atol=0)
    torch.testing.assert_close(latent, original, rtol=0, atol=0)
    assert frames.shape == (9, 2, 2, 3)
    assert (frames == 128).all()


def test_renamed_decoder_requires_an_explicit_convention(monkeypatch):
    monkeypatch.delenv("WORLDFOUNDRY_LINGBOT_TAE_LATENT_FORMAT", raising=False)
    with pytest.raises(ValueError, match="Unknown TAE latent convention"):
        LingBotRealtimeSession._decoder_uses_raw_latents(Path("custom.pth"))
    for convention, expected in [("raw", True), ("normalized", False)]:
        monkeypatch.setenv("WORLDFOUNDRY_LINGBOT_TAE_LATENT_FORMAT", convention)
        assert LingBotRealtimeSession._decoder_uses_raw_latents(Path("custom.pth")) is expected


@pytest.mark.parametrize("vram,explicit,expected", [(80, {}, False), (48, {}, True), (80, {"dit_fsdp": True, "t5_fsdp": True}, True), (48, {"dit_fsdp": False, "t5_fsdp": False}, False)])
def test_fast_catalog_defaults_do_not_override_worker_vram_policy(tmp_path, monkeypatch, vram, explicit, expected):
    monkeypatch.setenv("WORLDFOUNDRY_STUDIO_TORCHRUN_LINGBOT_FAST", "1")
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("WORLDFOUNDRY_LINGBOT_FAST_USE_SP", "1")
    monkeypatch.delenv("WORLDFOUNDRY_LINGBOT_REPLICATED_MIN_VRAM_GB", raising=False)
    monkeypatch.setattr(execution, "_torch_module", lambda: None)
    monkeypatch.setattr(execution, "_torchrun_min_gpu_vram_gib", lambda: vram)
    entry = find_entry("lingbot-world")
    manager = execution.StudioManager(workspace_root=str(tmp_path))
    request = manager.prepare_inputs(
        entry=entry, prompt="scene", input_path="", image=None, video=None,
        last_frame=None, reference_files=None, interactions_text="", camera_view_text="",
        task_type="", intrinsics_text="", meta_path="", panorama_path="", scene_name="",
        fps=16, num_frames=9, call_kwargs_text='{"offload_model": false}',
        load_kwargs_text=json.dumps({"runtime_variant": "fast", **explicit}),
        model_ref="local-model", backend="from_pretrained", endpoint="", api_key="", device="cpu",
    )
    worker_request = manager._torchrun_worker_request(entry, request)
    assert worker_request.load_kwargs["dit_fsdp"] is expected
    assert worker_request.load_kwargs["t5_fsdp"] is expected
    assert worker_request.load_kwargs["ulysses_size"] == 4
    assert worker_request.load_kwargs["offload_model"] is False
    assert entry.default_load_kwargs["dit_fsdp"] is True
