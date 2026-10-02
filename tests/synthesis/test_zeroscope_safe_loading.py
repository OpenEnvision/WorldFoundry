from __future__ import annotations

from types import SimpleNamespace

import diffusers

from worldfoundry.synthesis.visual_generation.zeroscope.worldfoundry_runtime import (
    ZeroScopeRuntime,
)


def test_zeroscope_requires_safetensors(monkeypatch, tmp_path) -> None:
    captured = {}

    class _Pipe:
        def to(self, device):
            captured["device"] = device
            return self

    def _from_pretrained(path, **kwargs):
        captured["path"] = path
        captured["kwargs"] = kwargs
        return _Pipe()

    monkeypatch.setattr(diffusers.TextToVideoSDPipeline, "from_pretrained", _from_pretrained)
    runtime = ZeroScopeRuntime(
        profile=SimpleNamespace(),
        model_id="zeroscope",
        device="cpu",
        model_path=str(tmp_path),
    )

    runtime.ensure_pipe()

    assert captured["kwargs"]["use_safetensors"] is True
    assert captured["kwargs"]["local_files_only"] is True
