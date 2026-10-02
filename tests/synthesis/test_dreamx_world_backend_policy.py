"""Model initialization must preserve explicitly requested reproducibility."""

from pathlib import Path

import pytest
import torch

from worldfoundry.synthesis.visual_generation.dreamx_world import ar_realtime, realtime


@pytest.mark.parametrize("deterministic", [True, False])
@pytest.mark.parametrize("variant", ["ar", "camera"])
def test_realtime_model_loading_respects_deterministic_convolution_policy(monkeypatch, deterministic, variant):
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warning = torch.is_deterministic_algorithms_warn_only_enabled()
    observed = []
    monkeypatch.setenv("WORLDFOUNDRY_DREAMX_STAGE_CHECKPOINT", "0")
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda _device: None)

    def load(_session):
        observed.append(torch.backends.cudnn.benchmark)
        return (None, None, None, None) if variant == "ar" else (None, None)

    if variant == "ar":
        monkeypatch.setattr(ar_realtime, "resolve_checkpoint", lambda *_args, **_kwargs: Path("checkpoint"))
        session_type = ar_realtime.DreamXWorldARRealtimeSession
        monkeypatch.setattr(session_type, "_load_components", load)
    else:
        monkeypatch.setattr(realtime, "_resolve_checkpoint", lambda *_args, **_kwargs: Path("checkpoint"))
        monkeypatch.setattr(realtime, "_distributed_device", lambda: (0, 1, torch.device("cpu")))
        session_type = realtime.DreamXWorldRealtimeSession
        monkeypatch.setattr(session_type, "_load_pipeline", load)
    try:
        torch.use_deterministic_algorithms(deterministic)
        session_type("checkpoint", wan_model_path="wan")
        assert observed == [not deterministic]
        assert torch.are_deterministic_algorithms_enabled() is deterministic
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warning)
