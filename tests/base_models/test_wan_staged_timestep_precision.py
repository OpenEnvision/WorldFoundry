from __future__ import annotations

import pytest

# This test module imports worldfoundry code that requires the optional
# "ftfy" dependency at import time; skip when it is unavailable.
pytest.importorskip("ftfy")

import torch

from worldfoundry.base_models.diffusion_model.runners.wan_staged import (
    _prepare_scheduler_timestep,
    model_fn_wan_video,
)
from worldfoundry.core.nn import sinusoidal_embedding_1d


def test_wan_scheduler_timestep_keeps_reference_precision() -> None:
    timestep = torch.tensor(961.53845, dtype=torch.float32)

    prepared = _prepare_scheduler_timestep(timestep, device="cpu")

    assert prepared.dtype is torch.float32
    assert prepared.item() == timestep.item()


def test_wan_casts_only_the_finished_time_embedding_to_model_dtype() -> None:
    seen: list[torch.Tensor] = []

    class StopAfterTimeEmbedding(RuntimeError):
        pass

    class Recorder:
        def __call__(self, value: torch.Tensor) -> torch.Tensor:
            seen.append(value.detach().clone())
            raise StopAfterTimeEmbedding

    class DummyDiT:
        freq_dim = 256
        time_embedding = Recorder()

    timestep = torch.tensor([961.53845], dtype=torch.float32)
    x = torch.zeros(1, dtype=torch.bfloat16)

    try:
        model_fn_wan_video(DummyDiT(), x=x, timestep=timestep)
    except StopAfterTimeEmbedding:
        pass
    else:
        raise AssertionError("dummy time embedding did not run")

    expected = sinusoidal_embedding_1d(256, timestep).to(torch.bfloat16)
    quantized_early = sinusoidal_embedding_1d(256, timestep.to(torch.bfloat16)).to(torch.bfloat16)
    assert seen[0].dtype is torch.bfloat16
    assert torch.equal(seen[0], expected)
    assert not torch.equal(seen[0], quantized_early)
