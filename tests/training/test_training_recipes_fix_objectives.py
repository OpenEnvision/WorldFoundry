"""Regression tests for objectives import hygiene and schedule caching (review TR-3/TR-5)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import torch

from worldfoundry.training.api.contracts import PreparedBatch
from worldfoundry.training.objectives.classic_diffusion import (
    ClassicDiffusionConfig,
    ClassicDiffusionObjective,
    extract_schedule,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_flow_matching_symbols_import_without_torch_or_classic_diffusion() -> None:
    code = "\n".join(
        (
            "import sys",
            "from worldfoundry.training.objectives import flow_shift_sigmas",
            "assert 'torch' not in sys.modules, 'package import pulled torch'",
            "assert 'worldfoundry.training.objectives.classic_diffusion' not in sys.modules",
            "from worldfoundry.training.objectives import ClassicDiffusionObjective",
            "assert 'torch' in sys.modules, 'classic diffusion still requires torch'",
        )
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(_REPO_ROOT), env.get("PYTHONPATH")))
    )
    subprocess.run((sys.executable, "-c", code), check=True, env=env, cwd=_REPO_ROOT)


def test_classic_diffusion_corrupt_is_deterministic_and_reuses_schedule_tables() -> None:
    objective = ClassicDiffusionObjective(ClassicDiffusionConfig(num_train_timesteps=16))
    clean = torch.randn(2, 4, 3, 2, 2, generator=torch.Generator().manual_seed(11))
    prepared = PreparedBatch(sample_ids=("a", "b"), clean_latents=clean)

    first = objective.corrupt(prepared, generator=torch.Generator().manual_seed(7))
    second = objective.corrupt(prepared, generator=torch.Generator().manual_seed(7))

    torch.testing.assert_close(first.model_input, second.model_input)
    torch.testing.assert_close(first.target, second.target)
    torch.testing.assert_close(first.sigmas, second.sigmas)
    assert torch.equal(first.timesteps, second.timesteps)

    # The CPU cache entry is the constructor table itself (no copies), and the
    # emitted values match the uncached schedule math exactly.
    alphas, sigmas = objective._schedules_for_device(clean.device)
    assert alphas is objective.alphas
    assert sigmas is objective.sigmas
    assert set(objective._device_schedules) == {clean.device}
    torch.testing.assert_close(first.sigmas, objective.sigmas.gather(0, first.timesteps))
    expected_alpha = extract_schedule(objective.alphas, first.timesteps, clean)
    expected_sigma = extract_schedule(objective.sigmas, first.timesteps, clean)
    torch.testing.assert_close(
        first.model_input,
        expected_alpha.to(clean.dtype) * clean + expected_sigma.to(clean.dtype) * first.noise,
    )
