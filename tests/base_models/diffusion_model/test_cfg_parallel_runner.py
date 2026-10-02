from __future__ import annotations

from types import SimpleNamespace

import torch

from worldfoundry.base_models.diffusion_model.contracts import (
    Conditioning,
    DenoiserOutput,
)
from worldfoundry.base_models.diffusion_model.runners.base import (
    NativeDiffusionRunner,
    RunnerComponents,
)


class _ParallelRunner(NativeDiffusionRunner):
    def __init__(self, **kwargs) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        super().__init__(**kwargs)

    def _call_denoiser_with_conditioning(
        self,
        context,
        *,
        latents,
        branch,
        conditioning,
    ) -> DenoiserOutput:
        del context
        self.calls.append((branch, dict(conditioning)))
        value = 3.0 if branch == "positive" else 1.0
        return DenoiserOutput(sample=torch.full_like(latents, value))


def test_cfg_parallel_runs_one_local_branch_and_gathers_both(monkeypatch) -> None:
    import torch.distributed as dist

    from worldfoundry.core.distributed import sequence_parallel_runtime

    group = object()
    monkeypatch.setattr(
        sequence_parallel_runtime,
        "get_cfg_parallel_world_size",
        lambda: 2,
    )
    monkeypatch.setattr(
        sequence_parallel_runtime,
        "get_cfg_parallel_group",
        lambda: group,
    )
    monkeypatch.setattr(sequence_parallel_runtime, "get_cfg_parallel_rank", lambda: 0)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)

    def fake_all_gather(outputs, local, *, group) -> None:
        del group
        outputs[0].copy_(local)
        outputs[1].fill_(1.0)

    monkeypatch.setattr(dist, "all_gather", fake_all_gather)
    components = RunnerComponents(
        denoiser=object(),
        conditioner=object(),
        latent_initializer=object(),
        scheduler=object(),
        decoder=object(),
    )
    runner = _ParallelRunner(
        model_id="cfg-test",
        components=components,
        cfg_parallel_degree=2,
    )
    context = SimpleNamespace(
        conditioning=Conditioning(
            positive={"context": "positive"},
            negative={"context": "negative"},
            shared={},
        ),
        request=SimpleNamespace(
            sampling=SimpleNamespace(guidance_scale=5.0),
        ),
    )
    output = runner.predict(context, torch.zeros(1, 2))
    torch.testing.assert_close(output.sample, torch.full((1, 2), 11.0))
    assert len(runner.calls) == 1
    branch, conditioning = runner.calls[0]
    assert branch == "positive"
    assert conditioning["_worldfoundry_cfg_parallel_request"] is True
    assert output.extras["positive"]["cfg_parallel_rank"] == 0
    assert output.extras["negative"]["cfg_parallel_rank"] == 1
    assert runner._parallel_optimization_report() == {
        "requested": {"cfg_parallel": 2},
        "effective": {"cfg_parallel": "branch-per-rank"},
        "fallbacks": [],
        "quality_tier": "exact",
        "runtime": {
            "cfg_parallel_local_branch_calls": {"positive": 1, "negative": 0},
            "cfg_parallel_collective_calls": 1,
        },
    }
