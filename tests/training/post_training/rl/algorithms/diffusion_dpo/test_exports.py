from __future__ import annotations

from worldfoundry.training import post_training
from worldfoundry.training.post_training import rl
from worldfoundry.training.post_training.rl import algorithms
from worldfoundry.training.post_training.rl.algorithms import diffusion_dpo


def test_diffusion_dpo_lazy_facades_resolve_to_canonical_runtime() -> None:
    names = (
        "DiffusionDPOBatch",
        "NativeDiffusionDPOEngine",
        "NativeDiffusionDPOTrainingStack",
        "NativeDiffusionDPOTrainingSession",
        "build_native_diffusion_dpo_training_stack",
        "diffusion_dpo_forward_process",
        "sample_diffusion_dpo_forward_process",
        "diffusion_dpo_loss",
    )
    for name in names:
        canonical = getattr(diffusion_dpo, name)
        assert getattr(algorithms, name) is canonical
        assert getattr(rl, name) is canonical
        assert getattr(post_training, name) is canonical
