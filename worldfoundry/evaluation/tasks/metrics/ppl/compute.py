"""Perceptual Path Length computation via in-tree torch-fidelity."""

from __future__ import annotations

from typing import Any

from worldfoundry.evaluation.tasks.metrics._shared.torch_fidelity import calculate_metrics


def compute_ppl(
    generative_model: Any,
    *,
    batch_size: int = 64,
    cuda: bool = True,
    num_samples: int = 5000,
    **kwargs: Any,
) -> dict[str, Any]:
    """Return backend statistics and, for ``ppl_reduction='none'``, per-sample distances."""
    from torch import nn

    from worldfoundry.evaluation.tasks.metrics._shared.vendor.torch_fidelity import (
        GenerativeModelBase,
        GenerativeModelModuleWrapper,
    )

    # External torch-fidelity wrappers carry the same generator metadata.
    if isinstance(generative_model, nn.Module) and not isinstance(generative_model, GenerativeModelBase):
        generative_model = GenerativeModelModuleWrapper(
            generative_model,
            generative_model.z_size,
            generative_model.z_type,
            generative_model.num_classes,
            make_eval=False,
        )
    return calculate_metrics()(
        input1=generative_model,
        batch_size=batch_size,
        cuda=cuda,
        ppl=True,
        input1_model_num_samples=num_samples,
        **kwargs,
    )


__all__ = ["compute_ppl"]
