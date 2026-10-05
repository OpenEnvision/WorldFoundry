"""FID computation via in-tree torch-fidelity (Inception-v3 features)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from worldfoundry.evaluation.tasks.metrics._shared.torch_fidelity import calculate_metrics, prepare_image_input


def resolve_distribution_inputs(
    reference: str | Path | Sequence[str | Path] | None,
    generated: str | Path | Sequence[str | Path] | None,
) -> tuple[str | Path | Sequence[str | Path], str | Path | Sequence[str | Path]]:
    if reference is None or generated is None:
        raise ValueError("reference and generated inputs are required")
    return reference, generated


def _parse_numeric_result(result: Mapping[str, Any]) -> dict[str, float]:
    return {str(key): float(value) for key, value in result.items() if isinstance(value, (int, float))}


def compute_distribution_metrics(
    reference: str | Path | Sequence[str | Path],
    generated: str | Path | Sequence[str | Path],
    *,
    metrics: Sequence[str] = ("fid",),
    feature_extractor: str = "inception-v3-compat",
    batch_size: int = 64,
    cuda: bool = True,
    **kwargs: Any,
) -> dict[str, float]:
    """Run distribution metrics with backend input1=generated, input2=reference."""
    ref, gen = resolve_distribution_inputs(reference, generated)
    metric_flags = {name: True for name in metrics}
    result = calculate_metrics()(
        input1=prepare_image_input(gen, **kwargs),
        input2=prepare_image_input(ref, **kwargs),
        batch_size=batch_size,
        cuda=cuda,
        feature_extractor=feature_extractor,
        **metric_flags,
        **kwargs,
    )
    return _parse_numeric_result(result)


_SWAV_EXTRACTORS = frozenset({"swav", "swav-resnet50", "swav_resnet50"})


def compute_fid(
    reference: str | Path | Sequence[str | Path],
    generated: str | Path | Sequence[str | Path],
    *,
    batch_size: int = 64,
    cuda: bool = True,
    feature_extractor: str = "inception-v3-compat",
    **kwargs: Any,
) -> float:
    if feature_extractor in _SWAV_EXTRACTORS:
        from .swav import compute_swav_fid

        device = kwargs.pop("device", None)
        if device is None and not cuda:
            device = "cpu"
        return compute_swav_fid(
            reference,
            generated,
            batch_size=batch_size,
            device=device,
            **kwargs,
        )
    result = calculate_metrics()(
        input1=prepare_image_input(reference, **kwargs),
        input2=prepare_image_input(generated, **kwargs),
        batch_size=batch_size,
        cuda=cuda,
        feature_extractor=feature_extractor,
        fid=True,
        **kwargs,
    )
    return float(result["frechet_inception_distance"])


def compute_paired_fid_kid(
    reference_batches: Any,
    generated_batches: Any,
    *,
    device: str = "cuda",
    kid_subset_size: int = 100,
) -> tuple[float, float]:
    """FID and KID over preprocessed image batches (torchmetrics backend).

    Protocol-compatibility path for benchmark runtimes (e.g. MiraBench) whose
    official scores are defined against ``torchmetrics`` Inception statistics
    rather than the torch-fidelity path of :func:`compute_fid`. Runtimes keep
    only their frame sampling/preprocessing and delegate the metric math here.

    Both inputs are iterables of image tensors shaped ``(N, 3, H, W)`` in the
    value range expected by ``normalize=True`` (floats in ``[0, 1]``).
    Returns ``(fid, kid_mean)``.
    """
    import torch
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance
    from torchmetrics.utilities.data import dim_zero_cat

    fid_metric = FrechetInceptionDistance(normalize=True).to(device)
    kid_metric = KernelInceptionDistance(normalize=True, subset_size=kid_subset_size).to(device)
    with torch.no_grad():
        for batch in reference_batches:
            batch = batch.to(device)
            fid_metric.update(batch, real=True)
            kid_metric.update(batch, real=True)
        for batch in generated_batches:
            batch = batch.to(device)
            fid_metric.update(batch, real=False)
            kid_metric.update(batch, real=False)
    generated_count = dim_zero_cat(kid_metric.fake_features).shape[0]
    if generated_count < kid_metric.subset_size:
        raise ValueError(
            f"KID subset_size={kid_metric.subset_size} exceeds the number of "
            f"generated samples ({generated_count}); pass a smaller kid_subset_size"
        )
    fid_score = float(fid_metric.compute().item())
    kid_mean = float(kid_metric.compute()[0].item())
    return fid_score, kid_mean


def summarize_distribution_metrics(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize torch-fidelity metric keys to stable WorldFoundry names."""
    aliases = {
        "inception_score_mean": "is_mean",
        "inception_score_std": "is_std",
        "frechet_inception_distance": "fid",
        "kernel_inception_distance_mean": "kid_mean",
        "kernel_inception_distance_std": "kid_std",
        "monge_inception_distance": "mind",
        "perceptual_path_length_mean": "ppl_mean",
        "perceptual_path_length_std": "ppl_std",
        "perceptual_path_length_raw": "ppl_raw",
        "precision": "precision",
        "recall": "recall",
        "f_score": "f_score",
    }
    return {aliases.get(key, key): value for key, value in payload.items()}


__all__ = [
    "compute_distribution_metrics",
    "compute_fid",
    "compute_paired_fid_kid",
    "resolve_distribution_inputs",
    "summarize_distribution_metrics",
]
