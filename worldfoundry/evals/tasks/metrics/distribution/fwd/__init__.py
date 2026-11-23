"""Fréchet Wavelet Distance (FWD) metric."""

from __future__ import annotations

from pathlib import Path

from worldfoundry.evals.tasks.metrics.registry import metric_module_from_globals

METRIC_ID = "fwd"
ALIASES = ("frechet-wavelet-distance",)
HIGHER_IS_BETTER = False
FAMILY = "distribution"
TAGS = ("distribution", "image_generation")

METRIC_MODULE = metric_module_from_globals(
    metric_id=METRIC_ID,
    aliases=ALIASES,
    description="Fréchet Wavelet Distance (PyTorch-FWD, in-tree).",
    family=FAMILY,
    higher_is_better=HIGHER_IS_BETTER,
    tags=TAGS,
)


def compute_fwd(
    reference_dir: str | Path,
    generated_dir: str | Path,
    *,
    wavelet: str = "Haar",
    max_level: int = 4,
    log_scale: bool = False,
    batch_size: int = 128,
    resize: int | None = None,
) -> float:
    from worldfoundry.evals.tasks.metrics.distribution.fwd.vendor.pytorchfwd.fwd import compute_fwd as _compute_fwd

    return float(
        _compute_fwd(
            [str(reference_dir), str(generated_dir)],
            wavelet=wavelet,
            max_level=max_level,
            log_scale=log_scale,
            batch_size=batch_size,
            resize=resize,
        )
    )


compute = compute_fwd

__all__ = ["ALIASES", "FAMILY", "HIGHER_IS_BETTER", "METRIC_ID", "METRIC_MODULE", "compute", "compute_fwd"]
