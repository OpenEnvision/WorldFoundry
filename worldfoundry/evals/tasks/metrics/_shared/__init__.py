"""Cross-metric helpers used by metric packages and benchmark runtimes.

- ``aesthetic`` — LAION aesthetic predictor shared by benchmark quality evaluators
- ``bbox`` — IoU helpers for scene-level ``fid`` crops
- ``clip_embed`` — CLIP encode/cosine helpers for benchmark evaluators and ``semsr``, ``manipulation_direction``, ``quality_loss``
- ``conditional_frechet`` — TensorFlow CFID math shared by ``cfid`` and ``ssd``
- ``features`` — OOM-safe batched image features and pairwise cosine distance for benchmark runtimes
- ``images`` — RGB input normalization for image-based helpers
- ``imports`` — import-path setup for bundled metric implementations
- ``lazy`` — lightweight public exports for metric packages
- ``perceptual`` — tensor/device helpers and ``compute_perceptual_bundle`` for ``lpips``, ``ssim``, ``ms_ssim``, ``psnr``, ``fsim``
- ``pyiqa`` — pyiqa-backed image quality helpers for benchmark evaluators
- ``torch_fidelity`` — lazy loader for vendored ``vendor/torch_fidelity`` (``fid``, ``kid``, ``inception_score``, ``precision_recall``, ``ppl``, ``mind``)

Metric-specific helpers live in the metric package; only code shared by several callers belongs here.
"""

from __future__ import annotations

__all__: list[str] = []
