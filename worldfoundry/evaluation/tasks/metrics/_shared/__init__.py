"""Cross-metric helpers used by multiple evaluation modules.

Keep modules here only when at least one metric package imports them:

- ``clip_embed`` — CLIP encode/cosine helpers for ``semsr``, ``manipulation_direction``, ``quality_loss``
- ``conditional_frechet`` — TensorFlow CFID math shared by ``cfid`` and ``ssd``
- ``bbox`` — IoU helpers for ``object_wise_consistency``
- ``images`` — RGB input normalization for image-based metrics
- ``imports`` — import-path setup for bundled metric implementations
- ``lazy`` — lightweight public exports for metric packages
- ``perceptual`` — tensor/device helpers and ``compute_perceptual_bundle`` for ``lpips``, ``ssim``, ``ms_ssim``, ``psnr``, ``fsim``
- ``torch_fidelity`` — lazy loader for vendored ``vendor/torch_fidelity`` (``fid``, ``kid``, ``inception_score``, ``precision_recall``, ``ppl``, ``mind``)

Metric-specific distribution helpers live in local ``*/compute.py`` modules; only math shared by multiple
metrics belongs here.
"""

from __future__ import annotations

__all__: list[str] = []
