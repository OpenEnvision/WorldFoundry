"""ViGeo construction and lifecycle helpers for Evoke inference."""
from __future__ import annotations

from typing import Optional

_VIGEO_KEYS = ("num_tokens", "mode", "chunk_size", "intr_source", "conf_transform",
               "scale_mode", "anchor_windows", "total_budget", "cache_keep_frames",
               # only read by the baseline-free scale modes: scale_value for scale_mode=fixed,
               # depth_median_target for scale_mode=depth_median (see vigeo_cloud._resolve_scale_baseline_free)
               "scale_value", "depth_median_target")


def build_estimator(backend: str, device, process_res: int,
                    weights: Optional[str] = None, src: Optional[str] = None,
                    vigeo_opts: Optional[dict] = None):
    """Construct the sole supported inference depth backend."""
    if (backend or "vigeo").lower() != "vigeo":
        raise ValueError(f"this inference-only Evoke port supports depth_backend='vigeo', got {backend!r}")
    from .vigeo_cloud import ViGeoDepthEstimator, _VIGEO_WEIGHTS

    kwargs = dict(device=device, process_res=int(process_res),
                  weights=(weights or str(_VIGEO_WEIGHTS)))
    if src:
        kwargs["src"] = src
    unknown = sorted(set(vigeo_opts or {}) - set(_VIGEO_KEYS))
    if unknown:
        raise ValueError(f"unknown ViGeo option(s) {unknown}; expected any of {sorted(_VIGEO_KEYS)}")
    for key in _VIGEO_KEYS:
        value = (vigeo_opts or {}).get(key)
        if value is not None:
            kwargs[key] = value
    return ViGeoDepthEstimator(**kwargs)


def reset_stream(estimator) -> None:
    """Drop per-stream estimator state at a stream boundary; a no-op for stateless backends.

    A stream is one contiguous video: one training sample, one self-forcing rollout, or one generation.
    ViGeo holds a kv-cache and a locked depth scale across windows, and the training-side estimator is
    a process-wide singleton shared by every sample, so without this the first video's cache and scale
    leak into all later ones -- silently, since ViGeo does not raise on an inconsistent stream.
    """
    fn = getattr(estimator, "reset_stream", None)
    if callable(fn):
        fn()


def check_assets(backend: str, weights: Optional[str] = None, src: Optional[str] = None) -> None:
    """Fail before model loading when ViGeo source or weights are missing."""
    if (backend or "vigeo").lower() != "vigeo":
        raise ValueError(f"this inference-only Evoke port supports depth_backend='vigeo', got {backend!r}")
    from .vigeo_cloud import check_assets as _check
    _check(src, weights)
