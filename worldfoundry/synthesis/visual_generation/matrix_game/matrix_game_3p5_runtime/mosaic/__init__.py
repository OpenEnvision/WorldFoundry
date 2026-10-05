"""Lazy public surface for the Matrix-Game Mosaic inference runtime."""

from importlib import import_module

_EXPORTS = {
    "MOSAIC_INTRINSICS_MODES": (".config", "MOSAIC_INTRINSICS_MODES"),
    "parse_pipeline_args": (".config", "parse_pipeline_args"),
    "wan_mosaic_parser": (".config", "wan_mosaic_parser"),
    "build_mosaic_inference_dataset": (
        ".datasets",
        "build_mosaic_inference_dataset",
    ),
    "run_mosaic_segment_inference": (
        ".inference",
        "run_mosaic_segment_inference",
    ),
    "main": (".main", "main"),
    "run_mosaic_inference_task": (".runner", "run_mosaic_inference_task"),
    "WanMosaicPipelineModule": (".pipeline_module", "WanMosaicPipelineModule"),
    "build_mosaic_pipeline_module": (
        ".pipeline_module",
        "build_mosaic_pipeline_module",
    ),
    "run_mosaic_inference": (".rollout", "run_mosaic_inference"),
}


def __getattr__(name):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


__all__ = sorted(_EXPORTS)
