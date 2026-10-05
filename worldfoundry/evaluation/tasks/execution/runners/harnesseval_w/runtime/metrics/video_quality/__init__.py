"""
Video Quality metrics module (Renderer dimension).

Includes: aesthetic_quality, imaging_quality, temporal_flickering,
dynamic_degree, motion_smoothness, hpsv3_quality
"""


def get_aesthetic_quality_metric():
    from .aesthetic_quality import AestheticQualityMetric
    return AestheticQualityMetric


def get_imaging_quality_metric():
    from .imaging_quality import ImagingQualityMetric
    return ImagingQualityMetric


def get_temporal_flickering_metric():
    from .temporal_flickering import TemporalFlickeringMetric
    return TemporalFlickeringMetric


def get_dynamic_degree_metric():
    from .dynamic_degree import DynamicDegreeMetric
    return DynamicDegreeMetric


def get_motion_smoothness_metric():
    from .motion_smoothness import MotionSmoothnessMetric
    return MotionSmoothnessMetric


def get_hpsv3_quality_metric():
    from .hpsv3_quality import HPSv3QualityMetric
    return HPSv3QualityMetric


__all__ = [
    "get_aesthetic_quality_metric",
    "get_imaging_quality_metric",
    "get_temporal_flickering_metric",
    "get_dynamic_degree_metric",
    "get_motion_smoothness_metric",
    "get_hpsv3_quality_metric",
]
