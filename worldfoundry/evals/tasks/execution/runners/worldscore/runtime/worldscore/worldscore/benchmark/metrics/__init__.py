"""Lazy exports for WorldScore metrics.

WorldScore metrics have separate optional dependencies. Importing every metric
at package import time makes unrelated paths fail when a heavy dependency such
as lietorch is not installed.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "QAlignAestheticMetric": ".iqa_pytorch.qalign_metrics",
    "QAlignQualityMetric": ".iqa_pytorch.qalign_metrics",
    "IQACLIPImageQualityAssessmentMetric": (
        ".iqa_pytorch.clip_iqa_metrics",
        "CLIPImageQualityAssessmentMetric",
    ),
    "CLIPImageQualityAssessmentPlusMetric": ".iqa_pytorch.clip_iqa_metrics",
    "CLIPImageQualityAssessmentPlusRN50_512Metric": ".iqa_pytorch.clip_iqa_metrics",
    "CLIPimageQualityAssessmentPlusVITL14_512Metric": ".iqa_pytorch.clip_iqa_metrics",
    "IQACLIPScoreMetric": (".iqa_pytorch.clip_score_metrics", "CLIPScoreMetric"),
    "IQACLIPAestheticScoreMetric": (
        ".iqa_pytorch.clip_aesthetic_metrics",
        "CLIPAestheticScoreMetric",
    ),
    "MultiScaleImageQualityMetric": ".iqa_pytorch.musiq_metrics",
    "CLIPMLPAestheticScoreMetric": ".metric_impls.clip_mlp_aesthetic_metrics",
    "CLIPConsistencyMetric": ".metric_impls.clip_consistency_metrics",
    "DINOConsistencyMetric": ".metric_impls.dino_consistency_metrics",
    "CameraErrorMetric": ".metric_impls.camera_error_metrics",
    "OpticalFlowAverageEndPointErrorMetric": ".metric_impls.flow_aepe_metrics",
    "GramMatrixMetric": ".metric_impls.gram_matrix_metrics",
    "QAlignVideoAestheticMetric": ".metric_impls.qalign_video_metrics",
    "QAlignVideoQualityMetric": ".metric_impls.qalign_video_metrics",
    "ObjectDetectionMetric": ".metric_impls.object_detection_metrics",
    "ReprojectionErrorMetric": ".metric_impls.reprojection_error_metrics",
    "OpticalFlowMetric": ".metric_impls.flow_metrics",
    "MotionAccuracyMetric": ".metric_impls.motion_accuracy_metrics",
    "MotionSmoothnessMetric": ".metric_impls.motion_smoothness_metrics",
    "CLIPImageQualityAssessmentMetric": ".torchmetrics.clip_iqa_metrics",
    "CLIPScoreMetric": ".torchmetrics.clip_score_metrics",
}

__all__ = tuple(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    target = _EXPORTS[name]
    if isinstance(target, tuple):
        module_name, attr_name = target
    else:
        module_name, attr_name = target, name
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
