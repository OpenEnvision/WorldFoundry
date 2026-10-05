"""Geometry Priors visual generation pipeline module."""

from __future__ import annotations

from worldfoundry.synthesis.visual_generation.memory.runtime import RuntimeMemory
from worldfoundry.operators.geometry_prior_operator import GeometryPriorOperator
from worldfoundry.pipelines.pipeline_utils import PipelineABC
from worldfoundry.synthesis.visual_generation.geometry_priors import GeometryPriorSynthesis


class GeometryPriorPipeline(PipelineABC):
    """Pipeline surface for standalone in-tree geometry-prior integrations."""

    MODEL_ID = "geometry-prior"
    MODEL_PATH_OPTION = "model_path"
    OPERATOR_CLS = GeometryPriorOperator
    MEMORY_CLS = RuntimeMemory
    SYNTHESIS_CLS = GeometryPriorSynthesis
    MEMORY_RECORD_TYPE = "geometry_prior_preflight"
    generation_type = "geometry_prior"


class DAPPipeline(GeometryPriorPipeline):
    """Pipeline implementation for DAP visual generation."""
    MODEL_ID = "dap"


class DepthAnythingV2PriorPipeline(GeometryPriorPipeline):
    """Pipeline implementation for DepthAnythingV2Prior visual generation."""
    MODEL_ID = "depth-anything-v2-prior"


class DepthAnythingV3PriorPipeline(GeometryPriorPipeline):
    """Pipeline implementation for DepthAnythingV3Prior visual generation."""
    MODEL_ID = "depth-anything-v3-prior"


class DUSt3RPipeline(GeometryPriorPipeline):
    """Pipeline implementation for DUSt3R visual generation."""
    MODEL_ID = "dust3r"


class DUSt3RBaseModelPipeline(GeometryPriorPipeline):
    """Pipeline implementation for DUSt3RBaseModel visual generation."""
    MODEL_ID = "dust3r-base-model"


class GeoCalibPriorPipeline(GeometryPriorPipeline):
    """Pipeline implementation for GeoCalibPrior visual generation."""
    MODEL_ID = "geocalib-prior"


class Metric3DPriorPipeline(GeometryPriorPipeline):
    """Pipeline implementation for Metric3DPrior visual generation."""
    MODEL_ID = "metric3d-prior"


class PriorDepthAnythingPipeline(GeometryPriorPipeline):
    """Pipeline implementation for PriorDepthAnything visual generation."""
    MODEL_ID = "prior-depth-anything"


class TrackAnythingPriorPipeline(GeometryPriorPipeline):
    """Pipeline implementation for TrackAnythingPrior visual generation."""
    MODEL_ID = "track-anything-prior"


class UniDepthV2PriorPipeline(GeometryPriorPipeline):
    """Pipeline implementation for UniDepthV2Prior visual generation."""
    MODEL_ID = "unidepth-v2-prior"


class UniK3DPriorPipeline(GeometryPriorPipeline):
    """Pipeline implementation for UniK3DPrior visual generation."""
    MODEL_ID = "unik3d-prior"


class VideoDepthAnythingPriorPipeline(GeometryPriorPipeline):
    """Pipeline implementation for VideoDepthAnythingPrior visual generation."""
    MODEL_ID = "video-depth-anything-prior"


__all__ = [
    "DAPPipeline",
    "DepthAnythingV2PriorPipeline",
    "DepthAnythingV3PriorPipeline",
    "DUSt3RBaseModelPipeline",
    "DUSt3RPipeline",
    "GeoCalibPriorPipeline",
    "GeometryPriorPipeline",
    "Metric3DPriorPipeline",
    "PriorDepthAnythingPipeline",
    "TrackAnythingPriorPipeline",
    "UniDepthV2PriorPipeline",
    "UniK3DPriorPipeline",
    "VideoDepthAnythingPriorPipeline",
]
