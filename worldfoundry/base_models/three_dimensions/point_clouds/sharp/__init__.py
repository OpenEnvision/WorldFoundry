"""Inference-only SHARP panorama model using WorldFoundry DINO/Depth Anything."""

from __future__ import annotations

from typing import Any

__all__ = [
    "GaussianDecoderParams",
    "Gaussians3D",
    "InitializerParams",
    "PanoGaussianPredictor",
    "PredictorParams",
    "build_panorama_predictor",
    "create_depth_model",
    "create_panorama_predictor",
    "load_predictor_checkpoint",
    "save_panorama_ply",
]


def __getattr__(name: str) -> Any:
    if name in {
        "GaussianDecoderParams",
        "Gaussians3D",
        "InitializerParams",
        "PredictorParams",
    }:
        from .types import (
            GaussianDecoderParams,
            Gaussians3D,
            InitializerParams,
            PredictorParams,
        )

        return {
            "GaussianDecoderParams": GaussianDecoderParams,
            "Gaussians3D": Gaussians3D,
            "InitializerParams": InitializerParams,
            "PredictorParams": PredictorParams,
        }[name]
    if name == "save_panorama_ply":
        from .io import save_panorama_ply

        return save_panorama_ply
    if name in {
        "PanoGaussianPredictor",
        "build_panorama_predictor",
        "create_depth_model",
        "create_panorama_predictor",
        "load_predictor_checkpoint",
    }:
        from .panorama import (
            PanoGaussianPredictor,
            build_panorama_predictor,
            create_depth_model,
            create_panorama_predictor,
            load_predictor_checkpoint,
        )

        return {
            "PanoGaussianPredictor": PanoGaussianPredictor,
            "build_panorama_predictor": build_panorama_predictor,
            "create_depth_model": create_depth_model,
            "create_panorama_predictor": create_panorama_predictor,
            "load_predictor_checkpoint": load_predictor_checkpoint,
        }[name]
    raise AttributeError(name)
