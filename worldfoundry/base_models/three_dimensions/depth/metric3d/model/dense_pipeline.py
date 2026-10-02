# This file includes code originally from the Metric3D repository:
# https://github.com/YvanYin/Metric3D
# Licensed under the BSD-2 License. See repository-root THIRD-PARTY-NOTICES for details.

"""Module for base_models -> three_dimensions -> depth -> metric3d -> model -> dense_pipeline.py functionality."""

import importlib

import torch
import torch.nn as nn


def get_func(func_name):
    """
    Helper to return a function object by name. func_name must identify
    a function in this module or the path to a function relative to the base
    module.
    @ func_name: function name.
    """
    if func_name == "":
        return None
    try:
        parts = func_name.split(".")
        # Refers to a function in this module
        if len(parts) == 1:
            return globals()[parts[0]]
        # Otherwise, assume we're referencing a module under modeling
        module_name = ".".join(parts[:-1])
        module = importlib.import_module(module_name)
        return getattr(module, parts[-1])
    except Exception:
        raise RuntimeError(f"Failed to find function: {func_name}")


class DensePredModel(nn.Module):
    """Dense pred model implementation."""
    def __init__(self, cfg) -> None:
        """Init.

        Args:
            cfg: The cfg.

        Returns:
            The return value.
        """
        super(DensePredModel, self).__init__()

        self.encoder = get_func(
            "worldfoundry.base_models.three_dimensions.depth.metric3d.model." + cfg.model.backbone.prefix + cfg.model.backbone.type
        )(**cfg.model.backbone)
        self.decoder = get_func(
            "worldfoundry.base_models.three_dimensions.depth.metric3d.model." + cfg.model.decode_head.prefix + cfg.model.decode_head.type
        )(cfg)

    def forward(self, input, **kwargs):
        """Forward.

        Args:
            input: The input.
        """
        # [f_32, f_16, f_8, f_4]
        features = self.encoder(input)
        out = self.decoder(features, **kwargs)
        return out


class BaseDepthModel(nn.Module):
    """Base depth model implementation."""
    def __init__(self, cfg, **kwargs) -> None:
        """Init.

        Args:
            cfg: The cfg.

        Returns:
            The return value.
        """
        super(BaseDepthModel, self).__init__()
        self.depth_model = DensePredModel(cfg)

    def forward(self, data):
        """Forward.

        Args:
            data: The data.
        """
        output = self.depth_model(**data)

        return output["prediction"], output["confidence"], output

    def inference(self, data):
        """Inference.

        Args:
            data: The data.
        """
        with torch.no_grad():
            pred_depth, confidence, _ = self.forward(data)
        return pred_depth, confidence
