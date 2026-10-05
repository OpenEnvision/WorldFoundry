# This file includes code originally from the Metric3D repository:
# https://github.com/YvanYin/Metric3D
# Licensed under the BSD-2 License. See repository-root THIRD-PARTY-NOTICES for details.

"""Module for base_models -> three_dimensions -> depth -> metric3d -> model -> monodepth_model.py functionality."""

import torch
import torch.nn as nn

from .dense_pipeline import BaseDepthModel


class DepthModel(BaseDepthModel):
    """Depth model implementation."""
    def __init__(self, cfg, **kwards):
        """Init.

        Args:
            cfg: The cfg.
        """
        super(DepthModel, self).__init__(cfg)

    def inference(self, data):
        """Inference.

        Args:
            data: The data.
        """
        with torch.no_grad():
            pred_depth, confidence, output_dict = self.forward(data)
        return pred_depth, confidence, output_dict


def get_monodepth_model(cfg: dict, **kwargs) -> nn.Module:
    """Get monodepth model.

    Args:
        cfg: The cfg.

    Returns:
        The return value.
    """
    # config depth  model
    model = DepthModel(cfg, **kwargs)
    # model.init_weights(load_imagenet_model, imagenet_ckpt_fpath)
    assert isinstance(model, nn.Module)
    return model


def get_configured_monodepth_model(
    cfg: dict,
) -> nn.Module:
    """
    Args:
    @ configs: configures for the network.
    @ load_imagenet_model: whether to initialize from ImageNet-pretrained model.
    @ imagenet_ckpt_fpath: string representing path to file with weights to initialize model with.
    Returns:
    # model: depth model.
    """
    model = get_monodepth_model(cfg)
    return model
