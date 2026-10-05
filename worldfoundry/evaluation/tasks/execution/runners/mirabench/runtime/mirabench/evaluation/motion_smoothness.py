"""MiraBench adapter for the shared in-tree AMT smoothness runtime."""

import torch

from worldfoundry.base_models.perception_core.frame_interpolation.amt.motion_smoothness import (
    FrameProcess,
    MotionSmoothness,
)


def EvaluateMotionSmoothness(model, store_image_folder, device):
    del device
    with torch.inference_mode():
        return model.motion_score(store_image_folder)


__all__ = ["EvaluateMotionSmoothness", "FrameProcess", "MotionSmoothness"]
