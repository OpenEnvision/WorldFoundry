# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
# Adapted from Robbyant LingBot-VLA and LingBot-VLA-v2 (Apache-2.0).
"""Shared joint metadata and checkpoint input padding for LingBot-VLA policies."""

import ast
from typing import Any

import torch
from pydantic import BaseModel
from torch.nn import functional as F


class FeatureInfo(BaseModel):
    # Stores metadata about robot joints and camera images used for feature extraction.
    # Parsed from the training data config to determine which joints/images to process.
    joints: list[str] | None = None
    images: list[str] | None = None
    joints_max_dim: dict | None = None

    def update_info(self, data_config: Any) -> None:
        joints_info = data_config.joints
        self.images = ["observation.images." + image for image in data_config.cameras]

        joints = []
        joints_max_dim = {}
        for s in joints_info:
            joint_info = dict(s) if isinstance(s, dict) else ast.literal_eval(s)
            joint = next(iter(joint_info.keys()))
            if joint_info[joint] == 0:
                continue

            joints.append(joint)
            joints_max_dim.update(joint_info)
        self.joints = joints
        self.joints_max_dim = joints_max_dim


def prepare_state(observation: dict[str, torch.Tensor], max_state_dim: int) -> torch.Tensor:
    """Pad the state to the maximum state dimension.

    Args:
        observation (dict[str, Tensor])

    Returns:
        state (torch.Tensor): (*b, max_state_dim) padded state tensor
    """
    state = observation["state"]
    state = F.pad(state, (0, max_state_dim - state.shape[-1]))
    return state


def prepare_action(observation: dict[str, torch.Tensor], max_action_dim: int) -> torch.Tensor:
    """Pad the action to the maximum action dimension.

    Args:
        observation (dict[str, Tensor])

    Returns:
        action (torch.Tensor): (*b, n, max_action_dim) padded action tensor
        action_dim (int): the actual dimension of the action before padding
    """
    action = observation["action"]
    action = F.pad(action, (0, max_action_dim - action.shape[-1]))
    return action
