"""
This module provides the `HunyuanGameCraftSynthesis` class, a synthesis adapter that wraps
the `HunyuanGameCraftRuntime` to integrate with the `BaseSynthesis` interface.
It also re-exports various components and utilities from the `hunyuan_game_craft.runtime`
module for direct access.
"""
from __future__ import annotations

from typing import Any

from worldfoundry.synthesis.visual_generation.hunyuan_world.gamecraft_inference.runtime import (
    ACTION_DICT,
    ActionToPoseFromID,
    Camera,
    GetPoseEmbedsFromPoses,
    GetPoseEmbedsFromTxt,
    HunyuanGameCraftRuntime,
    align_to,
    convert_videos_to_grid,
    custom_meshgrid,
    euler_to_quaternion,
    generate_motion_segment,
    get_c2w,
    get_relative_pose,
    load_runtime,
    quaternion_to_rotation_matrix,
    ray_condition,
)

from ...base_synthesis import BaseSynthesis


class HunyuanGameCraftSynthesis(BaseSynthesis):
    """
    A synthesis adapter that wraps the `HunyuanGameCraftRuntime` to conform to the
    `BaseSynthesis` interface.

    This class provides a thin layer over the core runtime, allowing it to be
    used within systems expecting a `BaseSynthesis` object. It delegates
    most calls directly to the underlying runtime instance.
    """

    def __init__(self, runtime: HunyuanGameCraftRuntime):
        """
        Initializes the `HunyuanGameCraftSynthesis` adapter with a `HunyuanGameCraftRuntime` instance.

        Args:
            runtime: An instance of `HunyuanGameCraftRuntime` to be wrapped.
        """
        super().__init__()
        self.runtime = runtime

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> "HunyuanGameCraftSynthesis":
        """
        Creates a `HunyuanGameCraftSynthesis` instance by loading a `HunyuanGameCraftRuntime`
        from pretrained weights.

        Args:
            *args: Positional arguments to pass to `HunyuanGameCraftRuntime.from_pretrained`.
            **kwargs: Keyword arguments to pass to `HunyuanGameCraftRuntime.from_pretrained`.

        Returns:
            A new `HunyuanGameCraftSynthesis` instance wrapping the loaded runtime.
        """
        return cls(HunyuanGameCraftRuntime.from_pretrained(*args, **kwargs))

    def __getattr__(self, name: str) -> Any:
        """
        Delegates attribute access to the wrapped `HunyuanGameCraftRuntime` instance.

        This allows direct access to methods and properties of the underlying
        runtime model without explicitly calling `self.runtime.<attribute>`.

        Args:
            name: The name of the attribute to retrieve.

        Returns:
            The attribute from the internal `runtime` object.
        """
        return getattr(self.runtime, name)

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        """
        Calls the `predict` method of the wrapped `HunyuanGameCraftRuntime` instance.

        Args:
            *args: Positional arguments to pass to `runtime.predict`.
            **kwargs: Keyword arguments to pass to `runtime.predict`.

        Returns:
            The result from `runtime.predict`.
        """
        return self.runtime.predict(*args, **kwargs)

    def predict_per_action(self, *args: Any, **kwargs: Any) -> Any:
        """
        Calls the `predict_per_action` method of the wrapped `HunyuanGameCraftRuntime` instance.

        Args:
            *args: Positional arguments to pass to `runtime.predict_per_action`.
            **kwargs: Keyword arguments to pass to `runtime.predict_per_action`.

        Returns:
            The result from `runtime.predict_per_action`.
        """
        return self.runtime.predict_per_action(*args, **kwargs)


__all__ = [
    "ACTION_DICT",
    "ActionToPoseFromID",
    "Camera",
    "GetPoseEmbedsFromPoses",
    "GetPoseEmbedsFromTxt",
    "HunyuanGameCraftRuntime",
    "HunyuanGameCraftSynthesis",
    "align_to",
    "convert_videos_to_grid",
    "custom_meshgrid",
    "euler_to_quaternion",
    "generate_motion_segment",
    "get_c2w",
    "get_relative_pose",
    "load_runtime",
    "quaternion_to_rotation_matrix",
    "ray_condition",
]