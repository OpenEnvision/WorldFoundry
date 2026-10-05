"""
Main module for the CameraCtrl package.

This module serves as the primary entry point, providing access to core CameraCtrl
functionality, including synthesis and runtime components, and defining default
resource paths. It implements lazy loading for submodules to optimize startup performance.
"""
from __future__ import annotations

from pathlib import Path


def runtime_root() -> Path:
    """
    Returns the absolute path to the 'cameractrl_runtime' directory.

    This directory is expected to contain various runtime assets or configurations
    for the CameraCtrl system.

    Returns:
        Path: The absolute path to the 'cameractrl_runtime' directory.
    """
    return Path(__file__).resolve().parent / "cameractrl_runtime"


def __getattr__(name: str):
    """
    Provides lazy loading for CameraCtrl components and constants from submodules.

    This function acts as a module-level attribute getter, allowing direct access
    to specific classes (e.g., CameraCtrlSynthesis, CameraCtrlRuntime) and
    default path constants (e.g., DEFAULT_CAMERACTRL_CKPT) without explicitly
    importing their respective submodules upfront. This helps in managing
    dependencies and potentially improving initial import times.

    Args:
        name (str): The name of the attribute being accessed.

    Returns:
        Any: The requested attribute (class, constant, etc.).

    Raises:
        AttributeError: If the requested attribute is not recognized or available.
    """
    # Define the set of attributes that are intended to be exposed via __getattr__.
    # These attributes will be imported on first access.
    if name in {
        "CameraCtrlSynthesis",
        "CameraCtrlRuntime",
        "DEFAULT_CAMERACTRL_CKPT",
        "DEFAULT_CAMERACTRL_CONFIG",
        "DEFAULT_CAMERACTRL_IMAGE_LORA",
        "DEFAULT_CAMERACTRL_MOTION_ADAPTER",
        "DEFAULT_SD15_ROOT",
    }:
        # Handle the CameraCtrlSynthesis class, which resides in its own dedicated submodule.
        # This allows for specific dependency management for synthesis components.
        if name == "CameraCtrlSynthesis":
            from .synthesis import CameraCtrlSynthesis
            return CameraCtrlSynthesis
        
        # For all other recognized attributes, they are expected to be defined
        # within the 'runtime' submodule.
        from . import runtime
        # Retrieve the attribute from the runtime submodule using getattr.
        return getattr(runtime, name)
    
    # If the requested attribute name is not in the predefined set,
    # it indicates an invalid access, so raise an AttributeError.
    raise AttributeError(name)


__all__ = [
    "CameraCtrlSynthesis",
    "CameraCtrlRuntime",
    "DEFAULT_CAMERACTRL_CKPT",
    "DEFAULT_CAMERACTRL_CONFIG",
    "DEFAULT_CAMERACTRL_IMAGE_LORA",
    "DEFAULT_CAMERACTRL_MOTION_ADAPTER",
    "DEFAULT_SD15_ROOT",
    "runtime_root",
]
