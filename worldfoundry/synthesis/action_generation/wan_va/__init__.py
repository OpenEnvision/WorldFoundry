"""LingBot-VA Wan-VA foundation source package."""

from __future__ import annotations

from pathlib import Path

RUNTIME_ROOT = Path(__file__).resolve().parent
WAN_VA_ROOT = RUNTIME_ROOT
WAN_VA_PACKAGE = "worldfoundry.synthesis.action_generation.wan_va"


__all__ = ["RUNTIME_ROOT", "WAN_VA_PACKAGE", "WAN_VA_ROOT"]
