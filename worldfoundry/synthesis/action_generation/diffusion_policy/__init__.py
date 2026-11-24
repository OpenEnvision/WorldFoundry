"""
This module serves as a public interface or re-export mechanism for the
DiffusionPolicySynthesis class.

It enables direct import of DiffusionPolicySynthesis from this module,
simplifying access to this key component within the package structure.
"""

from .diffusion_policy_synthesis import DiffusionPolicySynthesis

# Defines the public API of this module, specifying which names will be
# imported when `from module import *` is used.
__all__ = ["DiffusionPolicySynthesis"]
