"""
This module serves as a convenient aggregation point for key components related to VMem (Visual Memory) synthesis
and runtime environment management within the `worldfoundry` framework.

It re-exports essential constants, functions, and classes from `worldfoundry.synthesis.visual_generation.vmem.runtime_env`
and `worldfoundry.synthesis.visual_generation.vmem.vmem_synthesis`, providing a simplified interface for
accessing them. This includes default repository paths, runtime root directories, configuration utilities,
and the core `VMemSynthesis` class.
"""

from worldfoundry.synthesis.visual_generation.vmem.runtime_env import (
    DEFAULT_VMEM_RUNTIME_ROOT,
    default_config_path,
    runtime_root,
)

from .vmem_synthesis import DEFAULT_VMEM_REPO, DEFAULT_VMEM_SURFEL_REPO, VMemSynthesis

__all__ = [
    "DEFAULT_VMEM_REPO",
    "DEFAULT_VMEM_RUNTIME_ROOT",
    "DEFAULT_VMEM_SURFEL_REPO",
    "VMemSynthesis",
    "default_config_path",
    "runtime_root",
]
