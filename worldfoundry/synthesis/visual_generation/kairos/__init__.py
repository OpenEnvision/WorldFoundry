"""
This package module serves as the public interface for the Kairos framework.

It re-exports key components like `KairosRuntime` for managing the execution environment
and `KairosSynthesis` for facilitating the synthesis process. This makes these core
classes directly accessible when importing from the top-level 'kairos' package.
"""

from .kairos_synthesis import KairosSynthesis
from .runtime import KairosRuntime

# Defines the public API for this module, specifying which names are exported when
# a client does 'from kairos import *'.
__all__ = ["KairosRuntime", "KairosSynthesis"]
