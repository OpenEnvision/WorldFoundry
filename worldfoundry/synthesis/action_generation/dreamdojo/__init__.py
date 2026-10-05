"""
This module serves as the primary entry point for the dreamdojo package,
exposing key components for runtime evaluation and synthesis.

It simplifies access to core functionalities by re-exporting
`DreamDojoRuntime` for managing and executing world evaluations,
and `DreamDojoSynthesis` for generating or synthesizing world configurations.
"""
from .worldfoundry_runtime import DreamDojoRuntime
from .dreamdojo_synthesis import DreamDojoSynthesis

# Defines the public interface of the module, specifying which names are
# imported when a client does 'from package import *'.
__all__ = ["DreamDojoRuntime", "DreamDojoSynthesis"]