"""
This module serves as a public interface for the `infinite_world_synthesis` subpackage.

It re-exports key components, specifically `DEFAULT_NEGATIVE_PROMPT` and `InfiniteWorldSynthesis`
from the `.infinite_world_synthesis` submodule, making them directly accessible when
importing from the parent package. This simplifies imports for users of the library.
"""

from .infinite_world_synthesis import DEFAULT_NEGATIVE_PROMPT, InfiniteWorldSynthesis

# Define the public API of this module for tools like 'from package import *'
__all__ = ["DEFAULT_NEGATIVE_PROMPT", "InfiniteWorldSynthesis"]
