"""Compatibility re-export of :mod:`worldfoundry.core.platform.compile_cache`.

The implementation lives in core so attention, kernels, inference, and I/O can
use the persistent compiler cache without importing the runtime layer.
"""

from worldfoundry.core.platform.compile_cache import (
    CompileCacheLayout,
    CompilePolicy,
    compile_callable_cached,
    compile_module_cached,
    configure_persistent_compile_cache,
)

__all__ = [
    "CompileCacheLayout",
    "CompilePolicy",
    "compile_callable_cached",
    "compile_module_cached",
    "configure_persistent_compile_cache",
]
