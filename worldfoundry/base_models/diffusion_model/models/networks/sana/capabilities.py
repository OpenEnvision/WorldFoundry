"""Optional acceleration probes used by the SANA graph.

FlashAttention, xFormers, and Triton LiteMLA / MBConv are imported only when
these helpers report the extra is installed.  Image inference can stay on
the PyTorch LiteLA path without pulling the fused kernels.
"""

import importlib.util
import importlib.metadata

_xformers_available = importlib.util.find_spec("xformers") is not None
try:
    if _xformers_available:
        importlib.metadata.version("xformers")
except importlib.metadata.PackageNotFoundError:
    _xformers_available = False

_triton_modules_available = importlib.util.find_spec("triton") is not None
try:
    if _triton_modules_available:
        importlib.metadata.version("triton")
except (ImportError, importlib.metadata.PackageNotFoundError):
    _triton_modules_available = False


_flash_attn_func = None
try:
    from flash_attn.cute import flash_attn_func as _flash_attn_func
except ImportError:
    try:
        from flash_attn_interface import flash_attn_func as _flash_attn_func
    except ImportError:
        try:
            from flash_attn import flash_attn_func as _flash_attn_func
        except ImportError:
            _flash_attn_func = None

_flash_attn_available = _flash_attn_func is not None


def get_flash_attn_func():
    """Return the first compatible FlashAttention callable, if installed."""
    return _flash_attn_func


def is_flash_attn_available():
    """Return whether a compatible FlashAttention callable is available."""
    return _flash_attn_available


def is_xformers_available():
    """Is xformers available."""
    return _xformers_available


def is_triton_module_available():
    """Is triton module available."""
    return _triton_modules_available


__all__ = [
    "get_flash_attn_func",
    "is_flash_attn_available",
    "is_triton_module_available",
    "is_xformers_available",
]
