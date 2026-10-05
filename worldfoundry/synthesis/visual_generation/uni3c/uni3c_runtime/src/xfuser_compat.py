"""Optional xFuser bridge for Uni3C's sequence-parallel paths."""

from __future__ import annotations

from typing import Any

try:
    from xfuser.core.distributed import (
        get_sequence_parallel_rank as _get_sequence_parallel_rank,
    )
    from xfuser.core.distributed import (
        get_sequence_parallel_world_size as _get_sequence_parallel_world_size,
    )
    from xfuser.core.distributed import (
        get_sp_group as _get_sp_group,
    )
    from xfuser.core.long_ctx_attention import (
        xFuserLongContextAttention as _XfuserLongContextAttention,
    )
except ImportError as exc:
    # xFuser (and its FlashAttention stack) is optional for the official
    # single-GPU CPU-offload route. Preserve the import error so a multi-GPU
    # request still fails explicitly instead of silently losing SP semantics.
    _XFUSER_IMPORT_ERROR: ImportError | None = exc
    _get_sequence_parallel_rank = None
    _get_sequence_parallel_world_size = None
    _get_sp_group = None
    _XfuserLongContextAttention = None
else:
    _XFUSER_IMPORT_ERROR = None


def get_sequence_parallel_rank() -> int:
    if _XFUSER_IMPORT_ERROR is not None:
        return 0
    return int(_get_sequence_parallel_rank())


def get_sequence_parallel_world_size() -> int:
    if _XFUSER_IMPORT_ERROR is not None:
        return 1
    return int(_get_sequence_parallel_world_size())


def require_xfuser(feature: str = "sequence-parallel inference") -> None:
    if _XFUSER_IMPORT_ERROR is None:
        return
    raise RuntimeError(
        f"Uni3C {feature} requires a working xfuser/FlashAttention installation; "
        "use the single-GPU CPU-offload path or install the optional SP runtime."
    ) from _XFUSER_IMPORT_ERROR


def get_sp_group() -> Any:
    require_xfuser()
    return _get_sp_group()


def long_context_attention(*args: Any, **kwargs: Any) -> Any:
    require_xfuser("long-context sequence-parallel attention")
    return _XfuserLongContextAttention()(*args, **kwargs)


__all__ = [
    "get_sequence_parallel_rank",
    "get_sequence_parallel_world_size",
    "get_sp_group",
    "long_context_attention",
    "require_xfuser",
]
