"""Process-wide guard for thread-unsafe third-party lazy imports.

Some optional libraries expose classes through a mutable module-level lazy
loader.  Resolving two such exports concurrently can expose a partially
initialized module even though Python's individual submodule imports are
locked.  Keep only dependency discovery/imports inside this guard; checkpoint
I/O and inference must remain outside it so independent GPUs can run in
parallel.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from threading import RLock

_THIRD_PARTY_LAZY_IMPORT_LOCK = RLock()


@contextmanager
def third_party_lazy_import_guard() -> Iterator[None]:
    """Serialize process-wide lazy export resolution and remain re-entrant."""

    with _THIRD_PARTY_LAZY_IMPORT_LOCK:
        yield


__all__ = ["third_party_lazy_import_guard"]
