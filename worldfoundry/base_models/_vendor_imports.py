"""Shared sys.path helpers for vendored model wrappers under ``base_models``.

Vendored trees expose upstream top-level packages (``dust3r``, ``croco``,
``mast3r``, ``pi3``, ``open_flamingo``, ...) by prepending their source
directories to ``sys.path``. This module centralises the two safe primitives
so wrappers stop hand-rolling the ``remove + insert(0)`` preemption pattern:

- :func:`prepend_import_path` inserts idempotently and never displaces an
  entry that is already present, so import precedence cannot be reshuffled by
  whichever wrapper happens to be imported last.
- :func:`assert_top_level_not_shadowed` fails fast when a conflicting copy of
  a top-level module has already been imported from outside the expected
  vendored root, instead of letting ``sys.modules`` silently serve the wrong
  version.

Deliberately stdlib-only: wrappers must be importable without pulling any
heavy dependency.

Remaining [VI-14] collisions that this helper does **not** fail-fast on
(asserting them would hard-fail live pipeline combinations):

- ``dust3r`` / ``croco``: canonical ``general_3d/dust3r``, MonST3R's fork,
  and Stable Virtual Camera's older third_party snapshot all expose the
  same top-level names. Canonical + MASt3R wrappers already call
  :func:`assert_top_level_not_shadowed`; the other two copies still prepend
  their own trees (see [VI-8]).
- ``opensora``: Open-Sora and Open-Sora-Plan are different upstreams with
  the same package name.
- ``utils`` / ``models``: generic names exposed by CroCo and several
  synthesis/eval runtimes. ``vmem`` and ``fantasy_world`` additionally
  evict foreign ``sys.modules`` entries on init.
"""

from __future__ import annotations

import sys
from pathlib import Path

__all__ = ["assert_top_level_not_shadowed", "prepend_import_path"]

# Document-only registry of unresolved top-level collisions ([VI-14]).
# Do not wire these into assert_top_level_not_shadowed: several live trees
# legitimately expose the same names, and a global fail-fast would break
# current multi-model pipelines. Dedup is [VI-8]/[VI-9], not a one-line guard.
UNRESOLVED_TOP_LEVEL_COLLISIONS = (
    "dust3r",
    "croco",
    "opensora",
    "utils",
    "models",
)


def prepend_import_path(path: str | Path) -> None:
    """Idempotently put *path* at the front of ``sys.path``.

    Position 0 lets the vendored copy win over site-packages, but an entry
    that is already present is left untouched (no remove + re-insert).
    """
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)


def _module_origins(module: object) -> list[Path]:
    origins: list[Path] = []
    module_file = getattr(module, "__file__", None)
    if module_file:
        origins.append(Path(str(module_file)).resolve())
    module_paths = getattr(module, "__path__", None)
    if module_paths is not None:
        origins.extend(Path(str(entry)).resolve() for entry in module_paths)
    return origins


def assert_top_level_not_shadowed(name: str, expected_root: str | Path) -> None:
    """Fail fast if *name* is already imported from outside *expected_root*.

    Several vendored trees expose identical top-level names; once a foreign
    copy sits in ``sys.modules``, every later ``import name`` reuses it and the
    failure surfaces far away (isinstance mismatches, missing attributes).
    Raising a clear ImportError here beats debugging that.
    """
    module = sys.modules.get(name)
    if module is None:
        return
    root = Path(expected_root).resolve()
    origins = _module_origins(module)
    if origins and any(origin == root or root in origin.parents for origin in origins):
        return
    origin_text = ", ".join(str(origin) for origin in origins) or "<unknown origin>"
    raise ImportError(
        f"Top-level module {name!r} is already imported from {origin_text}, "
        f"which is not the vendored copy under {root}. Two integrations expose "
        f"the same top-level name, and importing this wrapper now would silently "
        f"reuse the wrong copy. Run the conflicting integrations in separate "
        f"processes, or import this wrapper before the conflicting one."
    )
