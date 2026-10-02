"""Guard the lazy ``_EXPORT_MODULES`` facades against silent rot.

``worldfoundry.core`` resolves every public name through ``__getattr__``, so a
table entry pointing at a moved or renamed target still lets ``import
worldfoundry.core`` succeed -- the failure only surfaces when someone touches
that one symbol. These tests force every entry to resolve so a relocation that
forgets to update the table fails here instead of in a caller.
"""

from __future__ import annotations

from importlib import import_module

import pytest


def _table(module_name: str) -> dict[str, str]:
    module = import_module(module_name)
    return dict(module._EXPORT_MODULES)


def _rot(error: Exception) -> bool:
    """Whether *error* means the table is stale rather than an optional dep missing.

    Several targets pull optional third-party stacks (imageio, torch, hydra).
    A missing one of those says nothing about the table; a missing
    ``worldfoundry.*`` module or absent attribute does.
    """
    if isinstance(error, ModuleNotFoundError):
        return (error.name or "").startswith("worldfoundry")
    return True


class TestCoreFacade:
    def test_all_matches_export_table(self) -> None:
        import worldfoundry.core as core

        assert core.__all__ == sorted(core._EXPORT_MODULES)

    def test_every_export_resolves(self) -> None:
        import worldfoundry.core as core

        unresolved: dict[str, str] = {}
        for name, target in core._EXPORT_MODULES.items():
            try:
                getattr(core, name)
            except Exception as error:  # noqa: BLE001 - report every broken entry at once
                if _rot(error):
                    unresolved[name] = f"{target}: {type(error).__name__}: {error}"
        assert not unresolved, f"stale worldfoundry.core export entries: {unresolved}"

    def test_export_targets_are_core_modules(self) -> None:
        import worldfoundry.core as core

        foreign = {
            name: target for name, target in core._EXPORT_MODULES.items() if not target.startswith("worldfoundry.core")
        }
        assert not foreign, f"core facade points outside core: {foreign}"


@pytest.mark.parametrize(
    "package",
    [
        "worldfoundry.core.attention",
        "worldfoundry.core.io",
        "worldfoundry.core.model_loading",
        "worldfoundry.core.nn",
        "worldfoundry.core.safety",
        "worldfoundry.core.utils",
        "worldfoundry.core.vram",
    ],
)
class TestSubpackageFacades:
    def test_every_export_resolves(self, package: str) -> None:
        module = import_module(package)
        unresolved: dict[str, str] = {}
        for name, target in _table(package).items():
            try:
                getattr(module, name)
            except Exception as error:  # noqa: BLE001
                if _rot(error):
                    unresolved[name] = f"{target}: {type(error).__name__}: {error}"
        assert not unresolved, f"stale {package} export entries: {unresolved}"

    def test_all_matches_export_table(self, package: str) -> None:
        module = import_module(package)
        expected = sorted({*getattr(module, "_SUBMODULES", {}), *_table(package)})
        assert sorted(module.__all__) == expected
