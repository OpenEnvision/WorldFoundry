"""Canonical FlowFormer++ optical-flow runtime."""

from __future__ import annotations

from importlib import import_module

__all__ = ["build_flowformer", "checkpoint_path", "get_cfg"]

_LAZY_EXPORTS = {
    "build_flowformer": (".core.FlowFormer", "build_flowformer"),
    "checkpoint_path": (".paths", "checkpoint_path"),
    "get_cfg": (".configs.submissions", "get_cfg"),
}


def __getattr__(name: str):
    """Load model-only dependencies when their public symbol is requested."""

    try:
        module_name, symbol_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    symbol = getattr(import_module(module_name, __name__), symbol_name)
    globals()[name] = symbol
    return symbol


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
