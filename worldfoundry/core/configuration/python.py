"""Lightweight Python config loading helpers.

Trust boundary: config files are **trusted code**. ``load_python_config``
executes the file via ``importlib`` (``exec_module``). Leaf strings may
also run ``eval(...)`` wrappers and ``${...}`` interpolation through a
restricted ``eval``. Only load paths you would run as a program; do not
point this loader at untrusted uploads.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

# ──────────────────────────────────────────────────────────────────────────
# Trusted .py configs — exec_module + restricted eval; never load untrusted files
# ──────────────────────────────────────────────────────────────────────────


class EasyDict(dict):
    """Dictionary with recursive attribute access."""

    def __init__(self, mapping: Any | None = None, **kwargs: Any) -> None:
        """Wrap nested dicts/lists so attribute access works at every level."""

        super().__init__()
        mapping = {} if mapping is None else mapping
        items = mapping.items() if hasattr(mapping, "items") else mapping
        for key, value in items:
            self[key] = self._wrap(value)
        for key, value in kwargs.items():
            self[key] = self._wrap(value)

    def __getattr__(self, name: str) -> Any:
        """Attribute lookup that raises :class:`AttributeError`, not :class:`KeyError`."""

        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        """Assign through the dict so nested values stay wrapped."""

        self[name] = self._wrap(value)

    @classmethod
    def _wrap(cls, value: Any) -> Any:
        """Recursively wrap mappings and lists; other values are left as-is."""

        if isinstance(value, dict) and not isinstance(value, EasyDict):
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap(item) for item in value]
        return value


def load_python_config(config_file: str | Path, overrides: Iterable[Any] = ()) -> EasyDict:
    """Exec a trusted ``.py`` config, apply dotted overrides, then eval leaf strings."""

    cfg = EasyDict(_module_public_values(Path(config_file)))
    merge_list(cfg, list(overrides))
    eval_leaf_values(cfg)
    return cfg


def merge_list(cfg: EasyDict, opts: list[Any]) -> EasyDict:
    """Apply ``[key, value, ...]`` overrides; unknown dotted keys raise :class:`KeyError`."""

    if len(opts) % 2 != 0:
        raise ValueError(f"Expected key/value override pairs, got {opts!r}")
    for idx in range(0, len(opts), 2):
        keys = str(opts[idx]).split(".")
        value = opts[idx + 1]
        node: Any = cfg
        for key in keys[:-1]:
            if key not in node:
                raise KeyError(f"Config key does not exist: {'.'.join(keys)}")
            node = node[key]
        if keys[-1] not in node:
            raise KeyError(f"Config key does not exist: {'.'.join(keys)}")
        node[keys[-1]] = EasyDict._wrap(value)
    return cfg


def eval_leaf_values(node: EasyDict, root: EasyDict | None = None) -> EasyDict:
    """Walk leaves and resolve ``eval(...)`` / ``${...}`` against *root*."""

    root = node if root is None else root
    for key, value in list(node.items()):
        if isinstance(value, EasyDict):
            eval_leaf_values(value, root)
        else:
            node[key] = EasyDict._wrap(_eval_string(value, root))
    return node


def _module_public_values(path: Path) -> dict[str, Any]:
    """``exec_module`` a file and deepcopy public names; always unregister the temp module."""

    module_name = f"_worldfoundry_config_{path.stem}_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load config: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        return {
            name: deepcopy(value)
            for name, value in vars(module).items()
            if not name.startswith("__") and not isinstance(value, ModuleType)
        }
    finally:
        sys.modules.pop(module_name, None)


def _restricted_eval(expression: str, root: EasyDict) -> Any:
    """Evaluate a config expression with ``d`` bound to the config root.

    ``__builtins__`` is explicitly emptied: with a bare ``{}`` globals dict
    CPython injects the full builtins module, which would let a *config
    value* reach ``__import__`` and execute arbitrary code. Config
    expressions only need ``d``-rooted lookups, literals, and operators.
    """
    return eval(expression, {"__builtins__": {}}, {"d": root})  # noqa: S307 - restricted namespace


def _is_single_whole_interpolation(value: str) -> bool:
    """Return whether the entire string is one ``${...}`` placeholder.

    Nested placeholders such as ``${TextEncoders[${text_enc}]}`` count as a
    single whole-string interpolation; ``${a}...${b}`` does not.
    """
    if not (value.startswith("${") and value.endswith("}")):
        return False
    depth = 0
    index = 0
    while index < len(value):
        if value.startswith("${", index):
            depth += 1
            index += 2
            continue
        if value[index] == "}":
            depth -= 1
            if depth == 0:
                return index == len(value) - 1
        index += 1
    return False


def _interpolate_mixed(value: str, root: EasyDict) -> str:
    """Substitute every ``${...}`` placeholder inside literal text.

    The pre-fix greedy rewrite turned such strings into invalid Python and
    crashed with ``SyntaxError``, so no existing config can depend on the
    old behavior; string-join interpolation is the obvious intent.
    """
    result: list[str] = []
    index = 0
    while index < len(value):
        start = value.find("${", index)
        if start == -1:
            result.append(value[index:])
            break
        result.append(value[index:start])
        depth = 0
        end = None
        position = start
        while position < len(value):
            if value.startswith("${", position):
                depth += 1
                position += 2
                continue
            if value[position] == "}":
                depth -= 1
                if depth == 0:
                    end = position
                    break
            position += 1
        if end is None:  # unterminated placeholder: keep the tail literal
            result.append(value[start:])
            break
        result.append(str(_eval_string(value[start : end + 1], root)))
        index = end + 1
    return "".join(result)


def _eval_string(value: Any, root: EasyDict) -> Any:
    """Resolve one leaf: ``eval(...)``, whole-string ``${...}``, mixed interpolations, or literal."""

    if not isinstance(value, str):
        return value
    if value.startswith("eval(") and value.endswith(")"):
        return _restricted_eval(value[5:-1], root)

    if _is_single_whole_interpolation(value):
        # Iterative greedy rewrite handles nested lookups like
        # ``${TextEncoders[${text_enc}]}`` and preserves the resolved
        # object's type (the placeholder may name a dict or int).
        original = value
        resolved = re.sub(r"\${(.*)}", r"d.\1", original)
        while resolved != original:
            original = resolved
            resolved = re.sub(r"\${(.*)}", r"d.\1", original)
        return _restricted_eval(resolved, root)
    if "${" in value:
        return _interpolate_mixed(value, root)

    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


__all__ = ["EasyDict", "eval_leaf_values", "load_python_config", "merge_list"]
