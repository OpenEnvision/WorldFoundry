# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Loading and serialization for inference-oriented lazy configurations.

Trust boundary: ``.py`` configs are **trusted code**. ``LazyConfig.load``
executes them with ``exec`` (detectron2 LazyConfig design) — the same as
running that file. Relative imports of other ``.py`` configs are also
``exec``'d. YAML/YML configs use ``yaml.safe_load`` and cannot construct
arbitrary Python objects; anything that needs Python objects belongs in a
``.py`` file you already trust.
"""

from __future__ import annotations

import ast
import builtins
import importlib.machinery
import importlib.util
import inspect
import uuid
from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict as dataclass_asdict
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any

import attrs
import yaml
from omegaconf import DictConfig, ListConfig, OmegaConf

from .lazy_call import get_default_params

# ──────────────────────────────────────────────────────────────────────────
# Trusted exec helpers — unique package names so relative imports stay isolated
# ──────────────────────────────────────────────────────────────────────────


def _cast_to_config(value: Any) -> Any:
    """Wrap a plain dict so nested LazyCall objects survive OmegaConf storage."""

    return DictConfig(value, flags={"allow_objects": True}) if isinstance(value, dict) else value


def _validate_python(path: Path) -> None:
    """Parse before ``exec`` so a syntax error names the config file, not ``<string>``."""

    try:
        ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        raise SyntaxError(f"Config file {path} has invalid Python syntax.") from exc


_CONFIG_PACKAGE_PREFIX = "worldfoundry._lazy_config_"


def _module_name(path: Path) -> str:
    """Build a unique ``__package__`` so two configs with the same stem do not collide."""

    return f"{_CONFIG_PACKAGE_PREFIX}{path.stem}_{uuid.uuid4().hex[:8]}"


def _config_builtins() -> dict[str, Any]:
    """Create an exec-local builtins mapping with config-relative imports.

    Import statements resolve ``__import__`` through the executing globals'
    builtins mapping. A copied mapping scopes this hook to the trusted config
    execution and avoids mutating the host process for unrelated threads.
    """

    original_import = builtins.__import__
    config_builtins = dict(vars(builtins))

    def patched_import(name, globals=None, locals=None, fromlist=(), level=0):
        """Resolve relative imports as sibling ``.py`` configs; leave absolute imports alone.

        Only packages whose name starts with :data:`_CONFIG_PACKAGE_PREFIX`
        take this path, so a config ``from torch import nn`` still hits the
        real interpreter import.
        """

        package = "" if globals is None else globals.get("__package__", "") or ""
        if level and package.startswith(_CONFIG_PACKAGE_PREFIX):
            if not name:
                raise ImportError("Relative config imports must name a Python config file.")
            source = Path(globals["__file__"])
            target = source.parent
            for _ in range(level - 1):
                target = target.parent
            target = target.joinpath(*name.split(".")).with_suffix(".py")
            if not target.is_file():
                raise ImportError(f"Relative config import does not exist: {target}")
            _validate_python(target)
            spec = importlib.machinery.ModuleSpec(_module_name(target), loader=None, origin=str(target))
            module = importlib.util.module_from_spec(spec)
            module.__file__ = str(target)
            module.__package__ = spec.name
            module.__dict__["__builtins__"] = config_builtins
            exec(compile(target.read_text(encoding="utf-8"), str(target), "exec"), module.__dict__)
            for imported_name in fromlist:
                if imported_name == "*":
                    continue
                if imported_name not in module.__dict__:
                    raise ImportError(f"cannot import name {imported_name!r} from config {target}")
                module.__dict__[imported_name] = _cast_to_config(module.__dict__[imported_name])
            return module
        return original_import(name, globals, locals, fromlist=fromlist, level=level)

    config_builtins["__import__"] = patched_import
    return config_builtins


def _plain_config(value: Any) -> Any:
    """Flatten OmegaConf / attrs / dataclass trees into YAML-safe containers.

    Values that ``yaml.safe_dump`` cannot represent become ``str(value)``
    so a save never fails because a live callable leaked into the tree.
    """

    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True, enum_to_str=False)
    if attrs.has(type(value)):
        return attrs.asdict(value, recurse=True)
    if is_dataclass(value) and not isinstance(value, type):
        return dataclass_asdict(value)
    if isinstance(value, Mapping):
        return {key: _plain_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_config(item) for item in value]
    try:
        yaml.safe_dump(value)
    except Exception:
        return str(value)
    return value


def _sort_recursive(value: Any) -> Any:
    """Sort mapping keys so two dumps of the same config compare as equal text."""

    if isinstance(value, dict):
        return OrderedDict((key, _sort_recursive(item)) for key, item in sorted(value.items()))
    if isinstance(value, list):
        return [_sort_recursive(item) for item in value]
    return value


class LazyConfig:
    """Load local Python/YAML lazy configs and save resolved inference configs.

    Trust boundary: ``.py`` configs are executed (detectron2 LazyConfig
    design) -- only load Python configs you would run as code. YAML configs
    are parsed with ``yaml.safe_load`` and cannot instantiate arbitrary
    Python objects; ``save_yaml`` only ever emits plain YAML, so framework
    round-trips are unaffected. Configs needing Python objects belong in
    ``.py`` files.
    """

    @staticmethod
    def load_rel(filename: str, keys: str | tuple[str, ...] | None = None):
        """Load a config relative to the *caller's* file, not the process cwd.

        ``inspect.stack()[1]`` is the caller. ``<string>`` (``exec`` without
        a filename) cannot resolve a sibling path and is rejected.
        """

        caller = Path(inspect.stack()[1].filename)
        if str(caller) == "<string>":
            raise RuntimeError("LazyConfig.load_rel cannot resolve a caller for <string>.")
        return LazyConfig.load(str(caller.parent / filename), keys)

    @staticmethod
    def load(filename: str, keys: str | tuple[str, ...] | None = None):
        """Load a trusted ``.py`` config via ``exec``, or YAML via ``safe_load``.

        A ``.py`` result drops names starting with ``_`` and keeps only
        dict / OmegaConf values so helper functions defined in the file
        do not become config keys. ``keys`` selects a subset after load.

        Raises:
            ValueError: Suffix is not ``.py`` / ``.yaml`` / ``.yml``.
            SyntaxError: The Python config does not parse.
        """

        path = Path(filename).expanduser().resolve()
        if path.suffix not in {".py", ".yaml", ".yml"}:
            raise ValueError(f"Config file must be Python or YAML: {path}")

        if path.suffix == ".py":
            _validate_python(path)
            namespace = {
                "__builtins__": _config_builtins(),
                "__file__": str(path),
                "__package__": _module_name(path),
            }
            exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
            result: Any = namespace
        else:
            # safe_load: YAML is data, not code. ``save_yaml`` emits plain
            # YAML only, and no repository config uses python object tags;
            # anything needing Python objects must be a ``.py`` config.
            result = OmegaConf.create(yaml.safe_load(path.read_text(encoding="utf-8")), flags={"allow_objects": True})

        if keys is not None:
            if isinstance(keys, str):
                return _cast_to_config(result[keys])
            return tuple(_cast_to_config(result[key]) for key in keys)
        if path.suffix == ".py":
            result = DictConfig(
                {
                    name: _cast_to_config(value)
                    for name, value in result.items()
                    if not name.startswith("_") and isinstance(value, (dict, DictConfig, ListConfig))
                },
                flags={"allow_objects": True},
            )
        return result

    @staticmethod
    def save_yaml(config: Any, filename: str | Path) -> str:
        """Write a resolved, key-sorted YAML snapshot that cannot ``exec``.

        A failed ``deepcopy`` is ignored so an un-copyable live object still
        serializes through :func:`_plain_config` rather than aborting the save.
        """

        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            config = deepcopy(config)
        except Exception:
            pass
        value = _sort_recursive(_plain_config(config))
        yaml.add_representer(
            OrderedDict,
            lambda dumper, data: dumper.represent_mapping("tag:yaml.org,2002:map", data.items()),
        )
        path.write_text(yaml.dump(value, default_flow_style=False), encoding="utf-8")
        return str(path)


__all__ = ["LazyConfig", "get_default_params"]
