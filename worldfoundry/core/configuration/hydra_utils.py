"""Hydra/OmegaConf helpers: resolvers, class registry, and config instantiation.

LEGACY (vendored jimfan-utils lineage): frozen API kept for existing callers.
This is the ``cls``/``class`` instantiation vocabulary; do not adopt it in new
code -- use ``worldfoundry.core.configuration.lazy_config.instantiate``
(``_target_``) or ``worldfoundry.core.model_loading.factory`` instead, and
``worldfoundry.core.registry.TypedRegistry`` for new registries.
"""

import importlib.resources
import os
import sys
from copy import deepcopy
from threading import RLock

import hydra
import tree
from omegaconf import DictConfig, OmegaConf

from worldfoundry.core.utils.python.functional_utils import call_once, is_mapping, is_sequence, meta_decorator
from worldfoundry.core.observability.formatting import to_scientific_str

# ──────────────────────────────────────────────────────────────────────────
# Hydra / OmegaConf — resolvers and run-context queries
# ──────────────────────────────────────────────────────────────────────────

_CLASS_REGISTRY = {}  # for instantiation
_CLASS_REGISTRY_LOCK = RLock()
_REGISTRY_MISSING = object()


def resource_file_path(pkg_name, fname) -> str:
    """Return the absolute path to a package resource file.

    Caveat: for zip/wheel installs where the resource is materialized to a
    temporary file, that file is deleted when the internal context manager
    exits, so the returned path may not exist. Reliable only for regular
    on-disk installs (the current deployment model).
    """
    with importlib.resources.path(pkg_name, fname) as p:
        return str(p)


def print_config(cfg: DictConfig) -> None:
    """Print a resolved Hydra config to stdout."""
    # DictConfig.pretty() was removed in omegaconf 2.1; to_yaml is the
    # supported replacement.
    print(OmegaConf.to_yaml(cfg, resolve=True))


def is_hydra_initialized() -> bool:
    """Return True when a Hydra run context is active."""
    return hydra.utils.HydraConfig.initialized()


def hydra_config():
    """Return the active Hydra config singleton, or None outside a Hydra run."""
    # https://github.com/facebookresearch/hydra/issues/377
    # HydraConfig() is a singleton
    if is_hydra_initialized():
        return hydra.utils.HydraConfig().cfg.hydra
    else:
        return None


def hydra_override_arg_list() -> list[str]:
    """Return CLI override strings from the active Hydra run, e.g. ``["lr=0.2"]``."""
    if is_hydra_initialized():
        return hydra_config().overrides.task
    else:
        return []


def hydra_override_name() -> str:
    """Return Hydra's override-derived job dirname, or an empty string."""
    if is_hydra_initialized():
        return hydra_config().job.override_dirname
    else:
        return ""


def hydra_original_dir(*subpaths) -> str:
    """Join *subpaths* under Hydra's original working directory."""
    return os.path.join(hydra.utils.get_original_cwd(), *subpaths)


@call_once(on_second_call="noop")
def register_omegaconf_resolvers() -> None:
    """Register custom OmegaConf resolvers used by WorldFoundry configs."""
    import numpy as np

    OmegaConf.register_new_resolver("scientific", lambda v, i=0: to_scientific_str(v, i))
    OmegaConf.register_new_resolver("_optional", lambda v: f"_{v}" if v else "")
    OmegaConf.register_new_resolver("optional_", lambda v: f"{v}_" if v else "")
    OmegaConf.register_new_resolver("_optional_", lambda v: f"_{v}_" if v else "")
    OmegaConf.register_new_resolver("__optional", lambda v: f"__{v}" if v else "")
    OmegaConf.register_new_resolver("optional__", lambda v: f"{v}__" if v else "")
    OmegaConf.register_new_resolver("__optional__", lambda v: f"__{v}__" if v else "")
    OmegaConf.register_new_resolver("iftrue", lambda cond, v_default: cond if cond else v_default)
    OmegaConf.register_new_resolver("ifelse", lambda cond, v1, v2="": v1 if cond else v2)
    OmegaConf.register_new_resolver("ifequal", lambda query, key, v1, v2: v1 if query == key else v2)
    OmegaConf.register_new_resolver("intbool", lambda cond: 1 if cond else 0)
    OmegaConf.register_new_resolver("mult", lambda *x: np.prod(x).tolist())
    OmegaConf.register_new_resolver("add", lambda *x: sum(x))
    OmegaConf.register_new_resolver("div", lambda x, y: x / y)
    OmegaConf.register_new_resolver("intdiv", lambda x, y: x // y)

    # try each key until the key exists. Useful for multiple classes that have different
    # names for the same key
    def _try_key(cfg, *keys):
        """Return the first present key; used when sibling classes name the same field differently."""

        for k in keys:
            if k in cfg:
                return cfg[k]
        raise KeyError(f"no key in {keys} is valid")

    OmegaConf.register_new_resolver("trykey", _try_key)
    # replace `resnet.gn.ws` -> `resnet_gn_ws`, because omegaconf doesn't support
    # keys with dots. Useful for generating run name with dots
    OmegaConf.register_new_resolver("underscore_to_dots", lambda s: s.replace("_", "."))

    def _no_instantiate(cfg):
        """Mark a subtree so :func:`instantiate` returns the config mapping as-is."""

        cfg = deepcopy(cfg)
        cfg[_NO_INSTANTIATE] = True
        return cfg

    OmegaConf.register_new_resolver("no_instantiate", _no_instantiate)


# ──────────────────────────────────────────────────────────────────────────
# Instantiation — legacy cls/class vocabulary; new code uses lazy_config
# ──────────────────────────────────────────────────────────────────────────


def _registry_put_many(bindings) -> None:
    """Atomically add *bindings*, refusing to rebind names to other callables."""
    pending = dict(bindings)
    with _CLASS_REGISTRY_LOCK:
        for name, class_type in pending.items():
            existing = _CLASS_REGISTRY.get(name, _REGISTRY_MISSING)
            if existing is not _REGISTRY_MISSING and existing is not class_type:
                raise ValueError(
                    f"config_utils class registry: {name!r} is already bound to {existing!r}, "
                    f"cannot rebind to {class_type!r}"
                )
        _CLASS_REGISTRY.update(pending)


def _registry_put(name, class_type) -> None:
    """Insert one name into the legacy registry."""
    _registry_put_many(((name, class_type),))


def register_callable(name, class_type) -> None:
    """Register a callable in the class registry under *name*."""
    if isinstance(class_type, str):
        class_type, name = name, class_type
    assert callable(class_type)
    _registry_put(name, class_type)


@meta_decorator
def register_class(cls, alias=None):
    """Decorator that registers a class (and optional aliases) for config instantiation."""
    assert callable(cls)
    names = [cls.__name__]
    if alias:
        assert is_sequence(alias)
        names.extend(str(a) for a in alias)
    _registry_put_many((name, cls) for name in names)
    return cls


def omegaconf_to_dict(cfg, resolve: bool = True, enum_to_str: bool = False):
    """Convert nested OmegaConf containers to plain Python dicts and lists."""
    kw = dict(resolve=resolve, enum_to_str=enum_to_str)
    if OmegaConf.is_config(cfg):
        return OmegaConf.to_container(cfg, **kw)
    elif is_sequence(cfg):
        return type(cfg)(omegaconf_to_dict(c, **kw) for c in cfg)
    elif is_mapping(cfg):
        return {k: omegaconf_to_dict(c, **kw) for k, c in cfg.items()}
    else:
        return cfg


def omegaconf_save(cfg, *paths: str, resolve: bool = True) -> None:
    """Save an OmegaConf config to YAML at the joined *paths*."""
    from worldfoundry.core.io.filesystem.file_utils import f_join

    OmegaConf.save(cfg, f_join(*paths), resolve=resolve)


def get_class(path):
    """Resolve a class from the registry or from a fully qualified import path."""
    with _CLASS_REGISTRY_LOCK:
        class_type = _CLASS_REGISTRY.get(path, _REGISTRY_MISSING)
    if class_type is not _REGISTRY_MISSING:
        return class_type
    assert "." in path, f"Because {path} is not found in class registry, it must be a full module path"
    try:
        from importlib import import_module

        module_path, _, class_name = path.rpartition(".")
        mod = import_module(module_path)
        try:
            class_type = getattr(mod, class_name)
        except AttributeError:
            raise ImportError("Class {} is not in module {}".format(class_name, module_path))
        return class_type
    except ValueError as e:
        print("Error initializing class " + path, file=sys.stderr)
        raise e


_DELETE_ARG = "__delete__"
_NO_INSTANTIATE = "__no_instantiate__"  # return config as-is
_OMEGA_MISSING = "???"


def _get_instantiate_params(cfg, kwargs=None):
    """Split ``cls``/``class`` out; ``???`` must be filled from *kwargs* or raise."""

    params = cfg
    f_args, f_kwargs = (), {}
    for k, value in params.items():
        if k in ["cls", "class"]:
            continue
        elif k == "*args":
            assert is_sequence(value), '"*args" value must be a sequence'
            f_args = list(value)
            continue
        if value == _OMEGA_MISSING:
            if kwargs and k in kwargs:
                value = kwargs[k]
            else:
                raise ValueError(f'Missing required keyword arg "{k}" in cfg: {cfg}')
        if value == _DELETE_ARG:
            continue
        else:
            f_kwargs[k] = value
    return f_args, f_kwargs


def _instantiate_single(cfg):
    """Instantiate one mapping; ``__no_instantiate__`` short-circuits to a deep copy."""

    if is_mapping(cfg) and ("cls" in cfg or "class" in cfg):
        assert bool("cls" in cfg) != bool("class" in cfg), (
            'to instantiate from config, one and only one of "cls" or "class" key should be provided'
        )
        if _NO_INSTANTIATE in cfg:
            no_instantiate = cfg.pop(_NO_INSTANTIATE)
            if no_instantiate:
                cfg = deepcopy(cfg)
                return cfg
            else:
                return _instantiate_single(cfg)

        cls = cfg.get("class", cfg.get("cls"))
        args, kwargs = _get_instantiate_params(cfg)
        try:
            class_type = get_class(cls)
            return class_type(*args, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Error instantiating {cls}: {e}") from e
    else:
        return None


def instantiate(_cfg_, **kwargs):
    """Recursively instantiate dicts that contain a ``cls`` or ``class`` key."""
    assert OmegaConf.is_config(_cfg_) or isinstance(_cfg_, (list, tuple)) or is_mapping(_cfg_), (
        f'"cfg" must be a dict, list, tuple, or OmegaConf config to be instantiated. Current its type is {type(_cfg_)}'
    )

    _cfg_ = omegaconf_to_dict(_cfg_, resolve=True)

    if kwargs:
        if is_mapping(_cfg_):
            _cfg_ = _cfg_.copy()
            _cfg_.update(kwargs)
            _cfg_ = {k: v for k, v in _cfg_.items() if v != _DELETE_ARG}
        else:
            raise RuntimeError(f"**kwargs specified, but the top-level cfg is not a dict. It has type {type(_cfg_)}")

    return tree.traverse(_instantiate_single, _cfg_, top_down=False)
