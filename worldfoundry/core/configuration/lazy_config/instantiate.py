# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Recursive object construction from LazyCall / ``_target_`` config trees.

WorldFoundry inference configs are *deferred constructor graphs*: a
:class:`~worldfoundry.core.configuration.lazy_config.lazy_call.LazyCall`
records ``_target_`` plus keyword arguments instead of building the object.
:func:`instantiate` walks that graph and materializes it.

Resolution rules:

- A mapping with ``_target_`` is treated as a constructor call. Nested
  values are instantiated first unless ``_recursive_=False``. Extra
  positional/keyword arguments override the stored ones.
- OmegaConf ``ListConfig`` / plain lists instantiate element-wise so
  layers such as ``nn.Sequential`` can take a list of objects.
- Structured OmegaConf configs backed by a dataclass or attrs class are
  converted through the LazyCall-aware :func:`to_object` patch (a
  ``_target_`` node is *not* flattened to a dict).
- Anything else is returned unchanged.

:func:`dump_dataclass` is the inverse for dataclass *instances*: it
emits a ``_target_`` mapping that :func:`instantiate` can rebuild.
"""

import collections.abc as abc
import dataclasses
import logging
from typing import Any

import attrs

from worldfoundry.core.configuration.lazy_config.registry import _convert_target_to_string, locate

__all__ = ["dump_dataclass", "instantiate"]

# ──────────────────────────────────────────────────────────────────────────
# Inverse + walk — dump_dataclass emits _target_; instantiate consumes it
# ──────────────────────────────────────────────────────────────────────────


def is_dataclass_or_attrs(target):
    """Return True when *target* is a dataclass or attrs class/instance."""
    return dataclasses.is_dataclass(target) or attrs.has(target)


def dump_dataclass(obj: Any):
    """Serialize a dataclass instance into a LazyCall-style ``_target_`` dict.

    Nested dataclass fields and dataclass items inside lists/tuples are
    dumped recursively. Non-dataclass values are copied as-is. The result
    is meant to be passed back to :func:`instantiate`, not used as a
    general-purpose JSON encoder.

    Args:
        obj: A dataclass *instance* (not a class).

    Returns:
        Mapping with ``_target_`` set to the type's import path plus one
        entry per dataclass field.

    Raises:
        AssertionError: *obj* is a type or is not a dataclass instance.
    """
    assert dataclasses.is_dataclass(obj) and not isinstance(obj, type), (
        "dump_dataclass() requires an instance of a dataclass."
    )
    ret = {"_target_": _convert_target_to_string(type(obj))}
    for f in dataclasses.fields(obj):
        v = getattr(obj, f.name)
        if dataclasses.is_dataclass(v):
            v = dump_dataclass(v)
        if isinstance(v, (list, tuple)):
            v = [dump_dataclass(x) if dataclasses.is_dataclass(x) else x for x in v]
        ret[f.name] = v
    return ret


def instantiate(cfg, *args, **kwargs):
    """Materialize a LazyCall / Hydra-style config into a live object.

    Faster than ``hydra.utils.instantiate(..., _convert_=all)`` for the
    common case of a mapping with ``_target_`` (see facebookresearch/hydra
    #1200). Callers may pass extra ``*args`` / ``**kwargs``; keyword
    overrides win over stored config keys.

    Args:
        cfg: A ``_target_`` mapping, OmegaConf container, list, or any
            already-materialized value.
        *args: Extra positional arguments forwarded to the target.
        **kwargs: Extra keyword arguments that override stored fields.

    Returns:
        The constructed object, a list/ListConfig of constructed objects,
        a structured dataclass/attrs instance, or *cfg* unchanged when it
        is not a constructor description.
    """
    from omegaconf import DictConfig, ListConfig

    if isinstance(cfg, ListConfig):
        lst = [instantiate(x) for x in cfg]
        return ListConfig(lst, flags={"allow_objects": True})
    if isinstance(cfg, list):
        # Specialize for list, because many classes take
        # list[objects] as arguments, such as ResNet, DatasetMapper
        return [instantiate(x) for x in cfg]

    # If input is a DictConfig backed by dataclasses (i.e. omegaconf's structured config),
    # instantiate it to the actual dataclass.
    if isinstance(cfg, DictConfig) and is_dataclass_or_attrs(cfg._metadata.object_type):
        # Call the package-local to_object directly instead of relying on the
        # global ``OmegaConf.to_object`` monkey-patch installed by
        # ``lazy_config/__init__`` (identical behavior; see CF-3).
        from worldfoundry.core.configuration.lazy_config.omegaconf_patch import to_object

        return to_object(cfg)

    if isinstance(cfg, abc.Mapping) and "_target_" in cfg:
        # conceptually equivalent to hydra.utils.instantiate(cfg) with _convert_=all,
        # but faster: https://github.com/facebookresearch/hydra/issues/1200
        is_recursive = getattr(cfg, "_recursive_", True)
        if is_recursive:
            cfg = {k: instantiate(v) for k, v in cfg.items()}
        else:
            cfg = {k: v for k, v in cfg.items()}
        # pop the _recursive_ key to avoid passing it as a parameter
        if "_recursive_" in cfg:
            cfg.pop("_recursive_")
        cls = cfg.pop("_target_")
        cls = instantiate(cls)

        if isinstance(cls, str):
            cls_name = cls
            cls = locate(cls_name)
            assert cls is not None, cls_name
        else:
            try:
                cls_name = cls.__module__ + "." + cls.__qualname__
            except Exception:
                # target could be anything, so the above could fail
                cls_name = str(cls)
        assert callable(cls), f"_target_ {cls} does not define a callable object"
        try:
            # override config with kwargs
            instantiate_kwargs = {}
            instantiate_kwargs.update(cfg)
            instantiate_kwargs.update(kwargs)
            return cls(*args, **instantiate_kwargs)
        except TypeError:
            logger = logging.getLogger(__name__)
            logger.error(f"Error when instantiating {cls_name}!")
            raise
    return cfg  # return as-is if don't know what to do
