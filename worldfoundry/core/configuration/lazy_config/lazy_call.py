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

"""Deferred constructor calls for editable inference configuration trees.

:class:`LazyCall` wraps a callable (or its import path) so that invoking
it records ``_target_`` plus keyword arguments in an OmegaConf
``DictConfig`` instead of constructing the object. The resulting tree is
mutable — later code can override ``out_channels`` or swap a nested
module — and :func:`~worldfoundry.core.configuration.lazy_config.instantiate.instantiate`
materializes it when a real instance is needed.

This is the Detectron2 / fvcore LazyCall contract used by Cosmos-family
inference configs. Positional arguments are intentionally unsupported at
call time so the stored mapping stays keyword-only and editable.
"""

import collections.abc as abc
import inspect
from dataclasses import is_dataclass

import attrs
from omegaconf import DictConfig

from worldfoundry.core.configuration.lazy_config.registry import convert_target_to_string

__all__ = ["LazyCall"]

# ──────────────────────────────────────────────────────────────────────────
# Defaults + call recording — inspect never imports a string _target_
# ──────────────────────────────────────────────────────────────────────────


def get_default_params(cls_or_func):
    """Return the target's keyword defaults, or ``{}`` when they can't be inspected.

    Classes are callable, so both functions and classes take the first branch
    (``inspect.signature`` already resolves a class to its ``__init__``).
    String import targets are *not* resolved here: their defaults are
    intentionally skipped so that building a config never imports the target
    module.
    """
    if callable(cls_or_func):
        signature = inspect.signature(cls_or_func)
    else:
        return {}
    params = signature.parameters
    default_params = {
        name: param.default for name, param in params.items() if param.default is not inspect.Parameter.empty
    }
    return default_params


# Used by tests to enforce conversion of target to string (module-level flag,
# not a ClassVar; the previous ClassVar annotation is only meaningful inside a
# class body).
_CONVERT_TARGET_TO_STRING = False


class LazyCall:
    """
    Wrap a callable so that when it's called, the call will not be executed,
    but returns a dict that describes the call.

    LazyCall object has to be called with only keyword arguments. Positional
    arguments are not yet supported.

    Example::
        from worldfoundry.core.configuration import LazyCall, instantiate

        layer_cfg = LazyCall(nn.Conv2d)(in_channels=32, out_channels=32)
        layer_cfg.out_channels = 64   # can edit it afterwards
        layer = instantiate(layer_cfg)
    """

    def __init__(self, target):
        """Bind a callable, import string, or mapping as the deferred target.

        Args:
            target: Callable to instantiate later, its importable string name,
                or an existing target mapping.

        Raises:
            TypeError: ``target`` cannot describe a callable.
        """
        if not (callable(target) or isinstance(target, (str, abc.Mapping))):
            raise TypeError(f"target of LazyCall must be a callable or defines a callable! Got {target}")
        self._target = target

    def __call__(self, **kwargs):
        """Return an editable config instead of invoking the target.

        Args:
            **kwargs: Named constructor arguments. Target defaults are copied
                first, then explicit values override them.

        Returns:
            OmegaConf ``DictConfig`` containing ``_target_`` and the eventual
            constructor arguments.

        Notes:
            Positional arguments are intentionally unsupported at this stage;
            they can be supplied later to ``instantiate`` when necessary.
        """
        if _CONVERT_TARGET_TO_STRING or is_dataclass(self._target) or attrs.has(self._target):
            # omegaconf object cannot hold dataclass type
            # https://github.com/omry/omegaconf/issues/784
            target = convert_target_to_string(self._target)
        else:
            target = self._target
        kwargs["_target_"] = target

        _final_params = get_default_params(self._target)
        _final_params.update(kwargs)

        return DictConfig(content=_final_params, flags={"allow_objects": True})
