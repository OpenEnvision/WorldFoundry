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

"""Map config-file import strings to live callables, and the inverse.

LazyCall stores ``_target_`` as either a live object or an import path
such as ``torch.nn.Conv2d``. :func:`locate` resolves that string without
the caller knowing which package owns the symbol; :func:`convert_target_to_string`
is the inverse used when a dataclass/attrs type cannot live inside
OmegaConf.

:func:`locate` tries :func:`pydoc.locate` first, then Hydra's private
``_locate`` for cases pydoc misses (for example ``torch.optim.sgd.SGD``).
:func:`convert_target_to_string` prefers the shortest prefix that still
resolves to the same object so configs stay stable when a class moves
from ``pkg._impl.Foo`` to ``pkg.Foo``.
"""

import inspect
import pydoc
from typing import Any

__all__ = ["locate", "convert_target_to_string"]

# ──────────────────────────────────────────────────────────────────────────
# Optional fvcore Registry — keep the name for configs that still import it
# ──────────────────────────────────────────────────────────────────────────

try:
    from fvcore.common.registry import Registry  # for backward compatibility.

    __all__ += ["Registry"]
except Exception:
    pass


# ──────────────────────────────────────────────────────────────────────────
# Bidirectional locate — compress paths so a moved class keeps its _target_
# ──────────────────────────────────────────────────────────────────────────


def convert_target_to_string(t: Any) -> str:
    """Return the shortest import path that :func:`locate` maps back to *t*.

    Classmethods are encoded as ``ClassName.method`` under the class
    module. Ordinary callables use ``__module__`` + ``__qualname__``.
    Intermediate prefixes are tried so ``pkg.sub._impl.Foo`` can shrink
    to ``pkg.sub.Foo`` when that alias exists.

    Args:
        t: Any object with ``__module__`` and ``__qualname__`` (or a
            classmethod bound on a class).
    """
    if hasattr(t, "__self__") and inspect.isclass(t.__self__):
        # classmethod
        cls = t.__self__
        module = cls.__module__
        qualname = f"{cls.__name__}.{t.__name__}"
    else:
        module = t.__module__
        qualname = t.__qualname__

    # Compress the path to this object, e.g. ``module.submodule._impl.class``
    # may become ``module.submodule.class``, if the later also resolves to the same
    # object. This simplifies the string, and also is less affected by moving the
    # class implementation.
    module_parts = module.split(".")
    for k in range(1, len(module_parts)):
        prefix = ".".join(module_parts[:k])
        candidate = f"{prefix}.{qualname}"
        try:
            if locate(candidate) is t:
                return candidate
        except ImportError:
            pass
    return f"{module}.{qualname}"


_convert_target_to_string = convert_target_to_string  # for backward compatibility.


def locate(name: str) -> Any:
    """Resolve ``"module.submodule.ClassName"`` to the live Python object.

    Args:
        name: Dotted import path as stored in a LazyCall ``_target_``.

    Returns:
        The located class, function, or other attribute.

    Raises:
        ImportError: Neither pydoc nor Hydra can find *name*, or Hydra
            itself is unavailable after pydoc failed.
        Exception: Hydra ``_locate`` failed after pydoc returned ``None``.
    """
    obj = pydoc.locate(name)

    # Some cases (e.g. torch.optim.sgd.SGD) not handled correctly
    # by pydoc.locate. Try a private function from hydra.
    if obj is None:
        try:
            # from hydra.utils import get_method - will print many errors

            from hydra.utils import _locate
        except ImportError as e:
            raise ImportError(f"Cannot dynamically locate object {name}!") from e
        else:
            obj = _locate(name)  # it raises if fails

    return obj
