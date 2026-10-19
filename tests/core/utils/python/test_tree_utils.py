# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Optional tree imports must not retain a model-owning caller frame."""

from __future__ import annotations

import builtins
import gc
import importlib.util
import weakref
from types import ModuleType
from typing import Any

import pytest

from worldfoundry.core.utils.python import tree_utils

pytestmark = pytest.mark.unit


class Owner:
    pass


def test_missing_tree_dependency_releases_import_caller(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def missing_tree(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "tree":
            raise ImportError("dm-tree unavailable")
        return original_import(name, *args, **kwargs)

    def import_with_owner(owner: Owner) -> ModuleType:
        spec = importlib.util.spec_from_file_location("_tree_import_lifetime", tree_utils.__file__)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    owner = Owner()
    reference = weakref.ref(owner)
    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", missing_tree)
        module = import_with_owner(owner)
    del owner
    gc.collect()
    assert reference() is None
    with pytest.raises(ImportError, match="pip install dm_tree") as failure:
        module._require_tree()
    cause = failure.value.__cause__
    assert cause is not None and str(cause) == "dm-tree unavailable"
    assert cause.__traceback__ is None
    assert cause.__context__ is None
