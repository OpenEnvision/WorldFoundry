"""Operator registry/import smoke tests (code review item OR-12).

纯 CPU、不加载权重、不联网的防回归冒烟：

- 静态一致性：`_OPERATOR_MODULES` 每项对应的模块文件存在，且 AST 顶层能找到
  该类名（类定义 / 别名赋值 / re-export import），不依赖任何第三方包。
- 导入冒烟：每个注册模块可导入。缺第三方依赖（本机无 imageio/transformers
  等）记 skip；worldfoundry 内部的 ImportError/NameError/SyntaxError 一律 fail。
- 契约冒烟：导入成功的注册类必须具备 BaseOperator 交互契约方法。
- 注册表一致性：`__all__` 与 `_OPERATOR_MODULES` 同步、懒加载 `__getattr__`
  对未知名字抛 AttributeError。
"""

from __future__ import annotations

import ast
import importlib
import os
from pathlib import Path

import pytest

# 防御：个别模块导入期可能碰 HF hub；冒烟测试禁止联网。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import worldfoundry.operators as operators_pkg
from worldfoundry.operators import _OPERATOR_MODULES

OPERATORS_DIR = Path(operators_pkg.__file__).resolve().parent

# BaseOperator 的交互契约（见 worldfoundry/operators/base_operator.py）。
CONTRACT_METHODS = (
    "get_interaction",
    "process_interaction",
    "process_perception",
    "delete_last_interaction",
)

_REGISTRY_ITEMS = sorted(_OPERATOR_MODULES.items())
_REGISTRY_MODULES = sorted(set(_OPERATOR_MODULES.values()))

_AST_CACHE: dict[str, set[str]] = {}


def _top_level_names(module_name: str) -> set[str]:
    """Collect names bindable at module top level, without importing it."""
    if module_name in _AST_CACHE:
        return _AST_CACHE[module_name]
    source = (OPERATORS_DIR / f"{module_name}.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
    _AST_CACHE[module_name] = names
    return names


def _import_or_skip(dotted: str):
    """Import a module; skip on missing third-party deps, fail on in-tree errors."""
    try:
        return importlib.import_module(dotted)
    except ModuleNotFoundError as exc:
        missing_root = (exc.name or "").split(".")[0]
        if missing_root and missing_root != "worldfoundry":
            pytest.skip(f"missing third-party dependency: {exc.name}")
        raise
    except ImportError as exc:
        origin = (getattr(exc, "name", "") or "").split(".")[0]
        if origin == "worldfoundry":
            raise
        pytest.skip(f"third-party import error: {exc}")


# ── 静态检查（无第三方依赖，必须全绿） ─────────────────────────


@pytest.mark.parametrize("module_name", _REGISTRY_MODULES)
def test_registry_module_file_exists(module_name):
    assert (OPERATORS_DIR / f"{module_name}.py").is_file(), (
        f"registry points to missing module file: {module_name}.py"
    )


@pytest.mark.parametrize(("class_name", "module_name"), _REGISTRY_ITEMS)
def test_registry_class_defined_in_module_source(class_name, module_name):
    names = _top_level_names(module_name)
    assert class_name in names, (
        f"{class_name!r} not bound at top level of {module_name}.py; "
        f"registry entry is stale"
    )


def test_registry_dunder_all_consistent():
    assert operators_pkg.__all__ == sorted(_OPERATOR_MODULES)


def test_dualcamctrl_operator_is_publicly_registered():
    assert _OPERATOR_MODULES.get("DualCamCtrlOperator") == "dualcamctrl_operator"
    assert "DualCamCtrlOperator" in operators_pkg.__all__


def test_lazy_getattr_rejects_unknown_names():
    with pytest.raises(AttributeError):
        operators_pkg.NoSuchOperator  # noqa: B018


def test_embodied_action_compat_registry_consistent_with_package():
    """OR-08：兼容垫片不再自带 class→module 表，解析必须代理到包级注册表。"""
    import warnings

    compat = importlib.import_module("worldfoundry.operators.embodied_action_operator")
    assert not hasattr(compat, "_OPERATOR_EXPORTS"), (
        "second operator registry must not return; resolve via worldfoundry.operators"
    )

    with pytest.raises(AttributeError, match="NoSuchOperator"):
        getattr(compat, "NoSuchOperator")

    # Lightweight VLA module: same object as the package lazy loader.
    compat.__dict__.pop("ACTOperator", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        via_compat = getattr(compat, "ACTOperator")
    assert any(issubclass(item.category, DeprecationWarning) for item in caught), (
        "compat getattr must warn so the shim can be deleted later"
    )
    via_pkg = getattr(operators_pkg, "ACTOperator")
    assert via_compat is via_pkg
    assert via_compat is getattr(
        importlib.import_module("worldfoundry.operators.act_operator"),
        "ACTOperator",
    )


def test_base_operator_contract_methods_exist():
    module = importlib.import_module("worldfoundry.operators.base_operator")
    base = module.BaseOperator
    for method in CONTRACT_METHODS:
        assert callable(getattr(base, method, None)), f"BaseOperator lacks {method}"


# ── 导入与契约冒烟（缺依赖 skip，内部错误 fail） ────────────────


@pytest.mark.parametrize("module_name", _REGISTRY_MODULES)
def test_registry_module_importable(module_name):
    _import_or_skip(f"worldfoundry.operators.{module_name}")


@pytest.mark.parametrize(("class_name", "module_name"), _REGISTRY_ITEMS)
def test_registry_class_resolvable_and_contract(class_name, module_name):
    module = _import_or_skip(f"worldfoundry.operators.{module_name}")
    cls = getattr(module, class_name, None)
    assert isinstance(cls, type), f"{module_name}.{class_name} missing or not a class"
    for method in CONTRACT_METHODS:
        assert callable(getattr(cls, method, None)), (
            f"{class_name} lacks contract method {method}"
        )
