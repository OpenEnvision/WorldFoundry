"""Core-layer XC leftovers: registry rebind, trusted configs, fstring, shard pool."""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse(relpath: str) -> ast.Module:
    return ast.parse((REPO_ROOT / relpath).read_text(encoding="utf-8"))


def _function_uses_eval(tree: ast.Module, name: str) -> bool:
    func = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
    return any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "eval" for node in ast.walk(func))


@pytest.fixture
def isolated_class_registry():
    sys.modules.setdefault("tree", types.ModuleType("tree"))
    from worldfoundry.core.configuration import hydra_utils as cu

    saved = cu._CLASS_REGISTRY.copy()
    try:
        yield cu
    finally:
        cu._CLASS_REGISTRY.clear()
        cu._CLASS_REGISTRY.update(saved)


def test_xc11_register_class_rejects_name_rebind(isolated_class_registry) -> None:
    cu = isolated_class_registry
    first = type("XCCoreFixClash", (), {})
    second = type("XCCoreFixClash", (), {})
    cu.register_class(first)
    with pytest.raises(ValueError, match="already bound"):
        cu.register_class(second)
    assert cu.get_class("XCCoreFixClash") is first


def test_xc11_register_class_rejects_alias_rebind(isolated_class_registry) -> None:
    cu = isolated_class_registry
    owner = type("XCCoreFixAliasOwner", (), {})
    other = type("XCCoreFixAliasOther", (), {})
    cu.register_class(alias=["xc_core_fix_alias"])(owner)
    with pytest.raises(ValueError, match="already bound"):
        cu.register_class(alias=["xc_core_fix_alias"])(other)
    cu.register_class(alias=["xc_core_fix_alias"])(owner)
    assert cu.get_class("xc_core_fix_alias") is owner


def test_xc11_make_registry_metaclass_removed() -> None:
    tree = _parse("worldfoundry/core/utils/python/functional_utils.py")
    names = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))}
    assert "make_registry_metaclass" not in names
    import worldfoundry.core.utils.python.functional_utils as fu

    assert not hasattr(fu, "make_registry_metaclass")


def test_xc4_python_and_lazy_config_declare_trust_boundary() -> None:
    python_cfg = (REPO_ROOT / "worldfoundry/core/configuration/python.py").read_text(encoding="utf-8")
    lazy_cfg = (REPO_ROOT / "worldfoundry/core/configuration/lazy_config/config.py").read_text(encoding="utf-8")
    lazy_init = (REPO_ROOT / "worldfoundry/core/configuration/lazy_config/__init__.py").read_text(encoding="utf-8")
    for text in (python_cfg, lazy_cfg, lazy_init):
        assert "Trust boundary" in text
        assert "trusted" in text.lower()
    assert "exec_module" in python_cfg or "exec" in python_cfg
    assert "exec" in lazy_cfg


def test_xc4_fstring_supports_index_templates_without_eval() -> None:
    from worldfoundry.core.observability.formatting import fstring

    assert fstring("_v{i+1}", i=0) == "_v1"
    assert fstring("{i}", i=3) == "3"
    with pytest.raises((ValueError, NameError, SyntaxError)):
        fstring("{__import__('os').name}", i=0)
    print_tree = _parse("worldfoundry/core/observability/formatting.py")
    file_tree = _parse("worldfoundry/core/io/filesystem/file_utils.py")
    assert not _function_uses_eval(print_tree, "fstring")
    assert not _function_uses_eval(file_tree, "fstring")


def test_xc4_next_available_file_name_uses_i_plus_one(tmp_path) -> None:
    from worldfoundry.core.io.filesystem.file_utils import next_available_file_name

    path = tmp_path / "out.txt"
    path.write_text("x", encoding="utf-8")
    nxt = next_available_file_name(str(path))
    assert Path(nxt).name == "out_v1.txt"


def test_xc6_pickle_base64_helpers_removed() -> None:
    import worldfoundry.core.utils.python.misc_utils as mu

    assert not hasattr(mu, "encode_base64")
    assert not hasattr(mu, "decode_base64")
    tree = _parse("worldfoundry/core/utils/python/misc_utils.py")
    names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert "encode_base64" not in names
    assert "decode_base64" not in names


def test_xc14_shard_download_uses_thread_pool() -> None:
    tree = _parse("worldfoundry/core/model_loading/checkpoints/load.py")
    imported: set[str] = set()
    called: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "concurrent.futures":
            imported.update(alias.name for alias in node.names)
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                called.add(func.id)
    assert "ThreadPoolExecutor" in imported
    assert "ProcessPoolExecutor" not in imported
    assert "ThreadPoolExecutor" in called
    assert "ProcessPoolExecutor" not in called
