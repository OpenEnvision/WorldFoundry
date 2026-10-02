"""CC-17 / XC-10 / SA-14: canonical dist init, CP reset, import cycles."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = REPO_ROOT / "worldfoundry/core/distributed"

_INDICATOR_VARS = (
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "MASTER_ADDR",
    "MASTER_PORT",
    "TORCHELASTIC_RUN_ID",
    "SLURM_PROCID",
    "SLURM_NTASKS",
    "HOSTNAME",
)


def _parse(relpath: str) -> ast.Module:
    return ast.parse((REPO_ROOT / relpath).read_text(encoding="utf-8"))


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name!r} not found")


def _names_in(node: ast.AST) -> set[str]:
    names = {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}
    names.update(child.attr for child in ast.walk(node) if isinstance(child, ast.Attribute))
    return names


def _print_calls(tree: ast.AST) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print"
    ]


def _package_reexport_imports(tree: ast.AST) -> list[str]:
    """Imports that load a sibling by going through the parent package."""

    hits: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level == 1 and node.module is None:
            names = ", ".join(alias.name for alias in node.names)
            hits.append(f"from . import {names}")
        if node.level == 0 and node.module == "worldfoundry.core.distributed":
            names = ", ".join(alias.name for alias in node.names)
            hits.append(f"from worldfoundry.core.distributed import {names}")
    return hits


@pytest.fixture()
def clean_dist_env(monkeypatch):
    for name in _INDICATOR_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_reset_context_parallel_clears_module_globals() -> None:
    pytest.importorskip("einops")
    from worldfoundry.core.distributed import context_parallel_util as cpu

    cpu.dp_size = 8
    cpu.cp_size = 4
    cpu.dp_group = object()
    cpu.cp_group = object()
    cpu.cp_stream = object()
    cpu.dp_ranks = [0, 1]
    cpu.cp_ranks = [2, 3]
    cpu.dp_rank = 1
    cpu.cp_rank = 0

    cpu.reset_context_parallel()

    assert cpu.dp_size is None
    assert cpu.cp_size is None
    assert cpu.dp_group is None
    assert cpu.cp_group is None
    assert cpu.cp_stream is None
    assert cpu.dp_ranks is None
    assert cpu.cp_ranks is None
    assert cpu.dp_rank is None
    assert cpu.cp_rank is None


def test_context_parallel_util_uses_logging_not_print() -> None:
    tree = _parse("worldfoundry/core/distributed/context_parallel_util.py")
    assert _print_calls(tree) == []
    names = _names_in(_function(tree, "init_context_parallel"))
    assert "logger" in names
    source = (DIST_DIR / "context_parallel_util.py").read_text(encoding="utf-8")
    assert "sequence_parallel.parallel_state" in source
    assert "reset_context_parallel" in source


def test_dist_init_wrapper_emits_deprecation_and_calls_canonical(clean_dist_env) -> None:
    import torch

    from worldfoundry.core.distributed import generic_collectives

    monkeypatch = clean_dist_env
    called: list[tuple[tuple, dict]] = []
    initialized = {"value": False}

    def fake_init(*args, **kwargs):
        called.append((args, kwargs))
        initialized["value"] = True

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: initialized["value"])
    monkeypatch.setattr(
        "worldfoundry.core.distributed.torch_process_group.init_torch_distributed",
        fake_init,
    )

    with pytest.warns(DeprecationWarning, match="generic_collectives.dist_init is deprecated"):
        generic_collectives.dist_init()

    assert len(called) == 1


def test_metric_sync_init_distributed_wrapper_emits_deprecation(clean_dist_env) -> None:
    from worldfoundry.core.distributed import metric_sync

    monkeypatch = clean_dist_env
    called: list[dict] = []

    def fake_init(*args, **kwargs):
        called.append(kwargs)

    monkeypatch.setattr(
        "worldfoundry.core.distributed.torch_process_group.init_torch_distributed",
        fake_init,
    )

    with pytest.warns(DeprecationWarning, match="metric_sync.init_distributed is deprecated"):
        world_size, rank, gpu, flag = metric_sync.init_distributed()

    assert called and called[0]["world_size"] == 1 and called[0]["rank"] == 0
    assert (world_size, rank, gpu, flag) == (1, 0, 0, True)


def test_generic_collectives_get_rank_delegates_to_torch_process_group(monkeypatch) -> None:
    from worldfoundry.core.distributed import generic_collectives

    monkeypatch.setattr(
        "worldfoundry.core.distributed.torch_process_group.get_rank",
        lambda group=None: 7,
    )
    monkeypatch.setattr(
        "worldfoundry.core.distributed.torch_process_group.get_world_size",
        lambda group=None: 4,
    )
    monkeypatch.setattr(
        "worldfoundry.core.distributed.torch_process_group.get_local_rank",
        lambda: 2,
    )
    assert generic_collectives.get_rank() == 7
    assert generic_collectives.get_world_size() == 4
    assert generic_collectives.get_local_rank() == 2


def test_canonical_init_and_wrappers_are_wired_in_source() -> None:
    tpg = _parse("worldfoundry/core/distributed/torch_process_group.py")
    assert any(
        isinstance(node, ast.FunctionDef) and node.name == "init_torch_distributed" for node in tpg.body
    )
    dist_init = _function(_parse("worldfoundry/core/distributed/generic_collectives.py"), "dist_init")
    dist_init_names = _names_in(dist_init)
    assert "DeprecationWarning" in dist_init_names
    assert "init_torch_distributed" in dist_init_names

    metric_init = _function(_parse("worldfoundry/core/distributed/metric_sync.py"), "init_distributed")
    metric_names = _names_in(metric_init)
    assert "DeprecationWarning" in metric_names
    assert "init_torch_distributed" in metric_names

    worker = _function(_parse("worldfoundry/core/distributed/multiprocess_launch.py"), "distributed_worker")
    worker_names = _names_in(worker)
    assert "DeprecationWarning" in worker_names
    assert "init_torch_distributed" in worker_names


@pytest.mark.parametrize(
    "relpath",
    [
        "worldfoundry/core/distributed/context_parallel.py",
        "worldfoundry/core/distributed/fsdp_runtime.py",
        "worldfoundry/core/distributed/inference_runtime.py",
        "worldfoundry/core/distributed/multiprocess_launch.py",
        "worldfoundry/core/distributed/pipeline_parallel.py",
        "worldfoundry/core/distributed/generic_collectives.py",
        "worldfoundry/core/distributed/metric_sync.py",
        "worldfoundry/core/distributed/sequence_parallel/parallel_state.py",
        "worldfoundry/core/distributed/sequence_parallel/logger.py",
        "worldfoundry/core/distributed/sequence_parallel/cuda_utils.py",
    ],
)
def test_sa14_submodules_do_not_import_siblings_via_package(relpath: str) -> None:
    hits = _package_reexport_imports(_parse(relpath))
    assert hits == [], hits
