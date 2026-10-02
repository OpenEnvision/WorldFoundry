"""CC-17 leftover wrappers: evaluation / runtime_setup / sequence_ops / WONTFIX."""

from __future__ import annotations

import ast
import os
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


@pytest.fixture()
def clean_dist_env(monkeypatch):
    for name in _INDICATOR_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def _assert_wrapper_calls_canonical(relpath: str, func_name: str) -> None:
    func = _function(_parse(relpath), func_name)
    names = _names_in(func)
    assert "DeprecationWarning" in names
    assert "init_torch_distributed" in names
    assert "init_process_group" not in names


def test_evaluation_collectives_dist_init_is_canonical_wrapper() -> None:
    _assert_wrapper_calls_canonical(
        "worldfoundry/core/distributed/evaluation_collectives.py",
        "dist_init",
    )


def test_runtime_setup_init_distributed_is_canonical_wrapper() -> None:
    _assert_wrapper_calls_canonical(
        "worldfoundry/core/distributed/runtime_setup.py",
        "init_distributed",
    )


def test_sequence_ops_init_distributed_group_is_canonical_wrapper() -> None:
    _assert_wrapper_calls_canonical(
        "worldfoundry/core/distributed/sequence_ops.py",
        "init_distributed_group",
    )


def test_inference_runtime_dist_init_wontfix_keeps_own_init() -> None:
    relpath = "worldfoundry/core/distributed/inference_runtime.py"
    func = _function(_parse(relpath), "dist_init")
    names = _names_in(func)
    assert "init_torch_distributed" not in names
    assert "DeprecationWarning" not in names
    assert "init_process_group" in names
    assert "initialize_model_parallel" in names
    source = (DIST_DIR / "inference_runtime.py").read_text(encoding="utf-8")
    assert "CC-17 WONTFIX" in source


def test_evaluation_collectives_wrapper_emits_deprecation_and_calls_canonical(clean_dist_env) -> None:
    import torch

    from worldfoundry.core.distributed import evaluation_collectives

    monkeypatch = clean_dist_env
    called: list[tuple[tuple, dict]] = []

    def fake_init(*args, **kwargs):
        called.append((args, kwargs))

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        "worldfoundry.core.distributed.torch_process_group.init_torch_distributed",
        fake_init,
    )

    with pytest.warns(DeprecationWarning, match="evaluation_collectives.dist_init is deprecated"):
        evaluation_collectives.dist_init()

    assert len(called) == 1
    assert called[0][1]["backend"] == "gloo"
    assert called[0][1]["init_method"] == "env://"
    assert os.environ["MASTER_ADDR"] == "localhost"
    assert os.environ["MASTER_PORT"] == "29500"
    assert os.environ["RANK"] == "0"
    assert os.environ["LOCAL_RANK"] == "0"
    assert os.environ["WORLD_SIZE"] == "1"


def test_evaluation_collectives_preserves_existing_master_env(clean_dist_env) -> None:
    import torch

    from worldfoundry.core.distributed import evaluation_collectives

    monkeypatch = clean_dist_env
    monkeypatch.setenv("MASTER_ADDR", "10.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "12345")
    monkeypatch.setenv("RANK", "3")
    called: list[dict] = []

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        "worldfoundry.core.distributed.torch_process_group.init_torch_distributed",
        lambda *args, **kwargs: called.append(kwargs),
    )

    with pytest.warns(DeprecationWarning, match="evaluation_collectives.dist_init is deprecated"):
        evaluation_collectives.dist_init()

    assert os.environ["MASTER_ADDR"] == "10.0.0.1"
    assert os.environ["MASTER_PORT"] == "12345"
    assert os.environ["RANK"] == "3"
    assert called and called[0]["backend"] == "gloo"


def test_evaluation_collectives_uses_nccl_when_cuda_available(clean_dist_env) -> None:
    import torch

    from worldfoundry.core.distributed import evaluation_collectives

    monkeypatch = clean_dist_env
    called: list[dict] = []

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "worldfoundry.core.distributed.torch_process_group.init_torch_distributed",
        lambda *args, **kwargs: called.append(kwargs),
    )

    with pytest.warns(DeprecationWarning, match="evaluation_collectives.dist_init is deprecated"):
        evaluation_collectives.dist_init()

    assert called and called[0]["backend"] == "nccl"


def test_evaluation_collectives_noop_when_already_initialized(clean_dist_env) -> None:
    import torch

    from worldfoundry.core.distributed import evaluation_collectives

    monkeypatch = clean_dist_env
    called: list[tuple] = []

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        "worldfoundry.core.distributed.torch_process_group.init_torch_distributed",
        lambda *args, **kwargs: called.append((args, kwargs)),
    )

    with pytest.warns(DeprecationWarning, match="evaluation_collectives.dist_init is deprecated"):
        evaluation_collectives.dist_init()

    assert called == []
    assert "MASTER_ADDR" not in os.environ


def test_runtime_setup_wrapper_emits_deprecation_and_forwards_ranks(clean_dist_env) -> None:
    import torch

    from worldfoundry.core.distributed import runtime_setup

    monkeypatch = clean_dist_env
    called: list[dict] = []
    devices: list[int] = []

    monkeypatch.setattr(torch.cuda, "set_device", lambda device: devices.append(int(device)))
    monkeypatch.setattr(
        "worldfoundry.core.distributed.torch_process_group.init_torch_distributed",
        lambda *args, **kwargs: called.append(kwargs),
    )

    with pytest.warns(DeprecationWarning, match="runtime_setup.init_distributed is deprecated"):
        runtime_setup.init_distributed(4, 1, 3)

    assert devices == [1]
    assert os.environ["LOCAL_RANK"] == "1"
    assert called == [{"backend": "nccl", "init_method": "env://", "rank": 3, "world_size": 4}]


def test_sequence_ops_wrapper_emits_deprecation_and_calls_canonical(clean_dist_env) -> None:
    import torch

    from worldfoundry.core.distributed import sequence_ops

    monkeypatch = clean_dist_env
    called: list[dict] = []

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(
        "worldfoundry.core.distributed.torch_process_group.init_torch_distributed",
        lambda *args, **kwargs: called.append(kwargs),
    )

    with pytest.warns(DeprecationWarning, match="sequence_ops.init_distributed_group is deprecated"):
        sequence_ops.init_distributed_group()

    assert called == [{"backend": "nccl"}]


def test_sequence_ops_skips_canonical_when_already_initialized(clean_dist_env) -> None:
    import torch

    from worldfoundry.core.distributed import sequence_ops

    monkeypatch = clean_dist_env
    called: list[tuple] = []

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        "worldfoundry.core.distributed.torch_process_group.init_torch_distributed",
        lambda *args, **kwargs: called.append((args, kwargs)),
    )

    with pytest.warns(DeprecationWarning, match="sequence_ops.init_distributed_group is deprecated"):
        sequence_ops.init_distributed_group()

    assert called == []
