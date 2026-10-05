"""CPU-only tests for the CC-12 fix: dist_init failure semantics.

If the environment carries distributed-launcher indicators, an
init_process_group failure must re-raise instead of masquerading the process
as RANK=0/WORLD_SIZE=1. Without such indicators the single-process fallback
is preserved.
"""

from __future__ import annotations

import pytest
import torch

from worldfoundry.core.distributed.collectives import generic as generic_collectives


_INDICATOR_VARS = (
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "MASTER_ADDR",
    "MASTER_PORT",
    "TORCHELASTIC_RUN_ID",
)


@pytest.fixture()
def clean_dist_env(monkeypatch):
    for name in _INDICATOR_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def _install_failing_init(monkeypatch, calls):
    def failing_init_process_group(*args, **kwargs):
        calls.append((args, kwargs))
        raise RuntimeError("simulated NCCL rendezvous failure")

    monkeypatch.setattr(torch.distributed, "init_process_group", failing_init_process_group)


def test_dist_init_reraises_when_distributed_env_present(clean_dist_env):
    monkeypatch = clean_dist_env
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("MASTER_ADDR", "10.0.0.1")
    calls = []
    _install_failing_init(monkeypatch, calls)

    with pytest.raises(RuntimeError, match="simulated NCCL rendezvous failure"):
        generic_collectives.dist_init()

    assert len(calls) == 1
    # The failing rank must not be rewritten into a fake standalone process.
    import os

    assert os.environ["RANK"] == "3"
    assert os.environ["WORLD_SIZE"] == "8"


@pytest.mark.parametrize("indicator", ["RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "TORCHELASTIC_RUN_ID"])
def test_dist_init_reraises_for_each_indicator(clean_dist_env, indicator):
    monkeypatch = clean_dist_env
    monkeypatch.setenv(indicator, "0" if indicator != "MASTER_ADDR" else "127.0.0.1")
    calls = []
    _install_failing_init(monkeypatch, calls)

    with pytest.raises(RuntimeError, match="simulated NCCL rendezvous failure"):
        generic_collectives.dist_init()


def test_dist_init_falls_back_to_single_process_without_indicators(clean_dist_env):
    monkeypatch = clean_dist_env
    calls = []
    _install_failing_init(monkeypatch, calls)

    generic_collectives.dist_init()  # must not raise

    import os

    assert len(calls) == 1
    assert os.environ["RANK"] == "0"
    assert os.environ["WORLD_SIZE"] == "1"
    assert os.environ["LOCAL_RANK"] == "0"


def test_dist_init_noop_when_already_initialized(clean_dist_env):
    monkeypatch = clean_dist_env
    calls = []
    _install_failing_init(monkeypatch, calls)
    monkeypatch.setattr(generic_collectives, "is_dist_initialized", lambda: True)

    generic_collectives.dist_init()

    assert calls == []


def test_master_port_alone_is_not_a_distributed_indicator(clean_dist_env):
    monkeypatch = clean_dist_env
    monkeypatch.setenv("MASTER_PORT", "29500")
    calls = []
    _install_failing_init(monkeypatch, calls)

    generic_collectives.dist_init()  # port alone must keep the fallback path


def test_model_parallel_world_sizes_default_to_one(monkeypatch):
    from worldfoundry.core.distributed.model_parallel import groups as model_parallel_groups

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    assert model_parallel_groups.get_dp_world_size() == 1
    assert model_parallel_groups.get_cp_world_size() == 1


def test_destroy_model_parallel_releases_unique_owned_subgroups(monkeypatch):
    from worldfoundry.core.distributed.model_parallel import groups as model_parallel_groups

    group_a = object()
    group_b = object()
    monkeypatch.setattr(model_parallel_groups, "_MODEL_PARALLEL_GROUP", group_a)
    monkeypatch.setattr(model_parallel_groups, "_TENSOR_MODEL_PARALLEL_GROUP", group_a)
    monkeypatch.setattr(model_parallel_groups, "_DATA_PARALLEL_GROUP", group_b)
    monkeypatch.setattr(model_parallel_groups, "_DATA_PARALLEL_GLOBAL_RANKS", [0, 1])
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    destroyed: list[object] = []
    monkeypatch.setattr(torch.distributed, "destroy_process_group", destroyed.append)

    model_parallel_groups.destroy_model_parallel()

    assert destroyed == [group_a, group_b]
    assert model_parallel_groups._MODEL_PARALLEL_GROUP is None
    assert model_parallel_groups._TENSOR_MODEL_PARALLEL_GROUP is None
    assert model_parallel_groups._DATA_PARALLEL_GROUP is None
    assert model_parallel_groups._DATA_PARALLEL_GLOBAL_RANKS is None
