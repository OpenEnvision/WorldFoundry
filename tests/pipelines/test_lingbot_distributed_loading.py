from __future__ import annotations

from typing import Any

import pytest

from worldfoundry.pipelines.lingbot_world.distributed import prepare_torchrun_kwargs

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("model_id", ["lingbot-world", "lingbot-world-v2", "lingbot-world-v2-1.3b"])
@pytest.mark.parametrize("vram,sharded", [(80.0, False), (48.0, True), (None, True)])
def test_lingbot_owns_replication_and_sharding_defaults(
    model_id: str,
    vram: float | None,
    sharded: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WORLDFOUNDRY_LINGBOT_REPLICATED_MIN_VRAM_GB", raising=False)
    load, call = prepare_torchrun_kwargs(
        model_id=model_id,
        load_kwargs={"rank": 2},
        call_kwargs={},
        world_size=4,
        minimum_gpu_vram_gib=vram,
        sequence_parallel=False,
    )
    assert load["rank"] == 2
    assert load["t5_fsdp"] is sharded and load["dit_fsdp"] is sharded
    assert load["t5_cpu"] is False and load["ulysses_size"] == 1
    assert load["offload_model"] is False and call["offload_model"] is False
    if model_id == "lingbot-world":
        assert "nproc_per_node" not in call
    else:
        assert call["nproc_per_node"] == 4


@pytest.mark.parametrize("explicit,expected", [((), False), (("dit_fsdp",), True)])
def test_catalog_launch_defaults_do_not_override_model_policy(
    explicit: tuple[str, ...],
    expected: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WORLDFOUNDRY_LINGBOT_REPLICATED_MIN_VRAM_GB", raising=False)
    load, _ = prepare_torchrun_kwargs(
        model_id="lingbot-world",
        load_kwargs={"t5_fsdp": True, "dit_fsdp": True},
        call_kwargs={},
        world_size=4,
        minimum_gpu_vram_gib=80.0,
        sequence_parallel=False,
        explicit_load_params=explicit,
    )
    assert load["dit_fsdp"] is expected
    assert load["t5_fsdp"] is False


def test_explicit_options_win_and_caller_mappings_are_not_mutated() -> None:
    original_load: dict[str, Any] = {"t5_fsdp": True, "dit_fsdp": False, "ulysses_size": 2, "t5_cpu": True}
    original_call: dict[str, Any] = {"offload_model": True, "nproc_per_node": 8}
    load, call = prepare_torchrun_kwargs(
        model_id="lingbot-world-v2",
        load_kwargs=original_load,
        call_kwargs=original_call,
        world_size=4,
        minimum_gpu_vram_gib=80.0,
        sequence_parallel=True,
    )
    assert load == {**original_load, "offload_model": True}
    assert call == {"offload_model": True, "nproc_per_node": 4}
    assert original_load == {"t5_fsdp": True, "dit_fsdp": False, "ulysses_size": 2, "t5_cpu": True}
    assert original_call == {"offload_model": True, "nproc_per_node": 8}


def test_sequence_parallel_and_memory_threshold_follow_operator_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_LINGBOT_REPLICATED_MIN_VRAM_GB", "96")
    load, _ = prepare_torchrun_kwargs(
        model_id="lingbot-world",
        load_kwargs={"offload_model": True},
        call_kwargs={"offload_model": False},
        world_size=4,
        minimum_gpu_vram_gib=80.0,
        sequence_parallel=True,
    )
    assert load["t5_fsdp"] is True and load["dit_fsdp"] is True
    assert load["ulysses_size"] == 4
    assert load["offload_model"] is False
