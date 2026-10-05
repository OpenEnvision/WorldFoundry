import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from worldfoundry.runtime.conda import RuntimeCondaEnvSpec, unified_env_blocker
from worldfoundry.runtime.device_pool import CudaDeviceLeasePool
from worldfoundry.runtime.inference_catalog import get_model_inference_spec
from worldfoundry.studio.inference import dispatch as conda_dispatch
from worldfoundry.studio.inference.dispatch import (
    TORCHRUN_DISTRIBUTED_MODELS,
    _automatic_cuda_device_count,
    _default_torchrun_nproc,
    _explicit_torchrun_nproc,
    _internal_torchrun_nproc,
    _payload_run_kwargs_with_secret_refs,
    _requested_torchrun_nproc,
    _retire_idle_resident_workers_for_cuda_request,
    _retire_idle_resident_workers_for_new_resident_key,
    _runtime_pythonpath,
    _with_lingbot_world_parallelism,
    _with_longvie2_parallelism,
    _with_matrix_game3_parallelism,
    _with_wan_vace_parallelism,
)
from worldfoundry.studio.inference.config import wmfactory_interactive_model_spec
from worldfoundry.studio.runtime_job import _resolve_secret_refs


def test_bernini_default_reserves_one_memory_safe_gpu() -> None:
    run_kwargs = {
        "device": "cuda",
        "call_kwargs_text": json.dumps({"nproc_per_node": 1, "ulysses_size": 1}),
    }

    assert _internal_torchrun_nproc("bernini", run_kwargs) == 1
    assert _automatic_cuda_device_count("bernini", run_kwargs) == 1


def test_bernini_rejects_mismatched_internal_ulysses_group() -> None:
    run_kwargs = {
        "call_kwargs_text": json.dumps({"nproc_per_node": 2, "ulysses_size": 1}),
    }

    with pytest.raises(ValueError, match="ulysses_size must equal nproc_per_node"):
        _internal_torchrun_nproc("bernini", run_kwargs)


def test_wan_vace_uses_outer_four_rank_torchrun_and_reserves_four_gpus() -> None:
    run_kwargs = {
        "device": "cuda",
        "call_kwargs_text": json.dumps(
            {"nproc_per_node": 4, "ulysses_size": 4, "ring_size": 1}
        ),
    }

    assert _explicit_torchrun_nproc("wan2.1-vace", run_kwargs) == 4
    assert _internal_torchrun_nproc("wan2.1-vace", run_kwargs) == 0
    assert _automatic_cuda_device_count("wan2.1-vace", run_kwargs) == 4
    assert "wan2.1-vace" in TORCHRUN_DISTRIBUTED_MODELS


def test_wan_vace_parallelism_enables_native_sequence_parallelism() -> None:
    result = _with_wan_vace_parallelism(
        "wan2.1-vace",
        {
            "load_kwargs_text": json.dumps(
                {"torch_dtype": "bfloat16", "sp_degree": 2}
            ),
            "call_kwargs_text": json.dumps(
                {"nproc_per_node": 4, "ulysses_size": 1, "ring_size": 1}
            ),
        },
        world_size=4,
    )

    assert json.loads(result["load_kwargs_text"]) == {
        "torch_dtype": "bfloat16",
        "sequence_parallel": 4,
    }
    assert json.loads(result["call_kwargs_text"]) == {
        "nproc_per_node": 4,
        "ulysses_size": 4,
        "ring_size": 1,
    }


def test_wan_vace_rejects_single_rank_and_ring_parallel_topologies() -> None:
    with pytest.raises(ValueError, match="nproc_per_node > 1"):
        _with_wan_vace_parallelism("wan2.1-vace", {}, world_size=1)
    with pytest.raises(ValueError, match="ring_size=1"):
        _with_wan_vace_parallelism(
            "wan2.1-vace",
            {"call_kwargs_text": json.dumps({"ring_size": 2})},
            world_size=4,
        )


def test_payload_secret_refs_cover_nested_kwargs_and_json_text():
    load_kwargs_text = json.dumps(
        {
            "provider": {"api_key": "load-secret"},
            "max_new_tokens": 64,
            "tokenizer": "gpt2",
        }
    )
    call_kwargs_text = json.dumps(
        {"auth_token": "call-text-secret", "max_new_tokens": 32}
    )
    payload, secret_env = _payload_run_kwargs_with_secret_refs(
        {
            "api_key": "top-level-secret",
            "load_kwargs_text": load_kwargs_text,
            "call_kwargs_text": call_kwargs_text,
            "call_kwargs": {
                "nested": [{"auth_token": "call-secret"}],
                "max_new_tokens": 32,
            },
        }
    )

    assert payload["api_key"] == {
        conda_dispatch.SECRET_ENV_REF_KEY: conda_dispatch.DISPATCH_API_KEY_ENV,
    }
    load_ref = payload["load_kwargs_text"]
    assert isinstance(load_ref, dict)
    assert secret_env[load_ref[conda_dispatch.SECRET_ENV_REF_KEY]] == load_kwargs_text
    call_text_ref = payload["call_kwargs_text"]
    assert isinstance(call_text_ref, dict)
    assert secret_env[call_text_ref[conda_dispatch.SECRET_ENV_REF_KEY]] == call_kwargs_text
    call_ref = payload["call_kwargs"]["nested"][0]["auth_token"]
    assert isinstance(call_ref, dict)
    assert secret_env[call_ref[conda_dispatch.SECRET_ENV_REF_KEY]] == "call-secret"
    assert payload["call_kwargs"]["max_new_tokens"] == 32
    assert "load-secret" not in json.dumps(payload)
    assert "call-text-secret" not in json.dumps(payload)
    assert "call-secret" not in json.dumps(payload)


def test_payload_secret_refs_preserve_non_secret_token_fields_and_resolve_in_child(monkeypatch):
    run_kwargs = {
        "load_kwargs_text": json.dumps(
            {"max_new_tokens": 64, "tokenizer": "provider-tokenizer"}
        ),
        "endpoint": "https://alice:password@example.test/v1",
    }

    payload, secret_env = _payload_run_kwargs_with_secret_refs(run_kwargs)

    assert payload["load_kwargs_text"] == run_kwargs["load_kwargs_text"]
    endpoint_ref = payload["endpoint"]
    endpoint_env = endpoint_ref[conda_dispatch.SECRET_ENV_REF_KEY]
    assert secret_env[endpoint_env] == run_kwargs["endpoint"]
    assert "alice:password" not in json.dumps(payload)
    for name, value in secret_env.items():
        monkeypatch.setenv(name, value)
    assert _resolve_secret_refs(payload) == run_kwargs


def test_resident_worker_request_timeout_covers_multi_hour_inference(monkeypatch):
    monkeypatch.delenv(conda_dispatch.RESIDENT_WORKER_REQUEST_TIMEOUT_ENV, raising=False)

    assert conda_dispatch._resident_worker_request_timeout() == 6 * 60 * 60


def test_resident_worker_request_timeout_keeps_environment_override(monkeypatch):
    monkeypatch.setenv(conda_dispatch.RESIDENT_WORKER_REQUEST_TIMEOUT_ENV, "7200")

    assert conda_dispatch._resident_worker_request_timeout() == 7200.0


def test_runtime_pythonpath_drops_foreign_site_packages(tmp_path):
    child_root = tmp_path / "child"
    spec = RuntimeCondaEnvSpec(model_id="example", env_name="child", env_root=tmp_path)
    child_packages = child_root / "lib" / "python3.10" / "site-packages"
    parent_packages = tmp_path / "parent" / "lib" / "python3.11" / "site-packages"
    source_root = tmp_path / "source"
    env = {
        "PYTHONPATH": os.pathsep.join((str(parent_packages), str(source_root), str(child_packages))),
    }

    result = _runtime_pythonpath(spec, env).split(os.pathsep)

    assert str(parent_packages) not in result
    assert str(source_root) in result
    assert str(child_packages) in result


def test_runtime_env_does_not_inject_hf_endpoint_by_default(tmp_path, monkeypatch):
    spec = RuntimeCondaEnvSpec(model_id="example", env_name="child", env_root=tmp_path)
    monkeypatch.delenv("WORLDFOUNDRY_HF_ENDPOINT", raising=False)
    monkeypatch.delenv("HF_ENDPOINT", raising=False)

    result = conda_dispatch._runtime_env(spec)

    assert "HF_ENDPOINT" not in result


def test_runtime_env_disables_user_site_packages(tmp_path, monkeypatch):
    spec = RuntimeCondaEnvSpec(model_id="example", env_name="child", env_root=tmp_path)
    monkeypatch.setenv("PYTHONNOUSERSITE", "")

    result = conda_dispatch._runtime_env(spec)

    assert result["PYTHONNOUSERSITE"] == "1"


def test_runtime_env_uses_explicit_worldfoundry_hf_endpoint(tmp_path, monkeypatch):
    spec = RuntimeCondaEnvSpec(model_id="example", env_name="child", env_root=tmp_path)
    monkeypatch.setenv("HF_ENDPOINT", "https://parent.example")
    monkeypatch.setenv("WORLDFOUNDRY_HF_ENDPOINT", " https://opt-in.example ")

    result = conda_dispatch._runtime_env(spec)

    assert result["HF_ENDPOINT"] == "https://opt-in.example"


def test_runtime_env_preserves_parent_hf_endpoint_without_nonempty_override(tmp_path, monkeypatch):
    spec = RuntimeCondaEnvSpec(model_id="example", env_name="child", env_root=tmp_path)
    monkeypatch.setenv("HF_ENDPOINT", "https://parent.example")
    monkeypatch.setenv("WORLDFOUNDRY_HF_ENDPOINT", "  ")

    result = conda_dispatch._runtime_env(spec)

    assert result["HF_ENDPOINT"] == "https://parent.example"


def test_strict_transformers_major_cap_stays_in_isolated_environment():
    spec = RuntimeCondaEnvSpec(
        model_id="strict-transformers-model",
        env_name="strict-transformers-model",
        pip_packages=("transformers>=4.57,<5",),
    )

    assert unified_env_blocker(spec) == "transformers_upper_bound_5_requires_isolated_env"


def test_explicit_active_unified_env_remains_available_for_torchrun(tmp_path, monkeypatch):
    env_prefix = tmp_path / "worldfoundry-unified-cu121"
    python = env_prefix / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.touch()
    spec = RuntimeCondaEnvSpec(
        model_id="longvie-2",
        env_name="worldfoundry-longvie-cu128",
        pip_packages=("transformers>=4.57,<5",),
    )
    monkeypatch.setenv("WORLDFOUNDRY_USE_UNIFIED_ENV", "1")
    monkeypatch.setattr(conda_dispatch.sys, "executable", str(python))

    resolved = conda_dispatch._current_explicit_unified_env_spec(spec)

    assert resolved is not None
    assert resolved.env_prefix == env_prefix
    assert resolved.python_executable == python


def test_gamecraft_single_device_request_does_not_force_eight_processes():
    assert _default_torchrun_nproc("hunyuan-game-craft", {"device": "cuda:3"}) == 1


def test_worldplay_workspace_task_defaults_to_one_process() -> None:
    spec = get_model_inference_spec("hunyuan-worldplay")
    task = next(item for item in spec.tasks if item.task_id == spec.default_task_id)
    nproc = next(item for item in task.inputs if item.field_id == "torchrun_nproc_per_node")

    assert nproc.default == 1
    run_kwargs = {
        "device": "cuda:3",
        "load_kwargs_text": json.dumps({"torchrun_nproc_per_node": nproc.default}),
    }
    assert _requested_torchrun_nproc("hunyuan-worldplay", run_kwargs) == 1
    assert _default_torchrun_nproc("hunyuan-worldplay", run_kwargs) == 0


def test_bare_cuda_requests_one_exclusive_workspace_device(monkeypatch):
    monkeypatch.delenv("WORLDFOUNDRY_STUDIO_CONDA_CHILD", raising=False)
    monkeypatch.delenv("WORLDFOUNDRY_STUDIO_AUTO_GPU_PLACEMENT", raising=False)

    assert _automatic_cuda_device_count("wan2.1-t2v-1.3b", {"device": "cuda"}) == 1
    assert _automatic_cuda_device_count("wan2.1-t2v-1.3b", {"device": "cuda:3"}) == 0


def test_explicit_visible_devices_bypass_workspace_allocator(monkeypatch):
    monkeypatch.delenv("WORLDFOUNDRY_STUDIO_CONDA_CHILD", raising=False)

    assert _automatic_cuda_device_count(
        "wan2.1-t2v-1.3b",
        {
            "device": "cuda",
            "load_kwargs_text": json.dumps({"cuda_visible_devices": "2"}),
        },
    ) == 0


def test_multi_gpu_request_retires_older_idle_resident_lease(monkeypatch):
    pool = CudaDeviceLeasePool(("0", "1", "2"))
    lease = pool.acquire()
    worker = SimpleNamespace(
        key=("idle-model",),
        in_use=0,
        last_used_at=1.0,
        device_lease=lease,
    )
    monkeypatch.setattr(conda_dispatch, "_automatic_gpu_pool", lambda: pool)
    monkeypatch.setattr(
        conda_dispatch,
        "_shutdown_resident_worker",
        lambda retired, force=False: retired.device_lease.release(),
    )
    with conda_dispatch._RESIDENT_WORKERS_LOCK:
        conda_dispatch._RESIDENT_WORKERS[worker.key] = worker
    try:
        assert _retire_idle_resident_workers_for_cuda_request(3) == 1
        assert pool.available_count == 3
    finally:
        with conda_dispatch._RESIDENT_WORKERS_LOCK:
            conda_dispatch._RESIDENT_WORKERS.pop(worker.key, None)


def test_new_resident_key_retires_an_unrelated_idle_gpu_worker(monkeypatch):
    pool = CudaDeviceLeasePool(("0",))
    lease = pool.acquire()
    worker = SimpleNamespace(
        key=("old-model",),
        base_key=("old-model",),
        process=SimpleNamespace(poll=lambda: None),
        in_use=0,
        last_used_at=1.0,
        device_lease=lease,
    )
    monkeypatch.setattr(conda_dispatch, "_automatic_gpu_pool", lambda: pool)
    monkeypatch.setattr(
        conda_dispatch,
        "_shutdown_resident_worker",
        lambda retired, force=False: retired.device_lease.release(),
    )
    with conda_dispatch._RESIDENT_WORKERS_LOCK:
        conda_dispatch._RESIDENT_WORKERS[worker.key] = worker
    try:
        assert _retire_idle_resident_workers_for_new_resident_key(("new-model",), 1) == 1
        assert pool.available_count == 1
    finally:
        with conda_dispatch._RESIDENT_WORKERS_LOCK:
            conda_dispatch._RESIDENT_WORKERS.pop(worker.key, None)


def test_new_resident_key_keeps_a_reusable_worker(monkeypatch):
    worker = SimpleNamespace(
        key=("same-model", "auto-cuda:0"),
        base_key=("same-model",),
        process=SimpleNamespace(poll=lambda: None),
    )
    retire_calls = []
    monkeypatch.setattr(
        conda_dispatch,
        "_retire_idle_resident_workers_for_cuda_request",
        lambda count: retire_calls.append(count) or 1,
    )
    with conda_dispatch._RESIDENT_WORKERS_LOCK:
        conda_dispatch._RESIDENT_WORKERS[worker.key] = worker
    try:
        assert _retire_idle_resident_workers_for_new_resident_key(("same-model",), 1) == 0
        assert retire_calls == []
    finally:
        with conda_dispatch._RESIDENT_WORKERS_LOCK:
            conda_dispatch._RESIDENT_WORKERS.pop(worker.key, None)


def test_gamecraft_default_processes_are_capped_to_visible_gpus(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")

    assert _default_torchrun_nproc("hunyuan-game-craft", {}) == 4


def test_realtime_nproc_environment_overrides_model_default(monkeypatch):
    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_NPROC", "2")

    assert _requested_torchrun_nproc("lingbot-world-v2", {}) == 2


def test_matrix_game3_defaults_to_four_processes(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")

    assert _default_torchrun_nproc("matrix-game-3", {}) == 4
    assert wmfactory_interactive_model_spec("matrix-game-3").preferred_visible_devices == 4


def test_matrix_game3_default_uses_supported_size_for_visible_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2")

    assert _default_torchrun_nproc("matrix-game-3", {}) == 2


def test_matrix_game3_default_respects_model_gpu_override(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    monkeypatch.setenv("WM_MATRIXGAME3_CUDA_VISIBLE_DEVICES", "5,7")

    assert _default_torchrun_nproc("matrix-game-3", {}) == 2


def test_matrix_game3_realtime_nproc_is_user_configurable(monkeypatch):
    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_NPROC_PER_NODE", "8")

    assert _requested_torchrun_nproc("matrix-game-3", {}) == 8


def test_matrix_game3_rejects_unsupported_process_count(monkeypatch):
    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_NPROC", "3")

    with pytest.raises(ValueError, match="supports nproc/ulysses_size values 1, 2, 4, 8"):
        _requested_torchrun_nproc("matrix-game-3", {})


def test_dreamx_defaults_to_largest_supported_visible_gpu_count(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4")

    assert _default_torchrun_nproc("dreamx-world-5b-cam", {}) == 4


@pytest.mark.parametrize("nproc", [1, 2, 3, 4, 6, 8])
def test_dreamx_gpu_count_is_user_configurable(monkeypatch, nproc):
    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_NPROC", str(nproc))

    assert _requested_torchrun_nproc("dreamx-world-5b-cam", {}) == nproc


def test_dreamx_rejects_gpu_counts_that_do_not_divide_24_heads(monkeypatch):
    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_NPROC", "5")

    with pytest.raises(ValueError, match="supports nproc values 1, 2, 3, 4, 6, 8"):
        _requested_torchrun_nproc("dreamx-world-5b-cam", {})


def test_matrix_game3_ulysses_size_tracks_torchrun_world_size():
    result = _with_matrix_game3_parallelism(
        "matrix-game-3",
        {
            "load_kwargs_text": '{"ulysses_size": 1, "world_size": 2, "dit_fsdp": true}',
        },
        world_size=4,
    )

    assert json.loads(result["load_kwargs_text"]) == {"ulysses_size": 4, "dit_fsdp": True}


def test_lingbot_world_four_rank_default_uses_validated_offload_topology():
    result = _with_lingbot_world_parallelism(
        "lingbot-world",
        {
            "load_kwargs_text": json.dumps(
                {"t5_fsdp": True, "dit_fsdp": True, "t5_cpu": False}
            ),
            "call_kwargs_text": json.dumps({"offload_model": False, "num_frames": 161}),
        },
        world_size=4,
    )

    assert json.loads(result["load_kwargs_text"]) == {
        "t5_fsdp": True,
        "dit_fsdp": False,
        "t5_cpu": False,
        "ulysses_size": 4,
    }
    assert json.loads(result["call_kwargs_text"]) == {
        "offload_model": True,
        "num_frames": 161,
    }


def test_lingbot_world_eight_rank_default_keeps_official_fsdp_topology():
    result = _with_lingbot_world_parallelism(
        "lingbot-world",
        {
            "load_kwargs_text": json.dumps(
                {"t5_fsdp": True, "dit_fsdp": True, "t5_cpu": False}
            ),
            "call_kwargs_text": json.dumps({"offload_model": False}),
        },
        world_size=8,
    )

    assert json.loads(result["load_kwargs_text"]) == {
        "t5_fsdp": True,
        "dit_fsdp": True,
        "t5_cpu": False,
        "ulysses_size": 8,
    }
    assert json.loads(result["call_kwargs_text"]) == {"offload_model": False}


def test_longvie2_realtime_nproc_accepts_only_single_or_official_four(monkeypatch):
    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_NPROC", "4")
    assert _requested_torchrun_nproc("longvie-2", {}) == 4

    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_NPROC", "2")
    with pytest.raises(ValueError, match="supports nproc values 1 or 4"):
        _requested_torchrun_nproc("longvie-2", {})


def test_longvie2_same_python_still_dispatches_for_torchrun(monkeypatch):
    """LongVie-2 needs a child even in the unified env so torchrun can create four ranks."""
    env_prefix = Path(conda_dispatch.sys.executable).parent.parent
    spec = RuntimeCondaEnvSpec(
        model_id="longvie-2",
        env_name=env_prefix.name,
        env_root=env_prefix.parent,
    )
    monkeypatch.delenv("WORLDFOUNDRY_STUDIO_CONDA_CHILD", raising=False)
    monkeypatch.delenv(conda_dispatch.FORCE_SUBPROCESS_MODELS_ENV, raising=False)
    monkeypatch.setattr(conda_dispatch, "workspace_runtime_spec", lambda model_id: spec)

    assert conda_dispatch.dispatch_spec_for_inference("longvie-2") is spec


@pytest.mark.parametrize(
    ("world_size", "expected"),
    [
        (1, {"model_option": "keep", "use_usp": False, "ring_degree": 1, "ulysses_degree": 1}),
        (4, {"model_option": "keep", "use_usp": True, "ring_degree": 1, "ulysses_degree": 4}),
    ],
)
def test_longvie2_parallelism_tracks_supported_world_size(world_size, expected):
    result = _with_longvie2_parallelism(
        "longvie-2",
        {
            "load_kwargs_text": json.dumps(
                {
                    "model_option": "keep",
                    "use_usp": False,
                    "ring_degree": 7,
                    "ulysses_degree": 7,
                    "torchrun_nproc_per_node": world_size,
                }
            ),
        },
        world_size=world_size,
    )

    assert json.loads(result["load_kwargs_text"]) == expected
