from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
from PIL import Image

import worldfoundry.studio.inference.execution as execution
from worldfoundry.studio.inference.catalog import CatalogEntry
from worldfoundry.studio.inference.execution import (
    BaseRuntimeDriver,
    PreparedInputs,
    StudioManager,
    _prepared_inputs_from_payload,
    _prepared_inputs_payload,
)
from worldfoundry.studio.ui.launch_config import StudioLaunchConfig, launch_uses_lingbot_torchrun_rollout


def _entry(model_id: str = "matrix-game-3") -> CatalogEntry:
    return CatalogEntry(
        model_id=model_id,
        display_name=model_id,
        module_path="tests.fake_pipeline",
        class_name="FakePipeline",
        family="world_model",
        category="World Model",
        summary="test entry",
        supports_stream=True,
        supports_from_pretrained=True,
    )


def _request(tmp_path, **overrides) -> PreparedInputs:
    values = {
        "prompt": "world",
        "input_path": "",
        "image": None,
        "image_path": None,
        "video_path": None,
        "last_frame": None,
        "last_frame_path": None,
        "reference_images": [],
        "reference_image_paths": [],
        "interactions": ["forward"],
        "camera_view": None,
        "task_type": "",
        "intrinsics": None,
        "meta_path": "",
        "panorama_path": "",
        "scene_name": "",
        "fps": 16,
        "num_frames": 9,
        "output_dir": str(tmp_path / "output"),
        "output_path": str(tmp_path / "output" / "video.mp4"),
        "call_kwargs": {},
        "load_kwargs": {},
        "model_ref": "",
        "backend": "from_pretrained",
        "endpoint": "",
        "api_key": "",
        "device": "cuda",
    }
    values.update(overrides)
    return PreparedInputs(**values)


def _enable_torchrun(monkeypatch, *, world_size: int, rank: int = 0, local_rank: int = 0) -> None:
    monkeypatch.setenv(execution.TORCHRUN_LINGBOT_FAST_ENV, "1")
    monkeypatch.setenv("WORLD_SIZE", str(world_size))
    monkeypatch.setenv("RANK", str(rank))
    monkeypatch.setenv("LOCAL_RANK", str(local_rank))


def test_matrix_game3_launch_enters_torchrun_bridge_only_for_multiple_ranks(monkeypatch):
    launch = StudioLaunchConfig(model_id="matrix-game-3", frontend="world")

    monkeypatch.setenv("WORLD_SIZE", "4")
    assert launch_uses_lingbot_torchrun_rollout(launch)

    monkeypatch.setenv("WORLD_SIZE", "1")
    assert not launch_uses_lingbot_torchrun_rollout(launch)


def test_resident_torchrun_control_group_outlives_idle_ui_sessions(monkeypatch):
    _enable_torchrun(monkeypatch, world_size=4)
    calls = []
    group = object()

    class _FakeDist:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def new_group(*, ranks, backend, timeout):
            calls.append((ranks, backend, timeout))
            return group

    monkeypatch.setattr(execution, "_torch_dist", lambda: _FakeDist())
    monkeypatch.setattr(execution, "_TORCHRUN_CONTROL_GROUP", None)

    assert execution._torchrun_control_group() is group
    ranks, backend, timeout = calls.pop()
    assert ranks == [0, 1, 2, 3]
    assert backend == "gloo"
    assert timeout.days >= 365


@pytest.mark.parametrize("error_type", (TypeError, RuntimeError))
def test_control_group_failure_is_reported_without_retry(monkeypatch, error_type):
    _enable_torchrun(monkeypatch, world_size=4)
    error = error_type("control group initialization failed")
    calls = []
    group = object()

    def new_group(*, ranks, backend, timeout):
        calls.append((ranks, backend, timeout))
        if len(calls) == 1:
            raise error
        return group

    dist = SimpleNamespace(is_available=lambda: True, is_initialized=lambda: True, new_group=new_group)
    monkeypatch.setattr(execution, "_torch_dist", lambda: dist)
    monkeypatch.setattr(execution, "_TORCHRUN_CONTROL_GROUP", None)

    with pytest.raises(error_type) as raised:
        execution._torchrun_control_group()
    assert raised.value is error
    assert len(calls) == 1
    assert execution._TORCHRUN_CONTROL_GROUP is None
    assert not execution._TORCHRUN_CONTROL_GROUP_CREATING
    # An explicit later initialization remains possible after failure cleanup.
    assert execution._torchrun_control_group() is group


def test_shutdown_rejects_control_group_created_by_an_older_generation(monkeypatch):
    _enable_torchrun(monkeypatch, world_size=4)
    creation_started = threading.Event()
    allow_creation = threading.Event()
    group = object()
    new_group_calls = 0
    destroyed = []

    class _FakeDist:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def new_group(*, ranks, backend, timeout):
            nonlocal new_group_calls
            new_group_calls += 1
            creation_started.set()
            assert allow_creation.wait(timeout=5)
            return group

        @staticmethod
        def destroy_process_group(process_group=None):
            destroyed.append(process_group)

    monkeypatch.setattr(execution, "_torch_dist", lambda: _FakeDist())
    monkeypatch.setattr(execution, "_TORCHRUN_CONTROL_GROUP", None)
    monkeypatch.setattr(execution, "_TORCHRUN_CONTROL_GROUP_STATE", "open")
    monkeypatch.setattr(execution, "_TORCHRUN_CONTROL_GROUP_GENERATION", 0)
    monkeypatch.setattr(execution, "_TORCHRUN_CONTROL_GROUP_CREATING", False)

    created = []
    creator = threading.Thread(
        target=lambda: created.append(execution._torchrun_control_group())
    )
    creator.start()
    assert creation_started.wait(timeout=5)

    shutdown = threading.Thread(target=execution.shutdown_torchrun_lingbot_fast_runtime)
    shutdown.start()
    with execution._TORCHRUN_CONTROL_GROUP_CONDITION:
        assert execution._TORCHRUN_CONTROL_GROUP_CONDITION.wait_for(
            lambda: execution._TORCHRUN_CONTROL_GROUP_STATE == "closing",
            timeout=5,
        )

    # Closing callers must not start a replacement or receive the group that
    # the active shutdown generation is about to destroy.
    assert execution._torchrun_control_group() is None
    assert new_group_calls == 1
    allow_creation.set()
    creator.join(timeout=5)
    shutdown.join(timeout=5)

    assert not creator.is_alive()
    assert not shutdown.is_alive()
    assert created == [None]
    assert destroyed == [group, None]
    assert execution._TORCHRUN_CONTROL_GROUP is None
    assert execution._TORCHRUN_CONTROL_GROUP_STATE == "open"


def test_control_group_stays_closed_until_default_group_destruction_finishes(monkeypatch):
    _enable_torchrun(monkeypatch, world_size=4)
    old_group = object()
    new_group = object()
    default_destroy_started = threading.Event()
    allow_default_destroy = threading.Event()
    new_group_calls = 0

    class _FakeDist:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def new_group(*, ranks, backend, timeout):
            nonlocal new_group_calls
            new_group_calls += 1
            return new_group

        @staticmethod
        def destroy_process_group(process_group=None):
            if process_group is None:
                default_destroy_started.set()
                assert allow_default_destroy.wait(timeout=5)

    monkeypatch.setattr(execution, "_torch_dist", lambda: _FakeDist())
    monkeypatch.setattr(execution, "_TORCHRUN_CONTROL_GROUP", old_group)
    monkeypatch.setattr(execution, "_TORCHRUN_CONTROL_GROUP_STATE", "open")
    monkeypatch.setattr(execution, "_TORCHRUN_CONTROL_GROUP_GENERATION", 0)
    monkeypatch.setattr(execution, "_TORCHRUN_CONTROL_GROUP_CREATING", False)

    shutdown = threading.Thread(target=execution.shutdown_torchrun_lingbot_fast_runtime)
    shutdown.start()
    assert default_destroy_started.wait(timeout=5)

    assert execution._torchrun_control_group() is None
    assert new_group_calls == 0
    allow_default_destroy.set()
    shutdown.join(timeout=5)

    assert not shutdown.is_alive()
    assert execution._TORCHRUN_CONTROL_GROUP_STATE == "open"
    assert execution._torchrun_control_group() is new_group
    assert new_group_calls == 1


def test_longvie2_launch_accepts_single_or_official_four_rank_only(monkeypatch):
    launch = StudioLaunchConfig(model_id="longvie-2", frontend="auto")

    monkeypatch.setenv("WORLD_SIZE", "4")
    assert launch_uses_lingbot_torchrun_rollout(launch)

    monkeypatch.setenv("WORLD_SIZE", "1")
    assert not launch_uses_lingbot_torchrun_rollout(launch)

    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="one GPU or the official four-rank USP"):
        launch_uses_lingbot_torchrun_rollout(launch)


def test_world_frontend_nonzero_matrix_game3_rank_enters_worker_loop(monkeypatch):
    from worldfoundry.studio.visualization.backends import world as world_backend

    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "2")
    calls: list[str] = []

    class _FakeManager:
        def run_torchrun_worker_loop(self):
            calls.append("worker")

    monkeypatch.setattr(world_backend, "StudioManager", _FakeManager)
    monkeypatch.setattr(
        world_backend,
        "ensure_torchrun_lingbot_fast_control_group",
        lambda: calls.append("control_group") or True,
    )
    monkeypatch.setattr(
        world_backend,
        "shutdown_torchrun_lingbot_fast_runtime",
        lambda: calls.append("shutdown"),
    )

    world_backend.serve_world_frontend(
        _entry(),
        StudioLaunchConfig(model_id="matrix-game-3", frontend="world"),
        host="127.0.0.1",
        port=7868,
        access_printer=lambda *_args: calls.append("server"),
    )

    assert calls == ["control_group", "worker", "shutdown"]


def test_matrix_game3_worker_request_sets_only_rank_device_and_ulysses(monkeypatch, tmp_path):
    _enable_torchrun(monkeypatch, world_size=4, rank=2, local_rank=2)
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 4,
        )
    )
    monkeypatch.setattr(execution, "torch", fake_torch)
    manager = StudioManager(workspace_root=str(tmp_path / "studio"))
    request = _request(
        tmp_path,
        load_kwargs={"world_size": 99, "model_option": "keep"},
        call_kwargs={"num_inference_steps": 4},
    )

    worker_request = manager._torchrun_worker_request(_entry(), request)

    assert worker_request.device == "cuda:2"
    assert worker_request.load_kwargs == {
        "rank": 2,
        "ulysses_size": 4,
        "model_option": "keep",
    }
    assert worker_request.call_kwargs == {"num_inference_steps": 4}
    for lingbot_key in ("t5_fsdp", "dit_fsdp", "t5_cpu"):
        assert lingbot_key not in worker_request.load_kwargs
    assert "offload_model" not in worker_request.call_kwargs
    assert request.load_kwargs == {"world_size": 99, "model_option": "keep"}


def test_longvie2_worker_request_pins_rank_and_official_usp_topology(monkeypatch, tmp_path):
    _enable_torchrun(monkeypatch, world_size=4, rank=3, local_rank=3)
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 4,
        )
    )
    monkeypatch.setattr(execution, "torch", fake_torch)
    manager = StudioManager(workspace_root=str(tmp_path / "studio"))
    request = _request(
        tmp_path,
        load_kwargs={"model_option": "keep", "use_usp": False, "ulysses_degree": 1},
    )

    worker_request = manager._torchrun_worker_request(_entry("longvie-2"), request)

    assert worker_request.device == "cuda:3"
    assert worker_request.load_kwargs == {
        "rank": 3,
        "model_option": "keep",
        "use_usp": True,
        "ring_degree": 1,
        "ulysses_degree": 4,
    }
    for lingbot_key in ("t5_fsdp", "dit_fsdp", "t5_cpu"):
        assert lingbot_key not in worker_request.load_kwargs


def test_longvie2_distributed_dispatch_uses_resident_command_bridge(monkeypatch, tmp_path):
    _enable_torchrun(monkeypatch, world_size=4)
    manager = StudioManager(workspace_root=str(tmp_path / "studio"))
    entry = _entry("longvie-2")

    assert manager._should_use_torchrun_lingbot_fast(entry, _request(tmp_path))
    assert not manager._should_use_torchrun_lingbot_fast(
        entry,
        _request(tmp_path, backend="api_init"),
    )


def test_longvie2_rank_keeps_one_pipeline_across_run_stream_and_reset(monkeypatch, tmp_path):
    _enable_torchrun(monkeypatch, world_size=4, rank=1, local_rank=1)
    monkeypatch.setattr(
        execution,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )
    manager = StudioManager(workspace_root=str(tmp_path / "studio"))
    entry = _entry("longvie-2")

    class _ResidentLongVie:
        instances = 0

        def __init__(self):
            type(self).instances += 1
            self.segments = 0

        @classmethod
        def from_pretrained(cls, **_kwargs):
            return cls()

        def __call__(self, **_kwargs):
            self.segments = 1
            return {"video": [Image.new("RGB", (2, 2), "red")]}

        def stream(self, **_kwargs):
            assert self.segments == 1
            self.segments += 1
            return {"video": [Image.new("RGB", (2, 2), "blue")]}

        def reset_memory(self):
            self.segments = 0

        def stream_state_ready(self):
            return self.segments > 0

    monkeypatch.setattr(manager, "import_pipeline_class", lambda _entry: _ResidentLongVie)
    request = _request(tmp_path, load_kwargs={"model_id": "longvie-2"})

    first = manager._run_local_torchrun_lingbot_fast(
        entry=entry,
        action="run",
        request=request,
        materialize=False,
    )
    second = manager._run_local_torchrun_lingbot_fast(
        entry=entry,
        action="stream",
        request=request,
        materialize=False,
    )
    context = next(iter(manager.pipeline_cache.values()))
    assert context.state["studio_stream_initialized"] is True
    manager._run_local_torchrun_lingbot_fast(
        entry=entry,
        action="reset",
        request=request,
        materialize=False,
    )

    assert _ResidentLongVie.instances == 1
    assert len(manager.pipeline_cache) == 1
    assert first["video"][0].getpixel((0, 0)) == (255, 0, 0)
    assert second["video"][0].getpixel((0, 0)) == (0, 0, 255)
    assert context.pipeline.segments == 0
    assert context.state == {}


def test_longvie2_public_reset_is_broadcast_to_every_resident_rank(monkeypatch, tmp_path):
    _enable_torchrun(monkeypatch, world_size=4)
    manager = StudioManager(workspace_root=str(tmp_path / "studio"))
    commands = []
    monkeypatch.setattr(manager, "_run_torchrun_command", lambda command: commands.append(command) or "reset")

    assert manager.reset_cached_model("longvie-2") == "reset"
    assert commands == [{"kind": "reset_model", "model_id": "longvie-2"}]


def test_matrix_game3_distributed_dispatch_requires_non_api_multi_rank(monkeypatch, tmp_path):
    _enable_torchrun(monkeypatch, world_size=4)
    manager = StudioManager(workspace_root=str(tmp_path / "studio"))
    entry = _entry()

    assert manager._should_use_torchrun_lingbot_fast(entry, _request(tmp_path))
    assert not manager._should_use_torchrun_lingbot_fast(
        entry,
        _request(tmp_path, backend="api_init"),
    )

    monkeypatch.setenv("WORLD_SIZE", "1")
    assert not manager._should_use_torchrun_lingbot_fast(entry, _request(tmp_path))


def test_single_rank_worker_request_stays_local(monkeypatch, tmp_path):
    _enable_torchrun(monkeypatch, world_size=1)
    manager = StudioManager(workspace_root=str(tmp_path / "studio"))
    request = _request(tmp_path, device="cuda:3", load_kwargs={"model_option": "keep"})

    assert manager._torchrun_worker_request(_entry(), request) is request


def test_uploaded_image_survives_worker_payload_via_shared_image_path(tmp_path):
    manager = StudioManager(workspace_root=str(tmp_path / "studio"))
    request = manager.prepare_inputs(
        entry=_entry(),
        prompt="world",
        input_path="",
        image=Image.new("RGB", (8, 6), "red"),
        video=None,
        last_frame=None,
        reference_files=None,
        interactions_text="forward",
        camera_view_text="",
        task_type="",
        intrinsics_text="",
        meta_path="",
        panorama_path="",
        scene_name="",
        fps=16,
        num_frames=9,
        call_kwargs_text="{}",
        load_kwargs_text="{}",
        model_ref="",
        backend="from_pretrained",
        endpoint="",
        api_key="",
        device="cuda",
    )

    worker_request = _prepared_inputs_from_payload(_prepared_inputs_payload(request))
    resolved = BaseRuntimeDriver()._resolve_init_image(worker_request)

    assert worker_request.image is None
    assert worker_request.image_path == request.image_path
    assert resolved is not None
    assert resolved.size == (8, 6)
    assert resolved.getpixel((0, 0)) == (255, 0, 0)


def test_nonzero_rank_accepts_matrix_game3_result_without_video(monkeypatch, tmp_path):
    _enable_torchrun(monkeypatch, world_size=2, rank=1, local_rank=1)
    manager = StudioManager(workspace_root=str(tmp_path / "studio"))

    class _FakeDist:
        def all_gather_object(self, statuses, local_status, *, group):
            del group
            statuses[:] = [
                {"rank": 0, "error": ""},
                local_status,
            ]

    fake_dist = _FakeDist()
    monkeypatch.setattr(manager, "_torchrun_dist", lambda: fake_dist)
    monkeypatch.setattr(execution, "_torchrun_control_group", lambda: object())
    monkeypatch.setattr(execution, "find_entry", lambda _model_id: _entry())
    monkeypatch.setattr(
        manager,
        "_run_local_torchrun_lingbot_fast",
        lambda **_kwargs: {"video": None, "realtime_metrics": {"rank": 1}},
    )
    request = _request(tmp_path)

    result = manager._execute_torchrun_command(
        {
            "kind": "realtime_action",
            "model_id": "matrix-game-3",
            "action": "stream",
            "request": _prepared_inputs_payload(request),
        }
    )

    assert result is None
