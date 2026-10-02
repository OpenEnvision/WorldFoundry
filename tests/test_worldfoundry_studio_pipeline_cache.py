from __future__ import annotations

import tempfile
import os
import threading
import unittest
import weakref
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from worldfoundry.studio.inference.catalog import find_entry
from worldfoundry.studio.inference import execution as execution
from worldfoundry.studio.inference.execution import BaseRuntimeDriver, PipelineContext, PreparedInputs, StudioManager


def _request(tmp_path: Path, *, model_ref: str = "new-model") -> PreparedInputs:
    return PreparedInputs(
        prompt="",
        input_path="",
        image=None,
        image_path=None,
        video_path=None,
        last_frame=None,
        last_frame_path=None,
        reference_images=[],
        reference_image_paths=[],
        interactions=None,
        camera_view=None,
        task_type="",
        intrinsics=None,
        meta_path="",
        panorama_path="",
        scene_name="",
        fps=0,
        num_frames=0,
        output_dir=str(tmp_path),
        output_path=str(tmp_path / "preview.mp4"),
        call_kwargs={},
        load_kwargs={},
        model_ref=model_ref,
        backend="from_pretrained",
        endpoint="",
        api_key="",
        device="cpu",
    )


class _DummyPipeline:
    manager: StudioManager
    cache_sizes_during_load: list[int] = []

    @classmethod
    def from_pretrained(cls, **kwargs):
        cls.cache_sizes_during_load.append(len(cls.manager.pipeline_cache))
        return cls()


class _OldPipeline:
    pass


def test_api_clients_are_request_scoped_for_explicit_and_environment_credentials(tmp_path, monkeypatch):
    manager = StudioManager(workspace_root=str(tmp_path), max_cached_pipelines=1)
    entry = find_entry("infinite-world")
    driver = BaseRuntimeDriver()
    credentials = []

    class DummyApi:
        @classmethod
        def api_init(cls, api_key="", **kwargs):
            client = SimpleNamespace(api_key=api_key or os.environ.get("TEST_PROVIDER_API_KEY"))
            credentials.append(client.api_key)
            return client

    manager.import_pipeline_class = lambda _: _DummyPipeline
    _DummyPipeline.manager = manager
    _DummyPipeline.cache_sizes_during_load = []
    weights = driver.load_pipeline(manager, entry, _request(tmp_path))
    manager.import_pipeline_class = lambda _: DummyApi
    request = replace(_request(tmp_path), backend="api_init", endpoint="https://example.invalid")
    clients = [driver.load_pipeline(manager, entry, replace(request, api_key=key))
               for key in ("dummy-A", "dummy-B")]
    for key in ("env-A", "env-B"):
        monkeypatch.setenv("TEST_PROVIDER_API_KEY", key)
        clients.append(driver.load_pipeline(manager, entry, request))

    assert credentials == ["dummy-A", "dummy-B", "env-A", "env-B"]
    assert len({id(client) for client in clients}) == 4
    assert list(manager.pipeline_cache.values()) == [weights]
    manager.import_pipeline_class = lambda _: _DummyPipeline
    assert driver.load_pipeline(manager, entry, _request(tmp_path)) is weights


class WorldFoundryStudioPipelineCacheTest(unittest.TestCase):
    def test_cache_evicts_and_detaches_before_loading_next_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = StudioManager(workspace_root=tmp_dir, max_cached_pipelines=1)
            entry = find_entry("infinite-world")
            old_pipeline = _OldPipeline()
            old_pipeline_ref = weakref.ref(old_pipeline)
            old_context = PipelineContext(
                entry=entry,
                pipeline=old_pipeline,
                cache_key="old",
                backend="from_pretrained",
                model_ref="old",
                endpoint="",
                load_kwargs={},
                device="cpu",
            )
            del old_pipeline
            manager.pipeline_cache["old"] = old_context
            collected_after_release = []
            manager._collect_device_memory = (  # type: ignore[method-assign]
                lambda: collected_after_release.append(old_pipeline_ref() is None)
            )
            manager.import_pipeline_class = lambda selected: _DummyPipeline  # type: ignore[method-assign]
            _DummyPipeline.manager = manager
            _DummyPipeline.cache_sizes_during_load = []

            context = BaseRuntimeDriver().load_pipeline(manager, entry, _request(Path(tmp_dir)))

            self.assertEqual(_DummyPipeline.cache_sizes_during_load, [0])
            self.assertIsNone(old_context.pipeline)
            self.assertEqual(collected_after_release, [True])
            self.assertIs(manager.pipeline_cache[context.cache_key], context)

    def test_zero_sized_cache_does_not_retain_loaded_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = StudioManager(workspace_root=tmp_dir, max_cached_pipelines=0)
            entry = find_entry("infinite-world")
            manager.import_pipeline_class = lambda selected: _DummyPipeline  # type: ignore[method-assign]
            _DummyPipeline.manager = manager
            _DummyPipeline.cache_sizes_during_load = []

            context = BaseRuntimeDriver().load_pipeline(manager, entry, _request(Path(tmp_dir)))

            self.assertIsInstance(context.pipeline, _DummyPipeline)
            self.assertEqual(manager.pipeline_cache, {})

    def test_explicit_unload_detaches_context_before_disposal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = StudioManager(workspace_root=tmp_dir, max_cached_pipelines=1)
            entry = find_entry("infinite-world")
            pipeline = _OldPipeline()
            pipeline_ref = weakref.ref(pipeline)
            context = PipelineContext(
                entry=entry,
                pipeline=pipeline,
                cache_key="cached",
                backend="from_pretrained",
                model_ref="cached",
                endpoint="",
                load_kwargs={},
                device="cpu",
            )
            del pipeline
            manager.pipeline_cache[context.cache_key] = context
            collected_after_release = []
            manager._collect_device_memory = (  # type: ignore[method-assign]
                lambda: collected_after_release.append(pipeline_ref() is None)
            )

            message = manager._unload_local(entry.model_id)

            self.assertIn(entry.display_name, message)
            self.assertIsNone(context.pipeline)
            self.assertEqual(collected_after_release, [True])
            self.assertEqual(manager.pipeline_cache, {})

    def test_active_iterator_pins_pipeline_until_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = StudioManager(workspace_root=tmp_dir, max_cached_pipelines=1)
            entry = find_entry("infinite-world")
            pipeline = _OldPipeline()
            context = PipelineContext(
                entry=entry,
                pipeline=pipeline,
                cache_key="active",
                backend="from_pretrained",
                model_ref="active",
                endpoint="",
                load_kwargs={},
                device="cpu",
            )
            manager.pipeline_cache[context.cache_key] = context

            leased = manager._lease_lazy_pipeline_result(context, iter(("frame",)))

            self.assertEqual(context.active_leases, 1)
            with self.assertRaisesRegex(RuntimeError, "active lazy streams"):
                manager._reserve_pipeline_cache_slot()
            self.assertIs(context.pipeline, pipeline)

            leased.close()
            manager._reserve_pipeline_cache_slot()

            self.assertEqual(context.active_leases, 0)
            self.assertIsNone(context.pipeline)
            self.assertEqual(manager.pipeline_cache, {})

    def test_unload_is_deferred_until_active_iterator_closes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = StudioManager(workspace_root=tmp_dir, max_cached_pipelines=1)
            entry = find_entry("infinite-world")
            context = PipelineContext(
                entry=entry,
                pipeline=_OldPipeline(),
                cache_key="active",
                backend="from_pretrained",
                model_ref="active",
                endpoint="",
                load_kwargs={},
                device="cpu",
            )
            manager.pipeline_cache[context.cache_key] = context
            leased = manager._lease_lazy_pipeline_result(context, iter(("frame",)))

            message = manager._unload_local(entry.model_id)

            self.assertIn("Scheduled after active stream", message)
            self.assertIsNotNone(context.pipeline)
            leased.close()
            self.assertIsNone(context.pipeline)
            self.assertEqual(manager.pipeline_cache, {})

    def test_close_during_next_keeps_lease_until_next_returns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = StudioManager(workspace_root=tmp_dir, max_cached_pipelines=1)
            entry = find_entry("infinite-world")
            context = PipelineContext(
                entry=entry,
                pipeline=_OldPipeline(),
                cache_key="active",
                backend="from_pretrained",
                model_ref="active",
                endpoint="",
                load_kwargs={},
                device="cpu",
            )
            manager.pipeline_cache[context.cache_key] = context
            started = threading.Event()
            resume = threading.Event()

            def chunks():
                started.set()
                self.assertTrue(resume.wait(timeout=2))
                yield "frame"

            leased = manager._lease_lazy_pipeline_result(context, chunks())
            values: list[str] = []
            consumer = threading.Thread(target=lambda: values.append(next(leased)))
            consumer.start()
            self.assertTrue(started.wait(timeout=2))

            leased.close()

            self.assertEqual(context.active_leases, 1)
            self.assertIsNotNone(context.pipeline)
            self.assertIn("Scheduled", manager._unload_local(entry.model_id))
            resume.set()
            consumer.join(timeout=2)

            self.assertFalse(consumer.is_alive())
            self.assertEqual(values, ["frame"])
            self.assertEqual(context.active_leases, 0)
            self.assertIsNone(context.pipeline)
            leased.close()  # idempotent; must not double-release the lock/lease

    def test_uncached_lazy_pipeline_is_disposed_when_iterator_closes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = StudioManager(workspace_root=tmp_dir, max_cached_pipelines=0)
            context = PipelineContext(
                entry=find_entry("infinite-world"),
                pipeline=_OldPipeline(),
                cache_key="uncached",
                backend="from_pretrained",
                model_ref="uncached",
                endpoint="",
                load_kwargs={},
                device="cpu",
            )

            leased = manager._lease_lazy_pipeline_result(context, iter(("frame",)))
            leased.close()

            self.assertEqual(context.active_leases, 0)
            self.assertIsNone(context.pipeline)

    def test_torchrun_replication_is_gated_by_vram_and_preserves_explicit_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = StudioManager(workspace_root=tmp_dir)
            request = _request(Path(tmp_dir))

            def fake_torch(total_gib: int):
                return SimpleNamespace(
                    cuda=SimpleNamespace(
                        is_available=lambda: True,
                        device_count=lambda: 2,
                        get_device_properties=lambda index: SimpleNamespace(
                            total_memory=total_gib * 1024**3
                        ),
                    )
                )

            environment = {
                "WORLDFOUNDRY_STUDIO_TORCHRUN_LINGBOT_FAST": "1",
                "WORLDFOUNDRY_LINGBOT_REPLICATED_MIN_VRAM_GB": "72",
                "WORLD_SIZE": "2",
                "LOCAL_RANK": "0",
            }
            with (
                mock.patch.dict("os.environ", environment, clear=False),
                mock.patch(
                    "worldfoundry.studio.inference.execution._torch_module",
                    return_value=fake_torch(48),
                ),
            ):
                execution._torchrun_min_gpu_vram_gib.cache_clear()
                sharded = manager._torchrun_worker_request(find_entry("lingbot-world"), request)
            self.assertIs(sharded.load_kwargs["t5_fsdp"], True)
            self.assertIs(sharded.load_kwargs["dit_fsdp"], True)

            with (
                mock.patch.dict("os.environ", environment, clear=False),
                mock.patch(
                    "worldfoundry.studio.inference.execution._torch_module",
                    return_value=fake_torch(80),
                ),
            ):
                execution._torchrun_min_gpu_vram_gib.cache_clear()
                replicated = manager._torchrun_worker_request(find_entry("lingbot-world"), request)
            self.assertIs(replicated.load_kwargs["t5_fsdp"], False)
            self.assertIs(replicated.load_kwargs["dit_fsdp"], False)

            explicit = replace(
                request,
                load_kwargs={"t5_fsdp": True, "dit_fsdp": True},
                call_kwargs={"offload_model": True},
            )
            with (
                mock.patch.dict("os.environ", environment, clear=False),
                mock.patch(
                    "worldfoundry.studio.inference.execution._torch_module",
                    return_value=fake_torch(80),
                ),
            ):
                execution._torchrun_min_gpu_vram_gib.cache_clear()
                preserved = manager._torchrun_worker_request(find_entry("lingbot-world"), explicit)
            execution._torchrun_min_gpu_vram_gib.cache_clear()
            self.assertIs(preserved.load_kwargs["t5_fsdp"], True)
            self.assertIs(preserved.load_kwargs["dit_fsdp"], True)
            self.assertIs(preserved.call_kwargs["offload_model"], True)


if __name__ == "__main__":
    unittest.main()
