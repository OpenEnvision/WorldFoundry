from __future__ import annotations

from types import SimpleNamespace

from worldfoundry.studio.inference.execution import PipelineContext, StudioManager


class _CleanupPipeline:
    def __init__(self) -> None:
        self.cleanup_calls = 0

    def cleanup(self) -> None:
        self.cleanup_calls += 1


def _context(pipeline: _CleanupPipeline, *, active_leases: int = 0) -> PipelineContext:
    return PipelineContext(
        entry=SimpleNamespace(model_id="cleanup-test", display_name="Cleanup test"),
        pipeline=pipeline,
        cache_key="cleanup-test",
        backend="from_pretrained",
        model_ref="",
        endpoint="",
        load_kwargs={},
        device="cpu",
        active_leases=active_leases,
    )


def test_close_runs_cached_pipeline_cleanup_once(tmp_path) -> None:
    manager = StudioManager(workspace_root=str(tmp_path), max_cached_pipelines=1)
    pipeline = _CleanupPipeline()
    context = _context(pipeline)
    manager.pipeline_cache[context.cache_key] = context

    manager.close()
    manager.close()

    assert pipeline.cleanup_calls == 1
    assert context.pipeline is None
    assert not manager.pipeline_cache


def test_close_defers_cleanup_until_active_lease_finishes(tmp_path) -> None:
    manager = StudioManager(workspace_root=str(tmp_path), max_cached_pipelines=1)
    pipeline = _CleanupPipeline()
    context = _context(pipeline, active_leases=1)
    manager.pipeline_cache[context.cache_key] = context

    manager.close()

    assert pipeline.cleanup_calls == 0
    assert context.dispose_when_idle is True
    manager._release_pipeline_lease(context)
    assert pipeline.cleanup_calls == 1
    assert context.pipeline is None
    assert not manager.pipeline_cache
