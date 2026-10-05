"""Readiness payloads for benchmark datasets and runtime profiles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from worldfoundry.evaluation.tasks.datasets import check_local_dataset

from .context import MCPToolContext, get_default_context


def check_benchmark_datasets_payload(
    *,
    benchmark_id: str,
    data_root: str | Path | None = None,
    context: MCPToolContext | None = None,
) -> dict[str, Any]:
    """Check local Hugging Face dataset readiness for one benchmark."""

    from worldfoundry.evaluation.tasks.catalog.zoo_registry import load_benchmark_zoo_registry

    ctx = context or get_default_context()
    registry = load_benchmark_zoo_registry(ctx.benchmark_manifest_dir)
    entry = registry.get(benchmark_id)
    cache_dir = Path(data_root).expanduser() if data_root else _default_dataset_root()
    refs = [entry.dataset, *entry.dataset_refs]
    seen: set[tuple[str | None, str | None, str | None, str | None]] = set()
    results: list[dict[str, Any]] = []
    for ref in refs:
        if not ref.hf_dataset_id:
            continue
        key = (ref.hf_dataset_id, ref.revision, ref.split, ref.path)
        if key in seen:
            continue
        seen.add(key)
        local = check_local_dataset(ref, cache_dir)
        public_status = _public_dataset_status(local)
        results.append(
            {
                "hf_dataset_id": ref.hf_dataset_id,
                "revision": ref.revision,
                "split": ref.split,
                "path": ref.path,
                "ready": local.ready,
                "status": public_status,
                "local_layout": local.local_layout,
                "reason": local.reason,
                "details": local.to_dict(),
            }
        )

    by_status: dict[str, int] = {}
    ready_count = 0
    for item in results:
        status = str(item["status"])
        by_status[status] = by_status.get(status, 0) + 1
        if item["ready"]:
            ready_count += 1
    not_ready = len(results) - ready_count
    return {
        "benchmark_id": entry.benchmark_id,
        "data_root": str(cache_dir),
        "results": results,
        "summary": {
            "total": len(results),
            "ready": ready_count,
            "not_ready": not_ready,
            "by_status": by_status,
        },
    }


def _default_dataset_root() -> Path:
    """Anchor the dataset default to the framework data root, not the CWD.

    The MCP server's working directory is client-controlled, so the previous
    ``datasets`` relative default pointed at an unpredictable location and
    reported every benchmark as not-ready (CM-29).
    """

    from worldfoundry.core.io.paths import local_data_root_path

    return local_data_root_path()


def _public_dataset_status(local: Any) -> str:
    if local.ready:
        return "ready"
    if local.status in {"not_found", "missing_snapshot"} or local.local_layout == "missing":
        return "missing"
    return str(local.status)


__all__ = ["check_benchmark_datasets_payload"]
