"""Benchmark-specific request and artifact-layout adapters.

The model-benchmark orchestrator resolves adapters from the ``generation`` block
of each benchmark catalog entry instead of growing benchmark-id conditionals.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Any

from worldfoundry.evals.api import GenerationRequest
from worldfoundry.evals.utils import BENCHMARK_ASSETS_ROOT


def materialize_vbench_generation_requests(
    *,
    limit: int | None = None,
    dataset_root: Path | None = None,
    split: str = "standard",
) -> tuple[GenerationRequest, ...]:
    """Materialize five seeded videos per official prompt; limit counts videos."""
    if split != "standard":
        raise ValueError("VBench generation supports the standard split")
    if limit is not None and limit <= 0:
        return ()
    source = (dataset_root or BENCHMARK_ASSETS_ROOT / "vbench") / "VBench_full_info.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError(f"VBench prompt suite must be a list: {source}")
    prompt_dimensions: dict[str, list[str]] = {}
    for index, row in enumerate(payload):
        if not isinstance(row, dict) or not row.get("prompt_en"):
            raise ValueError(f"invalid VBench prompt row {index}: {source}")
        prompt = str(row["prompt_en"])
        dimensions = prompt_dimensions.setdefault(prompt, [])
        dimensions.extend(str(item) for item in row.get("dimension") or () if str(item) not in dimensions)
    requests = []
    for index, (prompt, dimensions) in enumerate(prompt_dimensions.items()):
        for replicate in range(5):
            requests.append(
                GenerationRequest(
                    sample_id=f"vbench-{index:04d}-{replicate}",
                    task_name="vbench_t2v_standard",
                    split=split,
                    inputs={
                        "prompt": prompt,
                        "prompt_id": index,
                        "sample_index": replicate,
                        "dimensions": tuple(dimensions),
                        "official_video_name": f"{prompt}-{replicate}.mp4",
                    },
                    generation_kwargs={"seed": replicate},
                    output_schema={"generated_video": {"kind": "generated_video"}},
                )
            )
            if limit is not None and len(requests) >= limit:
                return tuple(requests)
    return tuple(requests)


def _load_target(target: str) -> Callable[..., Any]:
    module_name, separator, attribute = target.partition(":")
    if not separator:
        raise ValueError(f"adapter target must use module:attribute syntax: {target!r}")
    value = getattr(import_module(module_name), attribute)
    if not callable(value):
        raise TypeError(f"adapter target is not callable: {target!r}")
    return value


@dataclass(frozen=True)
class BenchmarkGenerationAdapter:
    """Thin bridge between a benchmark prompt suite and the common runner."""

    benchmark_id: str
    request_provider: str
    missing_requests_hint: str
    artifact_materializer: str | None = None
    dataset_root_parameter: str | None = None
    split_parameter: str | None = None

    def materialize_requests(
        self,
        *,
        limit: int | None,
        dataset_root: str | Path | None = None,
        split: str | None = None,
    ) -> tuple[GenerationRequest, ...]:
        kwargs: dict[str, Any] = {"limit": limit}
        if dataset_root is not None:
            if self.dataset_root_parameter is None:
                raise ValueError(
                    f"{self.benchmark_id} generation does not accept dataset_root; provide materialized requests"
                )
            kwargs[self.dataset_root_parameter] = Path(dataset_root)
        if split not in (None, "default"):
            if self.split_parameter is None:
                raise ValueError(f"{self.benchmark_id} generation does not accept a split override")
            kwargs[self.split_parameter] = split
        requests = _load_target(self.request_provider)(**kwargs)
        return tuple(requests)

    def materialize_artifacts(
        self,
        *,
        generation_output_dir: Path,
        generated_artifact_dir: Path,
        artifact_manifest_path: Path,
        output_artifact: str,
    ) -> tuple[int, int] | None:
        if self.artifact_materializer is None:
            return None
        return _load_target(self.artifact_materializer)(
            generation_output_dir=generation_output_dir,
            generated_artifact_dir=generated_artifact_dir,
            artifact_manifest_path=artifact_manifest_path,
            output_artifact=output_artifact,
        )


def _adapter_from_catalog_entry(benchmark_id: str, block: Mapping[str, Any]) -> BenchmarkGenerationAdapter:
    return BenchmarkGenerationAdapter(
        benchmark_id=benchmark_id,
        request_provider=str(block["request_provider"]),
        missing_requests_hint=str(block["missing_requests_hint"]),
        artifact_materializer=block.get("artifact_materializer"),
        dataset_root_parameter=block.get("dataset_root_parameter"),
        split_parameter=block.get("split_parameter"),
    )


@lru_cache(maxsize=1)
def list_benchmark_generation_adapters() -> dict[str, BenchmarkGenerationAdapter]:
    """Return the generation adapters declared by the ``generation`` block of catalog entries."""

    from worldfoundry.evals.tasks.catalog.benchmark_catalog import readable_catalog_shards

    adapters: dict[str, BenchmarkGenerationAdapter] = {}
    for shard in readable_catalog_shards():
        for entry in shard.entries:
            block = entry.get("generation") if isinstance(entry, Mapping) else None
            if isinstance(block, Mapping):
                benchmark_id = str(entry.get("id") or entry.get("benchmark_id")).casefold()
                adapters[benchmark_id] = _adapter_from_catalog_entry(benchmark_id, block)
    return adapters


def get_benchmark_generation_adapter(benchmark_id: str) -> BenchmarkGenerationAdapter | None:
    """Return the registered adapter for a canonical benchmark id, if any."""

    return list_benchmark_generation_adapters().get(benchmark_id.strip().casefold())


__all__ = [
    "BenchmarkGenerationAdapter",
    "get_benchmark_generation_adapter",
    "list_benchmark_generation_adapters",
    "materialize_vbench_generation_requests",
]
