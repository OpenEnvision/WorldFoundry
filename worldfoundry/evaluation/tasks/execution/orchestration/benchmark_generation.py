"""Benchmark-specific request and artifact-layout adapters.

The model-benchmark orchestrator deliberately depends on this small registry
instead of growing benchmark-id conditionals.  Most adapters follow the same
module/function naming convention; the registry only records the differences.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Sequence

from worldfoundry.evaluation.api import GenerationRequest
from worldfoundry.evaluation.utils import BENCHMARK_ASSETS_ROOT

_TARGET_PREFIX = "worldfoundry.evaluation.tasks.execution.runners"


def materialize_vbench_generation_requests(*, limit: int | None = None) -> tuple[GenerationRequest, ...]:
    """Materialize the checked-in VBench standard prompt suite for generation."""

    source = BENCHMARK_ASSETS_ROOT / "vbench" / "VBench_full_info.json"
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
        requests.append(
            GenerationRequest(
                sample_id=f"vbench-{index:04d}",
                task_name="vbench_t2v_standard",
                split="standard",
                inputs={
                    "prompt": prompt,
                    "prompt_id": index,
                    "dimensions": tuple(dimensions),
                    "official_video_name": f"{prompt}-0.mp4",
                },
                output_schema={"generated_video": {"kind": "generated_video"}},
            )
        )
        if limit is not None and len(requests) >= limit:
            break
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
        if self.dataset_root_parameter is not None and dataset_root is not None:
            kwargs[self.dataset_root_parameter] = Path(dataset_root)
        if self.split_parameter is not None and split is not None:
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


# benchmark id -> (module below runners, function prefix, missing-data hint,
#                    has custom artifact layout, optional dataset-root kwarg,
#                    optional split kwarg)
_ADAPTER_ROWS: dict[str, tuple[Any, ...]] = {
    "aigcbench": (
        "aigcbench.aigcbench_prompts",
        "aigcbench",
        "set WORLDFOUNDRY_AIGCBENCH_DATASET_ROOT or WORLDFOUNDRY_AIGCBENCH_PROMPT_MANIFEST",
        True,
    ),
    "evalcrafter": (
        "evalcrafter.evalcrafter_prompts",
        "evalcrafter",
        "restore the bundled EvalCrafter prompt700.txt asset",
        True,
    ),
    "ewmbench": (
        "ewmbench.ewmbench_prompts",
        "ewmbench",
        "restore the bundled EWMBench task_manifest.json asset",
        True,
    ),
    "fetv": (
        "fetv.fetv_prompts",
        "fetv",
        "restore the bundled FETV fetv_data.json prompt asset",
        True,
    ),
    "ipv-bench": (
        "ipv_bench.ipv_bench_prompts",
        "ipv_bench",
        "set WORLDFOUNDRY_IPV_BENCH_ROOT or WORLDFOUNDRY_IPV_BENCH_PROMPT_MANIFEST",
        False,
    ),
    "iworld-bench": (
        "iworldbench.iworldbench_prompts",
        "iworldbench",
        "set WORLDFOUNDRY_IWORLD_BENCH_DATASET_ROOT or WORLDFOUNDRY_IWORLD_BENCH_PROMPT_MANIFEST",
        False,
        "dataset_root",
        "split",
    ),
    "mirabench": (
        "mirabench.mirabench_prompts",
        "mirabench",
        "set WORLDFOUNDRY_MIRABENCH_ROOT or WORLDFOUNDRY_MIRABENCH_META_CSV",
        True,
    ),
    "phyeduvideo": (
        "phyeduvideo.phyeduvideo_prompts",
        "phyeduvideo",
        "set WORLDFOUNDRY_PHYEDUVIDEO_ROOT or WORLDFOUNDRY_PHYEDUVIDEO_PROMPTS_FILE",
        True,
    ),
    "phyfps-bench-gen": (
        "phyfps_bench_gen.phyfps_prompts",
        "phyfps",
        "set WORLDFOUNDRY_PHYFPS_BENCH_GEN_ROOT or WORLDFOUNDRY_PHYFPS_BENCH_GEN_PROMPT_MANIFEST",
        True,
    ),
    "phygenbench": (
        "phygenbench.phygenbench_prompts",
        "phygenbench",
        "set WORLDFOUNDRY_PHYGENBENCH_ROOT or WORLDFOUNDRY_PHYGENBENCH_PROMPT_MANIFEST",
        True,
    ),
    "phyground": (
        "phyground.phyground_prompts",
        "phyground",
        "set WORLDFOUNDRY_PHYGROUND_DATA_ROOT or WORLDFOUNDRY_PHYGROUND_PROMPT_MANIFEST",
        True,
    ),
    "physics-iq": (
        "physics_iq.physics_iq_prompts",
        "physics_iq",
        "set WORLDFOUNDRY_PHYSICS_IQ_ROOT or WORLDFOUNDRY_PHYSICS_IQ_DESCRIPTIONS",
        True,
        "dataset_root",
    ),
    "physics-iq-verified": (
        "physics_iq.physics_iq_prompts",
        "physics_iq_verified",
        "restore the bundled Physics-IQ Verified best-practice descriptions",
        True,
        "dataset_root",
    ),
    "physvidbench": (
        "physvidbench.physvidbench_prompts",
        "physvidbench",
        "set WORLDFOUNDRY_PHYSVIDBENCH_ROOT or WORLDFOUNDRY_PHYSVIDBENCH_PROMPT_MANIFEST",
        True,
    ),
    "t2v-safety-bench": (
        "t2v_safety_bench.t2v_safety_bench_prompts",
        "t2v_safety_bench",
        "restore the bundled T2VSafetyBench prompt files",
        False,
    ),
    "t2v-compbench": (
        "t2v_compbench.t2v_compbench_prompts",
        "t2v_compbench",
        "restore the bundled T2V-CompBench metadata assets",
        True,
    ),
    "videophy": (
        "videophy.videophy_prompts",
        "videophy",
        "restore the bundled VideoPhy prompt assets",
        True,
    ),
    "videophy2": (
        "videophy2.videophy2_prompts",
        "videophy2",
        "restore the bundled VideoPhy2 prompt assets",
        True,
    ),
    "videoverse": (
        "videoverse.videoverse_prompts",
        "videoverse",
        "set WORLDFOUNDRY_VIDEOVERSE_ROOT or WORLDFOUNDRY_VIDEOVERSE_PROMPT_MANIFEST",
        True,
    ),
    "vmbench": (
        "vmbench.vmbench_prompts",
        "vmbench",
        "restore the bundled VMBench prompt assets",
        False,
    ),
    "world-in-world": (
        "world_in_world.world_in_world_prompts",
        "world_in_world",
        "set WORLDFOUNDRY_WORLD_IN_WORLD_ASSETS_ROOT or an episode manifest",
        False,
    ),
    "wrbench": (
        "wrbench.wrbench_prompts",
        "wrbench",
        "restore the WRBench Natural-25 assets and first frames",
        True,
    ),
}


def _adapter_from_row(benchmark_id: str, row: Sequence[Any]) -> BenchmarkGenerationAdapter:
    module_suffix, function_prefix, hint, custom_layout, *options = row
    module = f"{_TARGET_PREFIX}.{module_suffix}"
    return BenchmarkGenerationAdapter(
        benchmark_id=benchmark_id,
        request_provider=f"{module}:materialize_{function_prefix}_generation_requests",
        artifact_materializer=(f"{module}:copy_{function_prefix}_generated_videos" if custom_layout else None),
        missing_requests_hint=str(hint),
        dataset_root_parameter=str(options[0]) if options else None,
        split_parameter=str(options[1]) if len(options) > 1 else None,
    )


BENCHMARK_GENERATION_ADAPTERS = {
    benchmark_id: _adapter_from_row(benchmark_id, row) for benchmark_id, row in _ADAPTER_ROWS.items()
}
BENCHMARK_GENERATION_ADAPTERS["vbench"] = BenchmarkGenerationAdapter(
    benchmark_id="vbench",
    request_provider=(
        "worldfoundry.evaluation.tasks.execution.orchestration.benchmark_generation:"
        "materialize_vbench_generation_requests"
    ),
    missing_requests_hint="restore the bundled VBench_full_info.json prompt suite",
)


def get_benchmark_generation_adapter(benchmark_id: str) -> BenchmarkGenerationAdapter | None:
    """Return the registered adapter for a canonical benchmark id, if any."""

    return BENCHMARK_GENERATION_ADAPTERS.get(benchmark_id.strip().casefold())


def benchmark_generation_ready(benchmark_id: str) -> bool:
    """Return whether WorldFoundry can materialize official generation requests."""

    return get_benchmark_generation_adapter(benchmark_id) is not None


__all__ = [
    "BENCHMARK_GENERATION_ADAPTERS",
    "BenchmarkGenerationAdapter",
    "benchmark_generation_ready",
    "get_benchmark_generation_adapter",
    "materialize_vbench_generation_requests",
]
