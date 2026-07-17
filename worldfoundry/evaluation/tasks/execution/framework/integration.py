"""Benchmark integration policy.

WorldFoundry ships scorer/normalizer logic under ``evaluation/tasks/execution/runners/``
and minimal official prompts/rubrics under ``data/benchmarks/assets/<benchmark_id>/``.
Runners resolve bundled assets by default; ``WORLDFOUNDRY_*`` env vars override only
when explicitly set. Release-facing benchmark execution should use checked-in runners,
WorldFoundry model artifacts, and explicit result imports.
Each benchmark picks the simplest workable path: pure in-tree scorers when the official
logic is small, model-backed judges via the catalog and HF/cache paths, or
normalizer-only imports when external judge services or unreleased assets are required.

Sections:

* **IntegrationTier** — in-tree, model-backed, or normalizer-only.
* **BenchmarkIntegrationSpec** — per-benchmark tier, runner script, optional asset hints.
* **BENCHMARK_INTEGRATION_REGISTRY** — canonical integration lookup table.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from worldfoundry.evaluation.tasks.execution.framework.runner_registry import VIDEO_RUNNER_REGISTRY
from worldfoundry.evaluation.utils import REPO_ROOT


class IntegrationTier(str, Enum):
    """Benchmark execution tier inside WorldFoundry."""

    IN_TREE = "in_tree"
    MODEL_BACKED = "model_backed"
    NORMALIZER_ONLY = "normalizer_only"


@dataclass(frozen=True)
class BenchmarkIntegrationSpec:
    """Integration metadata for one video benchmark."""

    benchmark_id: str
    tier: IntegrationTier
    runner_script: str
    hf_dataset_id: str | None = None
    judge_model_id: str | None = None

    @property
    def in_tree_runner(self) -> Path:
        """Absolute path to the checked-in runner script."""
        return REPO_ROOT / self.runner_script


def integration_spec(benchmark_id: str) -> BenchmarkIntegrationSpec | None:
    """Return integration spec for ``benchmark_id``, if registered."""
    return BENCHMARK_INTEGRATION_REGISTRY.get(benchmark_id)


def _runner_path(benchmark_id: str) -> str:
    """Resolve runner script path from :data:`VIDEO_RUNNER_REGISTRY`."""
    reg = VIDEO_RUNNER_REGISTRY.get(benchmark_id)
    if reg is None:
        raise KeyError(f"unknown video benchmark: {benchmark_id}")
    return reg.script


# ---------------------------------------------------------------------------
# Benchmark integration registry
# ---------------------------------------------------------------------------

BENCHMARK_INTEGRATION_REGISTRY: dict[str, BenchmarkIntegrationSpec] = {
    "aigcbench": BenchmarkIntegrationSpec(
        "aigcbench",
        IntegrationTier.IN_TREE,
        _runner_path("aigcbench"),
        hf_dataset_id="stevenfan/AIGCBench_v1.0",
    ),
    "camerabench": BenchmarkIntegrationSpec(
        "camerabench",
        IntegrationTier.MODEL_BACKED,
        _runner_path("camerabench"),
        hf_dataset_id="linzhiqiu/camerabench",
    ),
    "chronomagic-bench": BenchmarkIntegrationSpec(
        "chronomagic-bench",
        IntegrationTier.MODEL_BACKED,
        _runner_path("chronomagic-bench"),
        hf_dataset_id="BestWishYsh/ChronoMagic-Bench",
    ),
    "devil-dynamics": BenchmarkIntegrationSpec(
        "devil-dynamics",
        IntegrationTier.MODEL_BACKED,
        _runner_path("devil-dynamics"),
        judge_model_id="gemini",
    ),
    "evalcrafter": BenchmarkIntegrationSpec(
        "evalcrafter",
        IntegrationTier.IN_TREE,
        _runner_path("evalcrafter"),
    ),
    "ewmbench": BenchmarkIntegrationSpec(
        "ewmbench",
        IntegrationTier.IN_TREE,
        _runner_path("ewmbench"),
    ),
    "fetv": BenchmarkIntegrationSpec(
        "fetv", IntegrationTier.IN_TREE, _runner_path("fetv"),
    ),
    "genai-bench": BenchmarkIntegrationSpec(
        "genai-bench",
        IntegrationTier.IN_TREE,
        _runner_path("genai-bench"),
    ),
    "ipv-bench": BenchmarkIntegrationSpec(
        "ipv-bench", IntegrationTier.IN_TREE, _runner_path("ipv-bench"),
    ),
    "iworld-bench": BenchmarkIntegrationSpec(
        "iworld-bench", IntegrationTier.IN_TREE, _runner_path("iworld-bench"),
    ),
    "mirabench": BenchmarkIntegrationSpec(
        "mirabench", IntegrationTier.IN_TREE, _runner_path("mirabench"),
    ),
    "memobench": BenchmarkIntegrationSpec(
        "memobench",
        IntegrationTier.IN_TREE,
        _runner_path("memobench"),
    ),
    "phyeduvideo": BenchmarkIntegrationSpec(
        "phyeduvideo", IntegrationTier.IN_TREE, _runner_path("phyeduvideo"),
    ),
    "phyfps-bench-gen": BenchmarkIntegrationSpec(
        "phyfps-bench-gen",
        IntegrationTier.MODEL_BACKED,
        _runner_path("phyfps-bench-gen"),
        judge_model_id="xiangbog/Visual_Chronometer",
    ),
    "visual-chronometer": BenchmarkIntegrationSpec(
        "visual-chronometer",
        IntegrationTier.MODEL_BACKED,
        _runner_path("visual-chronometer"),
        judge_model_id="xiangbog/Visual_Chronometer",
    ),
    "phygenbench": BenchmarkIntegrationSpec(
        "phygenbench",
        IntegrationTier.MODEL_BACKED,
        _runner_path("phygenbench"),
        judge_model_id="gpt-4o-2024-05-13",
    ),
    "phyground": BenchmarkIntegrationSpec(
        "phyground",
        IntegrationTier.MODEL_BACKED,
        _runner_path("phyground"),
        hf_dataset_id="NU-World-Model-Embodied-AI/phyground",
        judge_model_id="NU-World-Model-Embodied-AI/phyjudge-9B",
    ),
    "physics-iq": BenchmarkIntegrationSpec(
        "physics-iq",
        IntegrationTier.IN_TREE,
        _runner_path("physics-iq"),
    ),
    "physics-iq-verified": BenchmarkIntegrationSpec(
        "physics-iq-verified",
        IntegrationTier.IN_TREE,
        _runner_path("physics-iq"),
    ),
    "physical-ai-bench": BenchmarkIntegrationSpec(
        "physical-ai-bench",
        IntegrationTier.IN_TREE,
        _runner_path("physical-ai-bench"),
        hf_dataset_id="shi-labs/physical-ai-bench-generation",
        judge_model_id="Qwen/Qwen2.5-VL-72B-Instruct+GroundingDINO+SAM2+Video-Depth-Anything",
    ),
    "physvidbench": BenchmarkIntegrationSpec(
        "physvidbench",
        IntegrationTier.MODEL_BACKED,
        _runner_path("physvidbench"),
        judge_model_id="models/gemini-2.0-flash",
    ),
    "t2v-compbench": BenchmarkIntegrationSpec(
        "t2v-compbench", IntegrationTier.IN_TREE, _runner_path("t2v-compbench"),
    ),
    "t2v-safety-bench": BenchmarkIntegrationSpec(
        "t2v-safety-bench",
        IntegrationTier.MODEL_BACKED,
        _runner_path("t2v-safety-bench"),
    ),
    "t2vworldbench": BenchmarkIntegrationSpec(
        "t2vworldbench",
        IntegrationTier.IN_TREE,
        _runner_path("t2vworldbench"),
    ),
    "vbench": BenchmarkIntegrationSpec(
        "vbench",
        IntegrationTier.IN_TREE,
        _runner_path("vbench"),
    ),
    "vbench-2.0": BenchmarkIntegrationSpec(
        "vbench-2.0",
        IntegrationTier.IN_TREE,
        _runner_path("vbench-2.0"),
    ),
    "vbench-plus-plus": BenchmarkIntegrationSpec(
        "vbench-plus-plus",
        IntegrationTier.IN_TREE,
        _runner_path("vbench-plus-plus"),
    ),
    "video-bench": BenchmarkIntegrationSpec(
        "video-bench", IntegrationTier.IN_TREE, _runner_path("video-bench"),
    ),
    "videophy": BenchmarkIntegrationSpec(
        "videophy",
        IntegrationTier.MODEL_BACKED,
        _runner_path("videophy"),
        judge_model_id="videophysics/videocon_physics",
    ),
    "videophy2": BenchmarkIntegrationSpec(
        "videophy2",
        IntegrationTier.MODEL_BACKED,
        _runner_path("videophy2"),
        hf_dataset_id="videophysics/videophy2_test",
        judge_model_id="videophysics/videophy_2_auto",
    ),
    "videoscience-bench": BenchmarkIntegrationSpec(
        "videoscience-bench",
        IntegrationTier.MODEL_BACKED,
        _runner_path("videoscience-bench"),
    ),
    "videoscore": BenchmarkIntegrationSpec(
        "videoscore",
        IntegrationTier.MODEL_BACKED,
        _runner_path("videoscore"),
        hf_dataset_id="TIGER-Lab/VideoScore-Bench",
        judge_model_id="TIGER-Lab/VideoScore-v1.1",
    ),
    "videoverse": BenchmarkIntegrationSpec(
        "videoverse", IntegrationTier.IN_TREE, _runner_path("videoverse"),
    ),
    "vmbench": BenchmarkIntegrationSpec(
        "vmbench", IntegrationTier.IN_TREE, _runner_path("vmbench"),
    ),
    "wbench": BenchmarkIntegrationSpec(
        "wbench",
        IntegrationTier.IN_TREE,
        _runner_path("wbench"),
        hf_dataset_id="meituan-longcat/WBench",
        judge_model_id="meituan-longcat/WBench-weights",
    ),
    "world-in-world": BenchmarkIntegrationSpec(
        "world-in-world",
        IntegrationTier.IN_TREE,
        _runner_path("world-in-world"),
    ),
    "worldarena": BenchmarkIntegrationSpec(
        "worldarena", IntegrationTier.IN_TREE, _runner_path("worldarena"),
    ),
    "worldbench": BenchmarkIntegrationSpec(
        "worldbench",
        IntegrationTier.IN_TREE,
        _runner_path("worldbench"),
        hf_dataset_id="worldbenchmark/IntuitivePhysics",
        judge_model_id="facebook/sam2.1-hiera-large",
    ),
    "worldmodelbench": BenchmarkIntegrationSpec(
        "worldmodelbench", IntegrationTier.MODEL_BACKED, _runner_path("worldmodelbench"),
    ),
    "worldscore": BenchmarkIntegrationSpec(
        "worldscore",
        IntegrationTier.IN_TREE,
        _runner_path("worldscore"),
        hf_dataset_id="Howieeeee/WorldScore",
    ),
    "wrbench": BenchmarkIntegrationSpec(
        "wrbench",
        IntegrationTier.IN_TREE,
        _runner_path("wrbench"),
        hf_dataset_id="WRBench/wrbench-natural25",
        judge_model_id="VGGT-Omega+DINOv2+Qwen3.5+Qwen3-VL",
    ),
}
