"""Video benchmark official-runner registry.

Maps ``benchmark_id`` to in-tree runner CLI scripts and result-path flags.
Consumed by :mod:`integration` and :class:`ManifestBenchmarkRunner` specialized
normalizer dispatch.

Sections:

* **VideoRunnerSpec** — per-benchmark script path and CLI flag metadata.
* **VIDEO_RUNNER_REGISTRY** — canonical lookup table for video benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VideoRunnerSpec:
    """In-tree official runner script and results-path CLI flag."""

    script: str
    results_flag: str
    extra_args: tuple[str, ...] = ()
    pass_benchmark_id: bool = True


# ---------------------------------------------------------------------------
# Video benchmark runner registry
# ---------------------------------------------------------------------------

VIDEO_RUNNER_REGISTRY: dict[str, VideoRunnerSpec] = {
    "aigcbench": VideoRunnerSpec(
        "worldfoundry/evaluation/tasks/execution/runners/aigcbench/run_aigcbench_official_runner.py",
        "--official-results-path",
    ),
    "camerabench": VideoRunnerSpec(
        "worldfoundry/evaluation/tasks/execution/runners/camerabench/run_camerabench_official_runner.py",
        "--official-results-path",
    ),
    "chronomagic-bench": VideoRunnerSpec(
        "worldfoundry/evaluation/tasks/execution/runners/chronomagic_bench/run_chronomagic_bench_official_runner.py",
        "--official-results-path",
    ),
    "devil-dynamics": VideoRunnerSpec(
        "worldfoundry/evaluation/tasks/execution/runners/devil_dynamics/run_devil_dynamics_official_runner.py",
        "--official-results-path",
    ),
    "evalcrafter": VideoRunnerSpec(
        "worldfoundry/evaluation/tasks/execution/runners/evalcrafter/run_evalcrafter_official_runner.py",
        "--official-results-path",
        pass_benchmark_id=False,
    ),
    "ewmbench": VideoRunnerSpec(
        "worldfoundry/evaluation/tasks/execution/runners/ewmbench/run_ewmbench_official_runner.py",
        "--official-results-path",
    ),
    "fetv": VideoRunnerSpec(
        "worldfoundry/evaluation/tasks/execution/runners/fetv/run_fetv_official_runner.py",
        "--official-results-path",
    ),
    "4dworldbench": VideoRunnerSpec(
        "worldfoundry/evaluation/tasks/execution/runners/four_d_worldbench/run_four_d_worldbench_official_runner.py",
        "--official-results-path",
    ),
    "genai-bench": VideoRunnerSpec(
        "worldfoundry/evaluation/tasks/execution/runners/genai_bench/run_genai_bench_official_runner.py",
        "--official-results-path",
    ),
    "ipv-bench": VideoRunnerSpec(
        "worldfoundry/evaluation/tasks/execution/runners/ipv_bench/run_ipv_bench_official_runner.py",
        "--official-results-path",
    ),
    "iworld-bench": VideoRunnerSpec(
        "worldfoundry/evaluation/tasks/execution/runners/iworldbench/run_iworldbench_official_runner.py",
        "--official-results-path",
        pass_benchmark_id=False,
    ),
    "larybench": VideoRunnerSpec(
        "worldfoundry/evaluation/tasks/execution/runners/larybench/run_larybench_official_runner.py",
        "--official-results-path",
    ),
    "mirabench": VideoRunnerSpec(
        "worldfoundry/evaluation/tasks/execution/runners/mirabench/run_mirabench_official_runner.py",
        "--official-results-path",
    ),
    "memobench": VideoRunnerSpec(
        "worldfoundry/evaluation/tasks/execution/runners/memobench/run_memobench_official_runner.py",
        "--official-results-path",
    ),
    "phyeduvideo": VideoRunnerSpec(
        "worldfoundry/evaluation/tasks/execution/runners/phyeduvideo/run_phyeduvideo_official_runner.py",
        "--official-results-path",
    ),
    "phyfps-bench-gen": VideoRunnerSpec(
        "worldfoundry/evaluation/tasks/execution/runners/phyfps_bench_gen/run_phyfps_bench_gen_official_runner.py",
        "--official-results-path",
    ),
    "visual-chronometer": VideoRunnerSpec(
        "worldfoundry/evaluation/tasks/execution/runners/phyfps_bench_gen/run_visual_chronometer_official_runner.py",
        "--official-results-path",
    ),
    "phygenbench": VideoRunnerSpec(
        "worldfoundry/evaluation/tasks/execution/runners/phygenbench/run_phygenbench_official_runner.py",
        "--official-results-path",
        pass_benchmark_id=False,
    ),
    "phyground": VideoRunnerSpec(
        "worldfoundry/evaluation/tasks/execution/runners/phyground/run_phyground_official_runner.py",
        "--official-results-path",
        pass_benchmark_id=False,
    ),
    "physics-iq": VideoRunnerSpec(
        "worldfoundry/evaluation/tasks/execution/runners/physics_iq/run_physics_iq_official_runner.py",
        "--official-results-path",
    ),
    "physics-iq-verified": VideoRunnerSpec(
        "worldfoundry/evaluation/tasks/execution/runners/physics_iq/run_physics_iq_official_runner.py",
        "--official-results-path",
    ),
    "physical-ai-bench": VideoRunnerSpec(
        "worldfoundry/evaluation/tasks/execution/runners/physical_ai_bench/run_physical_ai_bench_official_runner.py",
        "--official-results-path",
    ),
    "physvidbench": VideoRunnerSpec(
        "worldfoundry/evaluation/tasks/execution/runners/physvidbench/run_physvidbench_official_runner.py",
        "--official-results-path",
    ),
    "t2v-compbench": VideoRunnerSpec(
        "worldfoundry/evaluation/tasks/execution/runners/t2v_compbench/run_t2v_compbench_official_runner.py",
        "--official-results-path",
    ),
    "t2v-safety-bench": VideoRunnerSpec(
        "worldfoundry/evaluation/tasks/execution/runners/t2v_safety_bench/run_t2v_safety_bench_official_runner.py",
        "--official-results-path",
    ),
    "t2vworldbench": VideoRunnerSpec(
        "worldfoundry/evaluation/tasks/execution/runners/t2vworldbench/run_t2vworldbench_official_runner.py",
        "--official-results-path",
    ),
    "vbench": VideoRunnerSpec(
        "worldfoundry/evaluation/tasks/execution/runners/vbench/run_vbench_official_runner.py",
        "--official-results-path",
    ),
    "vbench-2.0": VideoRunnerSpec(
        "worldfoundry/evaluation/tasks/execution/runners/vbench_2_0/run_vbench_2_0_official_runner.py",
        "--official-results-path",
    ),
    "vbench-plus-plus": VideoRunnerSpec(
        "worldfoundry/evaluation/tasks/execution/runners/vbench_plus_plus/run_vbench_plus_plus_official_runner.py",
        "--official-results-path",
    ),
    "video-bench": VideoRunnerSpec(
        "worldfoundry/evaluation/tasks/execution/runners/videobench/run_videobench_official_runner.py",
        "--official-results-path",
    ),
    "videophy": VideoRunnerSpec(
        "worldfoundry/evaluation/tasks/execution/runners/videophy/run_videophy_official_runner.py",
        "--official-results-path",
        pass_benchmark_id=False,
    ),
    "videophy2": VideoRunnerSpec(
        "worldfoundry/evaluation/tasks/execution/runners/videophy2/run_videophy2_official_runner.py",
        "--official-results-path",
        pass_benchmark_id=False,
    ),
    "videoscience-bench": VideoRunnerSpec(
        "worldfoundry/evaluation/tasks/execution/runners/videoscience_bench/run_videoscience_bench_official_runner.py",
        "--official-results-path",
    ),
    "videoscore": VideoRunnerSpec(
        "worldfoundry/evaluation/tasks/execution/runners/videoscore/run_videoscore_official_runner.py",
        "--official-results-path",
    ),
    "videoverse": VideoRunnerSpec(
        "worldfoundry/evaluation/tasks/execution/runners/videoverse/run_videoverse_official_runner.py",
        "--official-results-path",
    ),
    "vmbench": VideoRunnerSpec(
        "worldfoundry/evaluation/tasks/execution/runners/vmbench/run_vmbench_official_runner.py",
        "--official-results-path",
    ),
    "wbench": VideoRunnerSpec(
        "worldfoundry/evaluation/tasks/execution/runners/wbench/run_wbench_official_runner.py",
        "--official-results-path",
    ),
    "world-in-world": VideoRunnerSpec(
        "worldfoundry/evaluation/tasks/execution/runners/world_in_world/run_world_in_world_official_runner.py",
        "--official-results-path",
    ),
    "worldarena": VideoRunnerSpec(
        "worldfoundry/evaluation/tasks/execution/runners/worldarena/run_worldarena_official_runner.py",
        "--official-results-path",
    ),
    "worldbench": VideoRunnerSpec(
        "worldfoundry/evaluation/tasks/execution/runners/worldbench/run_worldbench_official_runner.py",
        "--official-results-path",
    ),
    "worldmodelbench": VideoRunnerSpec(
        "worldfoundry/evaluation/tasks/execution/runners/worldmodelbench/run_worldmodelbench_official_runner.py",
        "--official-results-path",
    ),
    "worldreasonbench": VideoRunnerSpec(
        "worldfoundry/evaluation/tasks/execution/runners/worldreasonbench/run_worldreasonbench_official_runner.py",
        "--official-results-path",
    ),
    "worldscore": VideoRunnerSpec(
        "worldfoundry/evaluation/tasks/execution/runners/worldscore/run_worldscore_official_runner.py",
        "--official-results-path",
    ),
    "wrbench": VideoRunnerSpec(
        "worldfoundry/evaluation/tasks/execution/runners/wrbench/run_wrbench_official_runner.py",
        "--official-results-path",
    ),
}


def specialized_result_normalizer_scripts() -> dict[str, tuple[str, str]]:
    """Return ``benchmark_id → (script, results_flag)`` for all video runners."""
    return {benchmark_id: (spec.script, spec.results_flag) for benchmark_id, spec in VIDEO_RUNNER_REGISTRY.items()}
