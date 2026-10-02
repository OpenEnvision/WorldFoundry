"""In-tree wiring smoke tests for the video benchmark runner suite.

These tests keep the video-bench integrations honest without any GPU or
network access:

* every official-runner entry module imports and exposes a ``main`` callable;
* every catalog entry stays consistent with the code it points at (task yaml,
  runtime profile, in-tree runtime roots, bundled assets);
* the vendored CameraBench official runtime is present and reachable from the
  runner without executing it.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNERS_ROOT = REPO_ROOT / "worldfoundry" / "evaluation" / "tasks" / "execution" / "runners"
CATALOG_ROOT = REPO_ROOT / "worldfoundry" / "data" / "benchmarks" / "catalog" / "video"
TASKS_ROOT = REPO_ROOT / "worldfoundry" / "data" / "benchmarks" / "tasks" / "external"
PROFILES_ROOT = REPO_ROOT / "worldfoundry" / "data" / "benchmarks" / "runtime_profiles" / "official"
RUNNER_BASE = "worldfoundry.evaluation.tasks.execution.runners"

# runner package -> (entry module, catalog file stem)
VIDEO_BENCH_RUNNERS: dict[str, tuple[str, str]] = {
    "aigcbench": ("run_aigcbench_official_runner", "aigcbench"),
    "camerabench": ("run_camerabench_official_runner", "camerabench"),
    "chronomagic_bench": ("run_chronomagic_bench_official_runner", "chronomagic-bench"),
    "evalcrafter": ("run_evalcrafter_official_runner", "evalcrafter"),
    "fetv": ("run_fetv_official_runner", "fetv"),
    "genai_bench": ("run_genai_bench_official_runner", "genai-bench"),
    "ipv_bench": ("run_ipv_bench_official_runner", "ipv-bench"),
    "memobench": ("run_memobench_official_runner", "memobench"),
    "mirabench": ("run_mirabench_official_runner", "mirabench"),
    "phyfps_bench_gen": ("run_phyfps_bench_gen_official_runner", "phyfps-bench-gen"),
    "sana_wm_bench": ("run_sana_wm_bench_official_runner", "sana-wm-bench"),
    "t2v_compbench": ("run_t2v_compbench_official_runner", "t2v-compbench"),
    "t2v_safety_bench": ("run_t2v_safety_bench_official_runner", "t2v-safety-bench"),
    "t2vworldbench": ("run_t2vworldbench_official_runner", "t2vworldbench"),
    "vbench": ("run_vbench_official_runner", "vbench"),
    "vbench_2_0": ("run_vbench_2_0_official_runner", "vbench-2.0"),
    "vbench_plus_plus": ("run_vbench_plus_plus_official_runner", "vbench-plus-plus"),
    "videobench": ("run_videobench_official_runner", "video-bench"),
    "videophy": ("run_videophy_official_runner", "videophy"),
    "videophy2": ("run_videophy2_official_runner", "videophy2"),
    "videoscience_bench": ("run_videoscience_bench_official_runner", "videoscience-bench"),
    "videoscore": ("run_videoscore_official_runner", "videoscore"),
    "videoverse": ("run_videoverse_official_runner", "videoverse"),
    "vmbench": ("run_vmbench_official_runner", "vmbench"),
    "wbench": ("run_wbench_official_runner", "wbench"),
}


def _load_catalog(stem: str) -> dict:
    path = CATALOG_ROOT / f"{stem}.yaml"
    assert path.is_file(), f"missing catalog entry: {path}"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), f"catalog entry is not a mapping: {path}"
    return data


@pytest.mark.parametrize("runner_pkg", sorted(VIDEO_BENCH_RUNNERS))
def test_official_runner_entry_module_imports(runner_pkg: str) -> None:
    entry_module, _ = VIDEO_BENCH_RUNNERS[runner_pkg]
    script = RUNNERS_ROOT / runner_pkg / f"{entry_module}.py"
    assert script.is_file(), f"missing official runner script: {script}"
    module = importlib.import_module(f"{RUNNER_BASE}.{runner_pkg}.{entry_module}")
    assert callable(getattr(module, "main", None)), f"{runner_pkg}.{entry_module} has no callable main()"


@pytest.mark.parametrize("runner_pkg", sorted(VIDEO_BENCH_RUNNERS))
def test_catalog_wiring_is_consistent_with_tree(runner_pkg: str) -> None:
    _, stem = VIDEO_BENCH_RUNNERS[runner_pkg]
    catalog = _load_catalog(stem)
    assert catalog.get("id") == stem, f"catalog id mismatch for {stem}: {catalog.get('id')}"

    task_yaml = TASKS_ROOT / f"{stem}.yaml"
    assert task_yaml.is_file(), f"missing task yaml: {task_yaml}"
    profile_yaml = PROFILES_ROOT / f"{stem}.yaml"
    assert profile_yaml.is_file(), f"missing runtime profile: {profile_yaml}"

    data_refs = catalog.get("data_refs") or {}
    declared_task_yaml = data_refs.get("task_yaml")
    if declared_task_yaml:
        assert (REPO_ROOT / declared_task_yaml).is_file(), f"declared task yaml missing: {declared_task_yaml}"

    bundled_assets = data_refs.get("bundled_assets")
    if bundled_assets:
        assert (REPO_ROOT / bundled_assets).is_dir(), f"declared bundled assets missing: {bundled_assets}"

    for runtime_path in data_refs.get("in_tree_runtime_paths") or []:
        assert (REPO_ROOT / runtime_path).exists(), f"declared in-tree runtime path missing: {runtime_path}"

    runner_section = catalog.get("runner") or {}
    runtime_section = runner_section.get("runtime") or {}
    runtime_root = runtime_section.get("root")
    if runtime_root:
        assert (REPO_ROOT / runtime_root).exists(), f"declared runner.runtime.root missing: {runtime_root}"


def test_camerabench_vendored_runtime_is_wired() -> None:
    runner = importlib.import_module(f"{RUNNER_BASE}.camerabench.run_camerabench_official_runner")
    runtime_root = runner.DEFAULT_CAMERABENCH_RUNTIME_ROOT
    assert runtime_root.is_dir(), f"missing vendored CameraBench runtime: {runtime_root}"
    for task, script in runner.VENDORED_TASK_SCRIPTS.items():
        assert (runtime_root / script).is_file(), f"missing vendored CameraBench {task} script: {script}"
    assert callable(runner.run_vendored_official_scripts)


def test_camerabench_run_official_requires_vendored_scripts(tmp_path: Path) -> None:
    runner = importlib.import_module(f"{RUNNER_BASE}.camerabench.run_camerabench_official_runner")
    with pytest.raises(FileNotFoundError):
        runner.run_vendored_official_scripts(
            score_dir=tmp_path,
            output_dir=tmp_path / "out",
            runtime_root=tmp_path / "not-a-runtime",
            task="binary",
        )
