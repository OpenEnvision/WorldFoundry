"""LikePhys likelihood-probe runtime adapter.

LikePhys does not score generated videos: it runs the model under test as a denoiser over
a fixed set of paired valid/impossible clips. That probe needs the upstream evaluator,
which forks ``diffusers`` pipelines and schedulers to expose ``customize_add_noise`` and
per-timestep sigmas. WorldFoundry therefore drives a caller-supplied official checkout
rather than vendoring those forks.

The checkout resolves from ``--likephys-root``, ``WORLDFOUNDRY_LIKEPHYS_EVALUATOR_ROOT``, or
``thirdparty/LikePhys``. Because ``evaluator.py`` reads ``./data/<scenario>_videos`` and
writes ``./results/...`` relative to its working directory, each run gets a scratch
workspace of symlinks (upstream sources plus ``data`` pointing at the dataset root), so a
dataset stored outside the checkout never requires mutating it.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.core.execution.process import run_logged_subprocess
from worldfoundry.evaluation.tasks.execution.runners.likephys.likephys_scenarios import (
    CANONICAL_SUBGROUP_COUNT,
    DEFAULT_EXPERIMENT_NAME,
    DEFAULT_SEED,
    DEFAULT_TIMESTEP_NUM,
    OFFICIAL_SCENARIO_SWEEP,
    PROBE_MODELS_BY_KEY,
    SCENARIO_ORDER,
    VALID_VARIATION,
    official_experiment_dirname,
    scenario_for_id,
)

VIDEO_SUFFIX = ".mp4"
EVALUATOR_SCRIPT = "evaluator.py"
UPSTREAM_WORKSPACE_ENTRIES = ("evaluator.py", "pipeline", "scheduler", "utils")
DEFAULT_CHECKOUT_RELPATH = Path("thirdparty") / "LikePhys"
OFFICIAL_REPO_URL = "https://github.com/YuanJianhao508/LikePhys"
OFFICIAL_DATASET_REPO = "JianhaoDYDY/LikePhys-Benchmark"


class LikePhysRuntimeError(RuntimeError):
    """Raised when the LikePhys probe cannot be prepared or executed."""


@dataclass(frozen=True)
class LikePhysProbeConfig:
    """Resolved configuration for one LikePhys probe sweep."""

    probe_model: str
    checkout_root: Path
    dataset_root: Path
    scenarios: tuple[str, ...]
    seed: int = DEFAULT_SEED
    guidance_scale: bool = True
    tag_name: str | None = None
    experiment_name: str = DEFAULT_EXPERIMENT_NAME
    timestep_num: int = DEFAULT_TIMESTEP_NUM
    python_executable: str = "python"
    timeout_seconds: float | None = None
    extra_args: tuple[str, ...] = field(default_factory=tuple)

    @property
    def experiment_dirname(self) -> str:
        return official_experiment_dirname(
            seed=self.seed,
            guidance_scale=self.guidance_scale,
            tag_name=self.tag_name,
            experiment_name=self.experiment_name,
        )


def _env_path(*names: str) -> Path | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return Path(value).expanduser()
    return None


def resolve_checkout_root(explicit: Path | None = None, *, repo_root: Path | None = None) -> Path | None:
    """Resolve the official LikePhys checkout used for the probe stage.

    ``WORLDFOUNDRY_LIKEPHYS_ROOT`` follows the catalog convention of pointing at the
    in-tree runner package, so the upstream checkout is read from
    ``WORLDFOUNDRY_LIKEPHYS_EVALUATOR_ROOT`` first. Either variable is accepted, and a
    candidate only counts when it actually contains ``evaluator.py``.
    """
    candidates = [
        explicit,
        _env_path("WORLDFOUNDRY_LIKEPHYS_EVALUATOR_ROOT"),
        _env_path("WORLDFOUNDRY_LIKEPHYS_ROOT"),
    ]
    if repo_root is not None:
        candidates.append(repo_root / DEFAULT_CHECKOUT_RELPATH)
    for candidate in candidates:
        if candidate is None:
            continue
        resolved = Path(candidate).expanduser()
        if (resolved / EVALUATOR_SCRIPT).is_file():
            return resolved.resolve()
    return None


def resolve_dataset_root(explicit: Path | None = None, *, checkout_root: Path | None = None) -> Path | None:
    """Resolve the directory holding ``<scenario>_videos`` clip folders."""
    candidates = [
        explicit,
        _env_path(
            "WORLDFOUNDRY_LIKEPHYS_DATA_ROOT",
            # Set when the likephys_dataset_assets capability materializes the HF dataset.
            "WORLDFOUNDRY_LIKEPHYS_DATASET_ROOT",
            "WORLDFOUNDRY_BENCHMARK_DATA_ROOT",
        ),
    ]
    if checkout_root is not None:
        candidates.append(checkout_root / "data")
    for candidate in candidates:
        if candidate is None:
            continue
        resolved = Path(candidate).expanduser()
        if not resolved.is_dir():
            continue
        for probe in (resolved, resolved / "data"):
            if any((probe / scenario_for_id(scenario_id).dataset_dir).is_dir() for scenario_id in SCENARIO_ORDER):
                return probe.resolve()
    return None


def inspect_dataset(dataset_root: Path, scenarios: Sequence[str] | None = None) -> dict[str, Any]:
    """Report per-scenario subgroup/clip coverage for a local LikePhys dataset root."""
    selected = tuple(scenarios or SCENARIO_ORDER)
    per_scenario: dict[str, Any] = {}
    for scenario_id in selected:
        scenario = scenario_for_id(scenario_id)
        scenario_dir = dataset_root / scenario.dataset_dir
        if not scenario_dir.is_dir():
            per_scenario[scenario_id] = {"available": False, "reason": "scenario_directory_missing"}
            continue
        subgroups = sorted(path for path in scenario_dir.iterdir() if path.is_dir())
        variations: set[str] = set()
        clip_count = 0
        subgroups_with_valid = 0
        for subgroup in subgroups:
            clips = [path for path in subgroup.iterdir() if path.is_file() and path.suffix.lower() == VIDEO_SUFFIX]
            clip_count += len(clips)
            names = {path.name.rsplit("_", 1)[0] for path in clips}
            variations |= names
            if VALID_VARIATION in names:
                subgroups_with_valid += 1
        per_scenario[scenario_id] = {
            "available": bool(subgroups) and subgroups_with_valid > 0,
            "subgroup_count": len(subgroups),
            "subgroups_with_valid_clip": subgroups_with_valid,
            "clip_count": clip_count,
            "variations": sorted(variations),
            "missing_variations": sorted(set(scenario.variations) - variations),
            "expected_subgroup_count": CANONICAL_SUBGROUP_COUNT,
        }
    available = [scenario_id for scenario_id, entry in per_scenario.items() if entry.get("available")]
    return {
        "dataset_root": str(dataset_root.resolve()),
        "scenario_count": len(selected),
        "available_scenario_count": len(available),
        "available_scenarios": available,
        "complete": len(available) == len(selected),
        "per_scenario": per_scenario,
    }


def prepare_workspace(*, checkout_root: Path, dataset_root: Path, workspace: Path) -> Path:
    """Create a scratch workspace linking upstream sources and the dataset root.

    Returns:
        The workspace directory to use as the evaluator working directory.
    """
    workspace.mkdir(parents=True, exist_ok=True)
    for name in UPSTREAM_WORKSPACE_ENTRIES:
        source = checkout_root / name
        if not source.exists():
            if name == EVALUATOR_SCRIPT:
                raise LikePhysRuntimeError(f"LikePhys checkout is missing {name}: {checkout_root}")
            continue
        link = workspace / name
        if link.is_symlink() or link.exists():
            if link.is_symlink() or link.is_file():
                link.unlink()
            else:
                shutil.rmtree(link)
        link.symlink_to(source, target_is_directory=source.is_dir())
    data_link = workspace / "data"
    if data_link.is_symlink() or data_link.exists():
        if data_link.is_symlink() or data_link.is_file():
            data_link.unlink()
        else:
            shutil.rmtree(data_link)
    data_link.symlink_to(dataset_root, target_is_directory=True)
    return workspace


def probe_command(config: LikePhysProbeConfig, scenario_id: str) -> list[str]:
    """Build the upstream ``evaluator.py`` command line for one scenario."""
    command = [
        config.python_executable,
        EVALUATOR_SCRIPT,
        "--model",
        config.probe_model,
        "--data",
        scenario_id,
        "--seed",
        str(config.seed),
        "--timestep_num",
        str(config.timestep_num),
        "--exp_name",
        config.experiment_name,
    ]
    if config.guidance_scale:
        command.append("--guidance_scale")
    if config.tag_name:
        command.extend(["--tag_name", config.tag_name])
    command.extend(config.extra_args)
    return command


def probe_config_from_env(
    *,
    probe_model: str,
    checkout_root: Path | None,
    dataset_root: Path,
    scenarios: Sequence[str] | None = None,
    seed: int = DEFAULT_SEED,
    guidance_scale: bool = True,
    tag_name: str | None = None,
    timestep_num: int = DEFAULT_TIMESTEP_NUM,
    timeout_seconds: float | None = None,
) -> LikePhysProbeConfig:
    """Assemble a probe configuration, filling the interpreter from the environment."""
    if checkout_root is None:
        raise LikePhysRuntimeError(
            "LikePhys probe execution requires the official evaluator checkout. Clone "
            f"{OFFICIAL_REPO_URL} and set WORLDFOUNDRY_LIKEPHYS_EVALUATOR_ROOT or pass --likephys-root."
        )
    if probe_model not in PROBE_MODELS_BY_KEY:
        known = ", ".join(sorted(PROBE_MODELS_BY_KEY))
        raise LikePhysRuntimeError(f"unknown LikePhys probe backend {probe_model!r}; known: {known}")
    python_executable = os.environ.get("WORLDFOUNDRY_UNIFIED_PYTHON") or os.environ.get("PYTHON") or "python"
    selected = tuple(scenarios or OFFICIAL_SCENARIO_SWEEP)
    for scenario_id in selected:
        scenario_for_id(scenario_id)
    return LikePhysProbeConfig(
        probe_model=probe_model,
        checkout_root=checkout_root,
        dataset_root=dataset_root,
        scenarios=selected,
        seed=seed,
        guidance_scale=guidance_scale,
        tag_name=tag_name,
        timestep_num=timestep_num,
        python_executable=python_executable,
        timeout_seconds=timeout_seconds,
    )


def run_likephys_probe(
    *,
    config: LikePhysProbeConfig,
    output_dir: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run the LikePhys ELBO probe for every configured scenario.

    Args:
        config: Resolved probe configuration.
        output_dir: WorldFoundry run directory; logs and collected results land here.
        dry_run: Prepare the workspace and report the commands without executing them.

    Returns:
        A summary with the resolved results root and per-scenario execution status.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    workspace = prepare_workspace(
        checkout_root=config.checkout_root,
        dataset_root=config.dataset_root,
        workspace=output_dir / "probe_workspace",
    )
    log_dir = output_dir / "probe_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    results_root = workspace / "results" / config.experiment_dirname

    scenario_reports: list[dict[str, Any]] = []
    for scenario_id in config.scenarios:
        command = probe_command(config, scenario_id)
        expected_results = results_root / scenario_id / f"results_{config.probe_model}.json"
        report: dict[str, Any] = {
            "scenario_id": scenario_id,
            "command": list(command),
            "results_path": str(expected_results),
            "executed": not dry_run,
        }
        if dry_run:
            report["status"] = "skipped_dry_run"
            scenario_reports.append(report)
            continue
        env = os.environ.copy()
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        completed = run_logged_subprocess(
            command,
            stdout_path=log_dir / f"{scenario_id}_stdout.log",
            stderr_path=log_dir / f"{scenario_id}_stderr.log",
            cwd=workspace,
            env=env,
            timeout=config.timeout_seconds,
        )
        report["returncode"] = completed.returncode
        report["status"] = "succeeded" if completed.returncode == 0 and expected_results.is_file() else "failed"
        report["results_available"] = expected_results.is_file()
        scenario_reports.append(report)

    succeeded = [report for report in scenario_reports if report.get("status") == "succeeded"]
    return {
        "probe_model": config.probe_model,
        "checkout_root": str(config.checkout_root),
        "dataset_root": str(config.dataset_root),
        "workspace": str(workspace),
        "results_root": str(results_root),
        "experiment_dirname": config.experiment_dirname,
        "seed": config.seed,
        "guidance_scale": config.guidance_scale,
        "timestep_num": config.timestep_num,
        "scenario_count": len(config.scenarios),
        "succeeded_scenario_count": len(succeeded),
        "complete": len(succeeded) == len(config.scenarios) and not dry_run,
        "dry_run": dry_run,
        "scenarios": scenario_reports,
        "official_repo_url": OFFICIAL_REPO_URL,
        "official_dataset_repo": OFFICIAL_DATASET_REPO,
    }


def runtime_summary(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a JSON-safe copy of a probe summary for scorecard embedding."""
    return dict(config or {})


__all__ = [
    "DEFAULT_CHECKOUT_RELPATH",
    "EVALUATOR_SCRIPT",
    "LikePhysProbeConfig",
    "LikePhysRuntimeError",
    "OFFICIAL_DATASET_REPO",
    "OFFICIAL_REPO_URL",
    "inspect_dataset",
    "prepare_workspace",
    "probe_command",
    "probe_config_from_env",
    "resolve_checkout_root",
    "resolve_dataset_root",
    "run_likephys_probe",
    "runtime_summary",
]
