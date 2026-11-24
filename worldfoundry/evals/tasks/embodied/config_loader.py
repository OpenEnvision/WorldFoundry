"""Configuration loading for embodied evaluation runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from worldfoundry.evals.tasks.embodied.simulators.registry import get_simulator_entry
from worldfoundry.evals.utils import BENCHMARK_RUNTIME_PROFILE_DIR


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in override.items():
        if key in {"extends", "_base"}:
            continue
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_embodied_config(path: str | Path) -> dict[str, Any]:
    """Load an embodied YAML config with optional ``extends``/``_base`` support."""
    resolved = Path(path).resolve()
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"embodied config must be a mapping: {resolved}")
    base_ref = payload.get("extends") or payload.get("_base")
    if base_ref:
        base_path = Path(base_ref)
        if not base_path.is_absolute():
            base_path = resolved.parent / base_path
        base = load_embodied_config(base_path)
        payload = _deep_merge(base, payload)
    payload.setdefault("_config_path", str(resolved))
    return payload


def _profile_id_for_benchmark(benchmark_id: str) -> str:
    entry = get_simulator_entry(benchmark_id)
    return entry.benchmark_id if entry is not None else benchmark_id.strip().lower().replace("_", "-")


def _load_runtime_profile(benchmark_id: str) -> dict[str, Any]:
    profile_id = _profile_id_for_benchmark(benchmark_id)
    path = BENCHMARK_RUNTIME_PROFILE_DIR / "official" / f"{profile_id}.yaml"
    if not path.is_file():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return payload if isinstance(payload, dict) else {}


def _docker_defaults_for_benchmarks(benchmarks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for bench in benchmarks:
        benchmark_id = str(bench.get("benchmark_id") or bench.get("id") or "")
        profile_docker = dict(_load_runtime_profile(benchmark_id).get("docker") or {})
        if not profile_docker:
            continue
        if not defaults:
            defaults = profile_docker
            continue
        if profile_docker.get("image") != defaults.get("image"):
            return {}
        defaults = _deep_merge(defaults, profile_docker)
    return defaults


def _first_benchmark_id(config: Mapping[str, Any], fallback: str = "libero") -> str:
    benchmark_ids = config.get("benchmark_ids") or ()
    if isinstance(benchmark_ids, str):
        benchmark_ids = (benchmark_ids,)
    for benchmark_id in benchmark_ids:
        if benchmark_id:
            return str(benchmark_id)
    return str(config.get("benchmark") or fallback)


def canonicalize_embodied_config(
    payload: Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
    server_url: str | None = None,
) -> dict[str, Any]:
    """Normalize flat and multi-benchmark embodied configs to one native WF shape."""
    config = deepcopy(dict(payload))
    if output_dir is not None:
        config["output_dir"] = str(output_dir)
    config.setdefault("id", config.get("name") or "embodied_eval")
    config.setdefault("output_dir", "./results/embodied_eval")
    if server_url:
        config.setdefault("server", {})["url"] = server_url

    if "benchmarks" not in config:
        benchmark_id = str(config.get("benchmark_id") or _first_benchmark_id(config))
        benchmark_kwargs = dict(config.get("benchmark_kwargs") or config.get("params") or {})
        benchmark = {
            "id": config.get("benchmark_config_id") or config.get("id") or benchmark_id,
            "benchmark_id": benchmark_id,
            "params": benchmark_kwargs,
            "episodes_per_task": int(config.get("episodes_per_task", 1)),
            "max_tasks": config.get("max_tasks"),
            "max_steps": config.get("max_steps"),
            "tasks": config.get("tasks"),
            "mode": config.get("mode", "sync"),
            "recording": config.get("recording"),
        }
        config["benchmarks"] = [benchmark]

    model_parameters = dict(config.get("model_parameters") or {})
    model = dict(config.get("model") or {})
    model.setdefault("id", config.get("model_id", "openvla"))
    if config.get("model_variant_id") and "variant_id" not in model:
        model["variant_id"] = config["model_variant_id"]
    model["parameters"] = _deep_merge(model.get("parameters") or {}, model_parameters)
    config["model"] = model

    normalized_benchmarks = []
    for item in config.get("benchmarks") or []:
        if not isinstance(item, Mapping):
            raise ValueError("each embodied benchmark config must be a mapping")
        entry = dict(item)
        entry.setdefault(
            "benchmark_id",
            entry.get("id") or entry.get("benchmark") or config.get("benchmark_id") or _first_benchmark_id(config),
        )
        entry.setdefault("id", entry.get("benchmark_id"))
        entry.setdefault("params", entry.get("benchmark_kwargs") or {})
        entry.setdefault("episodes_per_task", config.get("episodes_per_task", 1))
        if entry.get("max_tasks") is None and config.get("max_tasks") is not None:
            entry["max_tasks"] = config["max_tasks"]
        normalized_benchmarks.append(entry)
    config["benchmarks"] = normalized_benchmarks

    docker_defaults = _docker_defaults_for_benchmarks(normalized_benchmarks)
    if docker_defaults:
        config["docker"] = _deep_merge(docker_defaults, dict(config.get("docker") or {}))
    return config


def load_canonical_embodied_config(
    path: str | Path,
    *,
    output_dir: str | Path | None = None,
    server_url: str | None = None,
) -> dict[str, Any]:
    """Load and canonicalize an embodied evaluation config."""
    return canonicalize_embodied_config(load_embodied_config(path), output_dir=output_dir, server_url=server_url)


__all__ = ["canonicalize_embodied_config", "load_canonical_embodied_config", "load_embodied_config"]
