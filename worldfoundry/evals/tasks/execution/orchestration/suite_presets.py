"""Named model × benchmark suite presets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from worldfoundry.evals.utils import (
    load_manifest,
)

from .suite_requests import ModelBenchmarkSuiteRequest


def _lookup_key(value: str) -> str:
    """Normalizes string aliases (e.g. replacing underscores with hyphens)."""
    return str(value).strip().lower().replace("_", "-")


def _load_suite_presets(path: str | Path | None) -> dict[str, dict[str, Any]]:
    """Loads optional declarative model-benchmark suite combinations from an external YAML manifest."""
    if path is None:
        return {}
    source = Path(path)
    if not source.is_file():
        return {}
    payload = load_manifest(source)
    if not isinstance(payload, Mapping):
        raise TypeError(f"suite preset file must be a mapping: {source}")
    raw_suites = payload.get("suites", [])
    if not isinstance(raw_suites, list):
        raise TypeError(f"suite preset file suites must be a list: {source}")
    suites: dict[str, dict[str, Any]] = {}
    for item in raw_suites:
        if not isinstance(item, Mapping) or not item.get("id"):
            continue
        suite = dict(item)
        suite_id = str(suite["id"])
        keys = [suite_id, *[str(alias) for alias in suite.get("aliases") or ()]]
        for key in keys:
            suites[_lookup_key(key)] = suite
    return suites


def list_model_benchmark_suite_presets(path: str | Path | None = None) -> tuple[Mapping[str, Any], ...]:
    """List unique suite presets from the suites YAML file."""
    suites = _load_suite_presets(path)
    unique: dict[str, Mapping[str, Any]] = {}
    for suite in suites.values():
        suite_id = str(suite.get("id"))
        unique.setdefault(suite_id, suite)
    return tuple(unique[key] for key in sorted(unique))


def get_model_benchmark_suite_preset(suite_id: str, path: str | Path | None = None) -> Mapping[str, Any]:
    """Look up one suite preset by id or alias."""
    suites = _load_suite_presets(path)
    try:
        return suites[_lookup_key(suite_id)]
    except KeyError as exc:
        known = ", ".join(str(item.get("id")) for item in list_model_benchmark_suite_presets(path))
        raise KeyError(f"unknown model-benchmark suite preset {suite_id!r}; known: {known}") from exc


def _selected_suite_presets(request: ModelBenchmarkSuiteRequest) -> tuple[Mapping[str, Any], ...]:
    """Retrieves a compiled list of all presets selected inside the suite request."""
    if not request.suite_ids:
        return ()
    selected = []
    for suite_id in request.suite_ids:
        selected.append(get_model_benchmark_suite_preset(str(suite_id), request.suite_preset_path))
    return tuple(selected)


def _preset_values(presets: Sequence[Mapping[str, Any]], key: str) -> tuple[str, ...]:
    """Extracts unique string values associated with a specific preset key (e.g. 'model_ids')."""
    values: list[str] = []
    for preset in presets:
        for item in preset.get(key) or ():
            text = str(item)
            if text not in values:
                values.append(text)
    return tuple(values)
