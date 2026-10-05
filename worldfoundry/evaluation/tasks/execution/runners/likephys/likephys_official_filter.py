"""LikePhys official result filtering and contract metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from worldfoundry.evaluation.tasks.execution.framework.official_result_scoring import _loose_id
from worldfoundry.evaluation.tasks.execution.runners.likephys.likephys_metrics import parse_results_filename

BENCHMARK_ID = "likephys"

OFFICIAL_REQUIREMENTS: dict[str, Any] = {
    "reason": "white_box_model_probe_required",
    "required_inputs": [
        "JianhaoDYDY/LikePhys-Benchmark paired valid/invalid video dataset",
        "a video diffusion pipeline whose denoiser, VAE, and scheduler can be called directly",
        "official LikePhys evaluator checkout (WORLDFOUNDRY_LIKEPHYS_EVALUATOR_ROOT) for the probe stage",
        "results_<model>.json per scenario with scene_evaluations losses",
    ],
}

_MODEL_KEYS = ("probe_model", "model", "model_key", "model_id")


def _record_model_key(record: Mapping[str, Any]) -> str | None:
    for key in _MODEL_KEYS:
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    source_path = record.get("source_path")
    if isinstance(source_path, str) and source_path:
        return parse_results_filename(Path(source_path))[1]
    return None


def filter_official_records(
    records: list[Mapping[str, Any]],
    *,
    generated_artifact_dir: Path | None = None,
    result_model_id: str | None = None,
) -> list[Mapping[str, Any]]:
    """Restrict LikePhys records to one probe backend when several were imported.

    LikePhys scores a diffusion model rather than a video directory, so ``result_model_id``
    is the probe backend key (for example ``wan2.1-T2V-1.3b``). When it is absent the
    directory name of ``generated_artifact_dir`` is used as a fallback hint.
    """
    if not records:
        return records
    model_id = result_model_id or (generated_artifact_dir.name if generated_artifact_dir is not None else None)
    if not model_id:
        return records
    labelled = [record for record in records if _record_model_key(record) not in (None, "")]
    if not labelled:
        return records
    exact = [record for record in labelled if _record_model_key(record) == model_id]
    if exact:
        return exact
    normalized = _loose_id(model_id)
    loose = [record for record in labelled if _loose_id(str(_record_model_key(record))) == normalized]
    return loose or records


__all__ = ["BENCHMARK_ID", "OFFICIAL_REQUIREMENTS", "filter_official_records"]
