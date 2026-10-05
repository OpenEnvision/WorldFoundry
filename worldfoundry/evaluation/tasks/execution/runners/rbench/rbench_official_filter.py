"""RBench official result filtering and contract metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from worldfoundry.evaluation.tasks.execution.framework.official_result_scoring import _loose_id

BENCHMARK_ID = "rbench"

OFFICIAL_REQUIREMENTS: dict[str, Any] = {
    "reason": "vlm_judge_and_operator_stack_required",
    "required_inputs": [
        "DAGroup-PKU/RBench prompts and conditioning images for all 650 prompts",
        "generated videos named 0001.mp4 ... per split under videos/",
        "GPT or Qwen3-VL judge outputs: VQA results.csv per embodiment question and per task rubric",
        "motion operator outputs: motion/results.json with amplitude and smoothness scores",
    ],
}

_MODEL_KEYS = ("model", "model_id", "result_model_id", "i2v_model_name")


def _record_model_id(record: Mapping[str, Any]) -> str | None:
    for key in _MODEL_KEYS:
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def filter_official_records(
    records: list[Mapping[str, Any]],
    *,
    generated_artifact_dir: Path | None = None,
    result_model_id: str | None = None,
) -> list[Mapping[str, Any]]:
    """Restrict imported RBench summary rows to a single generation model."""
    if not records:
        return records
    model_id = result_model_id or (generated_artifact_dir.name if generated_artifact_dir is not None else None)
    if not model_id:
        return records
    labelled = [record for record in records if _record_model_id(record) not in (None, "")]
    if not labelled:
        return records
    exact = [record for record in labelled if _record_model_id(record) == model_id]
    if exact:
        return exact
    normalized = _loose_id(model_id)
    loose = [record for record in labelled if _loose_id(str(_record_model_id(record))) == normalized]
    return loose or records


__all__ = ["BENCHMARK_ID", "OFFICIAL_REQUIREMENTS", "filter_official_records"]
