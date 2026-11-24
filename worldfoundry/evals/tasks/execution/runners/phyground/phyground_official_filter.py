"""PhyGround official result filtering and contract metadata."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from worldfoundry.evals.tasks.execution.framework.official_result_scoring import _loose_id

BENCHMARK_ID = "phyground"

OFFICIAL_REQUIREMENTS: dict[str, Any] = {
    "reason": "judge_required",
    "required_inputs": [
        "NU-World-Model-Embodied-AI/phyground prompts/phyground.json",
        "first_images conditioning assets",
        "generated videos named by id_stem",
        "PhyJudge-9B/vLLM scores.json or Gemini/GPT/Claude judge scores",
        "per-video SA, PTV, persistence, and physical.laws scores",
    ],
}


def filter_official_records(
    records: list[Mapping[str, Any]],
    *,
    generated_artifact_dir: Path | None = None,
    result_model_id: str | None = None,
) -> list[Mapping[str, Any]]:
    if not records:
        return records
    model_id = result_model_id or (generated_artifact_dir.name if generated_artifact_dir is not None else None)
    if not model_id:
        return records
    model_rows = [record for record in records if record.get("model") not in (None, "")]
    if not model_rows:
        return records
    exact = [record for record in model_rows if str(record.get("model")) == model_id]
    if exact:
        return exact
    normalized_model_id = _loose_id(model_id)
    loose = [record for record in model_rows if _loose_id(str(record.get("model"))) == normalized_model_id]
    return loose or records
