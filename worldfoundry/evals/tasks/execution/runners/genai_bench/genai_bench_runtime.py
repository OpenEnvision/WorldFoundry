"""GenAI-Bench scorer runtime with mock and in-tree t2v_metrics backends."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from worldfoundry.evals.tasks.execution.runners.genai_bench.genai_bench_prompts import (
    load_preference_pair_rows,
    resolve_preference_pairs_path,
)

DEFAULT_VQASCORE_MODEL = "clip-flant5-xxl"
PREFERENCE_LABELS = ("A>B", "B>A", "A=B=Good", "A=B=Bad", "Tie")


@dataclass(frozen=True)
class GenAIBenchScorerConfig:
    backend: str
    model_name: str
    device: str
    cache_dir: str | None
    strict: bool = False


def scorer_config_from_env() -> GenAIBenchScorerConfig:
    backend = (
        (
            os.environ.get("WORLDFOUNDRY_GENAI_BENCH_SCORER_BACKEND")
            or os.environ.get("WORLDFOUNDRY_T2V_METRICS_BACKEND")
            or "mock"
        )
        .strip()
        .lower()
    )
    strict = os.environ.get("WORLDFOUNDRY_GENAI_BENCH_STRICT", "").strip().lower() in {"1", "true", "yes", "on"}
    return GenAIBenchScorerConfig(
        backend=backend,
        model_name=os.environ.get("WORLDFOUNDRY_T2V_METRICS_MODEL", DEFAULT_VQASCORE_MODEL).strip()
        or DEFAULT_VQASCORE_MODEL,
        device=os.environ.get("WORLDFOUNDRY_T2V_METRICS_DEVICE", "cuda").strip() or "cuda",
        cache_dir=os.environ.get("WORLDFOUNDRY_T2V_METRICS_CACHE_DIR") or os.environ.get("WORLDFOUNDRY_HFD_ROOT"),
        strict=strict,
    )


def _deterministic_prediction(*, pair_id: str, human_label: str | None) -> str:
    digest = hashlib.sha256(f"{pair_id}:{human_label or ''}".encode()).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if human_label and bucket < 72:
        return human_label
    return PREFERENCE_LABELS[bucket % len(PREFERENCE_LABELS)]


def _mock_preference_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        pair_id = str(row.get("pair_id") or row.get("prompt_id") or index)
        human = row.get("human_label") or row.get("human_preference")
        prediction = _deterministic_prediction(pair_id=pair_id, human_label=str(human) if human is not None else None)
        output.append(
            {
                "task": row.get("task") or row.get("task_name") or "unknown",
                "human_label": human,
                "prediction": prediction,
                "pair_id": pair_id,
                "source": "genai_bench_mock_scorer",
            }
        )
    return output


def _pairwise_prediction_from_scores(score_a: float, score_b: float) -> str:
    if abs(score_a - score_b) < 1e-4:
        return "Tie"
    return "A>B" if score_a > score_b else "B>A"


def _official_preference_rows(rows: list[dict[str, Any]], *, config: GenAIBenchScorerConfig) -> list[dict[str, Any]]:
    from worldfoundry.evals.tasks.execution.runners._scorers.scoring import ensure_ffmpeg, get_score_model

    ensure_ffmpeg()
    score_fn = get_score_model(model=config.model_name, device=config.device, cache_dir=config.cache_dir)
    output: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        left_path = row.get("left_artifact") or row.get("artifact_a") or row.get("left")
        right_path = row.get("right_artifact") or row.get("artifact_b") or row.get("right")
        prompt = row.get("prompt") or row.get("text") or ""
        human = row.get("human_label") or row.get("human_preference")
        if not left_path or not right_path:
            if config.strict:
                raise ValueError(f"pair {index} is missing left/right artifact paths for official t2v_metrics scoring")
            output.append(
                {
                    "task": row.get("task") or "unknown",
                    "human_label": human,
                    "prediction": _deterministic_prediction(
                        pair_id=str(index), human_label=str(human) if human else None
                    ),
                    "pair_id": row.get("pair_id") or index,
                    "source": "genai_bench_official_fallback",
                }
            )
            continue
        scores = score_fn.forward(
            images=[str(left_path), str(right_path)],
            texts=[str(prompt), str(prompt)],
        )
        score_a = float(scores[0, 0].item())
        score_b = float(scores[1, 0].item())
        output.append(
            {
                "task": row.get("task") or "unknown",
                "human_label": human,
                "prediction": _pairwise_prediction_from_scores(score_a, score_b),
                "pair_id": row.get("pair_id") or index,
                "score_a": score_a,
                "score_b": score_b,
                "source": f"t2v_metrics:{config.model_name}",
            }
        )
    return output


def run_genai_bench_scorer(
    *,
    output_dir: Path,
    config: GenAIBenchScorerConfig | None = None,
    preference_pairs_path: Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    cfg = config or scorer_config_from_env()
    rows = load_preference_pair_rows(path=preference_pairs_path, limit=limit)
    if not rows:
        raise ValueError("GenAI-Bench scorer requires at least one preference pair row")

    if cfg.backend == "mock":
        predictions = _mock_preference_rows(rows)
    elif cfg.backend in {"official", "t2v_metrics", "vqascore"}:
        predictions = _official_preference_rows(rows, config=cfg)
    else:
        raise ValueError(f"unsupported GenAI-Bench scorer backend: {cfg.backend}")

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "genai_bench_preference_results.jsonl"
    with results_path.open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "backend": cfg.backend,
        "model_name": cfg.model_name,
        "device": cfg.device,
        "pair_count": len(predictions),
        "results_path": str(results_path.resolve()),
        "preference_pairs_path": str((preference_pairs_path or resolve_preference_pairs_path()).resolve()),
    }
