"""Score fixed HarnessEval plans from current metric bundles."""

from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .aggregate import clean_float, mean_valid, numeric, skill_result
from .io import (
    atomic_write_json,
    exclusive_lock,
    file_digest,
    read_json,
    value_digest,
)

SCORER_ID = "harnesseval.score"
SCORE_SCHEMA = "harnesseval.skill_eval"
BUNDLE_SCHEMA = "harnesseval.metric_bundle"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ScoreTask:
    model_id: str
    family: str
    case_id: str
    plan_path: Path
    bundle_path: Path
    metadata_path: Path
    output_path: Path
    input_digest: str
    status: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "probe_family": self.family,
            "case_id": self.case_id,
            "plan": str(self.plan_path),
            "bundle": str(self.bundle_path),
            "metadata": str(self.metadata_path),
            "output": str(self.output_path),
            "input_digest": self.input_digest,
            "status": self.status,
            "reason": self.reason,
        }


def load_plan_index(plan_root: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    index = {}
    for path in sorted(plan_root.glob("*/*.skill_plan.json")):
        plan = read_json(path)
        case_id = str(plan["case_id"])
        if case_id in index:
            raise ValueError(f"duplicate HarnessEval plan for {case_id}")
        validation = plan.get("validation") or {}
        if (
            validation.get("status") != "ok"
            or validation.get("selection_modified") is not False
        ):
            raise ValueError(f"HarnessEval plan is not valid and unmodified: {path}")
        index[case_id] = (path.resolve(), plan)
    return index


def _missing_result(skill_id: str, reason: str) -> dict[str, Any]:
    return skill_result(
        skill_id,
        "missing_cache",
        None,
        diagnostics={"reason": reason},
    )


def resolve_selected_skills(
    plan: dict[str, Any],
    bundle: dict[str, Any],
) -> list[dict[str, Any]]:
    cached = {str(item["skill_id"]): item for item in bundle.get("skill_results") or []}
    selected = []
    for selection in plan.get("selected_skills") or []:
        skill_id = str(selection["skill_id"])
        result = copy.deepcopy(cached.get(skill_id))
        if result is None:
            result = _missing_result(skill_id, "selected skill has no bundle result")
        result["plan_role"] = str(selection["role"])
        result["plan_reason"] = str(selection.get("reason") or "")
        result["plan_parameters"] = copy.deepcopy(selection.get("parameters") or {})
        selected.append(result)
    return selected


def score_components(skills: list[dict[str, Any]]) -> dict[str, Any]:
    core = [item for item in skills if item.get("plan_role") == "core"]
    observation = [item for item in skills if item.get("plan_role") == "observation"]
    diagnostic = [item for item in skills if item.get("plan_role") == "diagnostic"]
    missing_core = [
        str(item["skill_id"]) for item in core if numeric(item.get("score")) is None
    ]
    missing_observation = [
        str(item["skill_id"])
        for item in observation
        if numeric(item.get("score")) is None
    ]
    core_score = (
        mean_valid(item.get("score") for item in core)
        if core and not missing_core
        else None
    )
    other_score = (
        mean_valid(item.get("score") for item in observation)
        if observation and not missing_observation
        else None
    )
    final_score = (
        round(0.5 * core_score + 0.5 * other_score, 6)
        if core_score is not None and other_score is not None
        else None
    )
    return {
        "core": core,
        "observation": observation,
        "diagnostic": diagnostic,
        "missing_core": missing_core,
        "missing_observation": missing_observation,
        "core_score": core_score,
        "other_score": other_score,
        "final_score": final_score,
    }


def _video_and_metadata(
    generation_root: Path,
    model_id: str,
    plan: dict[str, Any],
    bundle: dict[str, Any],
) -> tuple[Path, Path, dict[str, Any]]:
    taxonomy = plan["taxonomy"]
    axis = str(taxonomy["primary_axis"])
    family = str(taxonomy["probe_family"])
    case_id = str(plan["case_id"])
    metadata_path = (
        generation_root
        / "outputs"
        / axis
        / family
        / model_id
        / case_id
        / "metadata.json"
    )
    return _video_from_metadata(metadata_path, model_id, plan, bundle)


def _video_from_metadata(
    metadata_path: Path,
    model_id: str,
    plan: dict[str, Any],
    bundle: dict[str, Any],
) -> tuple[Path, Path, dict[str, Any]]:
    identity = (
        model_id,
        str(plan["taxonomy"]["probe_family"]),
        str(plan["case_id"]),
    )
    metadata_path = metadata_path.resolve(strict=True)
    metadata = read_json(metadata_path)
    raw_video = metadata.get("output_video") or metadata.get("video_path")
    if not raw_video:
        raise ValueError(f"generation metadata has no output video: {metadata_path}")
    video_path = Path(str(raw_video))
    if not video_path.is_absolute():
        video_path = metadata_path.parent / video_path
    video = video_path.resolve(strict=True)
    bundle_video = (bundle.get("video") or {}).get("path")
    if not bundle_video or Path(str(bundle_video)).resolve(strict=True) != video:
        raise ValueError(f"generation/cache video mismatch for {identity}")
    fingerprint = bundle.get("video") or {}
    stat = video.stat()
    if fingerprint.get("size_bytes") != stat.st_size:
        raise ValueError(f"generation video changed after cache creation: {video}")
    return metadata_path, video, metadata


def blocking_reasons(
    plan: dict[str, Any],
    bundle: dict[str, Any],
    skills: list[dict[str, Any]],
) -> list[str]:
    reasons = []
    if bundle.get("schema_version") != BUNDLE_SCHEMA:
        reasons.append(f"metric bundle schema must be {BUNDLE_SCHEMA}")
    components = score_components(skills)
    if not components["core"]:
        reasons.append("plan has no core skill")
    if components["missing_core"]:
        reasons.append(
            "missing numeric core cache: " + ", ".join(components["missing_core"])
        )
    if not components["observation"]:
        reasons.append("plan has no observation skills")
    if components["missing_observation"]:
        reasons.append(
            "missing numeric observation cache: "
            + ", ".join(components["missing_observation"])
        )
    if components["other_score"] is None:
        reasons.append("observation score is unavailable")
    return reasons


def plan_score_tasks(
    generation_root: Path,
    plan_root: Path,
    cache_root: Path,
    eval_root: Path,
    *,
    models: set[str] | None = None,
    families: set[str] | None = None,
    limit: int = 0,
) -> list[ScoreTask]:
    plan_index = load_plan_index(plan_root)
    tasks = []
    for bundle_path in sorted((cache_root / "bundles").glob("*/*/*.metrics.json")):
        model_id = bundle_path.parent.parent.name
        family = bundle_path.parent.name
        case_id = bundle_path.name.removesuffix(".metrics.json")
        if models and model_id not in models:
            continue
        if families and family not in families:
            continue
        plan_path, plan = plan_index[case_id]
        bundle = read_json(bundle_path)
        if (
            str(bundle.get("model_id") or bundle.get("model_slug")) != model_id
            or str(bundle.get("probe_family")) != family
            or str(bundle.get("case_id")) != case_id
        ):
            raise ValueError(f"cache bundle identity mismatch: {bundle_path}")
        metadata_path, _video, _metadata = _video_and_metadata(
            generation_root, model_id, plan, bundle
        )
        input_digest = value_digest(
            {
                "scorer_id": SCORER_ID,
                "plan_sha256": file_digest(plan_path),
                "bundle_sha256": file_digest(bundle_path),
                "metadata_sha256": file_digest(metadata_path),
            }
        )
        destination = (
            eval_root
            / "rollout_scores"
            / model_id
            / family
            / f"{case_id}.scores.json"
        )
        reasons = blocking_reasons(plan, bundle, resolve_selected_skills(plan, bundle))
        if reasons:
            status, reason = "blocked", "; ".join(reasons)
        elif destination.is_file():
            existing = read_json(destination)
            existing_digest = (existing.get("provenance") or {}).get("input_digest")
            if existing_digest == input_digest:
                status, reason = "cached", "identical cache and plan already scored"
            else:
                status, reason = "stale", "plan, cache, or generation metadata changed"
        else:
            status, reason = "pending", "all scoring caches are available"
        tasks.append(
            ScoreTask(
                model_id,
                family,
                case_id,
                plan_path,
                bundle_path.resolve(),
                metadata_path,
                destination,
                input_digest,
                status,
                reason,
            )
        )
        if limit and len(tasks) >= limit:
            break
    return tasks


def task_summary(tasks: list[ScoreTask], details: bool = False) -> dict[str, Any]:
    status_counts = Counter(task.status for task in tasks)
    family_status_counts = Counter((task.family, task.status) for task in tasks)
    blocked_reasons = Counter(task.reason for task in tasks if task.status == "blocked")
    result: dict[str, Any] = {
        "schema_version": "harnesseval.score_plan",
        "task_count": len(tasks),
        "status_counts": dict(sorted(status_counts.items())),
        "family_status_counts": {
            f"{family}/{status}": count
            for (family, status), count in sorted(family_status_counts.items())
        },
        "blocked_reasons": dict(sorted(blocked_reasons.items())),
    }
    if details:
        result["tasks"] = [task.to_dict() for task in tasks]
    return result


def build_score(task: ScoreTask) -> dict[str, Any]:
    plan = read_json(task.plan_path)
    bundle = read_json(task.bundle_path)
    metadata_path, video, metadata = _video_from_metadata(
        task.metadata_path, task.model_id, plan, bundle
    )
    skills = resolve_selected_skills(plan, bundle)
    reasons = blocking_reasons(plan, bundle, skills)
    if reasons:
        raise ValueError(f"task became blocked: {'; '.join(reasons)}")
    components = score_components(skills)
    core = components["core"]
    observations = components["observation"]
    core_weight = 1.0 / len(core)
    selected = plan["selected_skills"]
    score = {
        "schema_version": SCORE_SCHEMA,
        "scorer_id": SCORER_ID,
        "created_at": utc_now(),
        "case_id": task.case_id,
        "action_id": str(metadata.get("action_id") or task.case_id),
        "taxonomy": copy.deepcopy(plan["taxonomy"]),
        "model": str(metadata.get("model") or task.model_id),
        "model_id": task.model_id,
        "paths": {
            "metadata": str(metadata_path),
            "output_video": str(video),
            "skill_plan": str(task.plan_path),
            "metric_bundle": str(task.bundle_path),
        },
        "video_validity": {
            "status": "ok",
            "score": 1.0,
            "score_type": "cache_fingerprint_verified",
        },
        "skill_plan": {
            "schema_version": plan.get("schema_version"),
            "routing_mode": plan.get("routing_mode"),
            "selected_skill_ids": [str(item["skill_id"]) for item in selected],
            "observation_skill_ids": [
                str(item["skill_id"])
                for item in selected
                if item["role"] == "observation"
            ],
            "core_skill_ids": [
                str(item["skill_id"]) for item in selected if item["role"] == "core"
            ],
            "diagnostic_skill_ids": [
                str(item["skill_id"])
                for item in selected
                if item["role"] == "diagnostic"
            ],
            "selected_skills": copy.deepcopy(selected),
        },
        "skills": skills,
        "case_score": {
            "score": components["core_score"],
            "status": "ok",
            "score_type": "selected_core_mean",
            "core_skill_ids": [str(item["skill_id"]) for item in core],
            "weights": {str(item["skill_id"]): core_weight for item in core},
            "routing_mode": plan.get("routing_mode"),
        },
        "observation_quality": {
            "status": "ok",
            "dimensions": {
                str(item["skill_id"]): item.get("score") for item in observations
            },
            "evidence_score": components["other_score"],
            "score_type": "selected_observation_mean",
        },
        "final_score": {
            "score": components["final_score"],
            "status": "ok",
            "score_type": "core_others_50_50",
            "core_score": components["core_score"],
            "others_score": components["other_score"],
            "core_weight": 0.5,
            "others_weight": 0.5,
        },
        "provenance": {
            "scorer_id": SCORER_ID,
            "input_digest": task.input_digest,
            "cache_only": True,
        },
    }
    return clean_float(score)


def execute_score_tasks(
    tasks: Iterable[ScoreTask],
    eval_root: Path,
    *,
    refresh_stale: bool = False,
) -> dict[str, Any]:
    tasks = list(tasks)
    eligible = [
        task
        for task in tasks
        if task.status == "pending" or (refresh_stale and task.status == "stale")
    ]
    written = []
    with exclusive_lock(
        eval_root / ".score.lock",
        {"scorer_id": SCORER_ID, "task_count": len(tasks)},
    ):
        for task in eligible:
            atomic_write_json(task.output_path, build_score(task))
            written.append(str(task.output_path))
    return {
        "schema_version": "harnesseval.score_execution",
        "requested": len(tasks),
        "eligible": len(eligible),
        "written": len(written),
        "blocked": sum(task.status == "blocked" for task in tasks),
        "cached": sum(task.status == "cached" for task in tasks),
        "stale_not_refreshed": (
            sum(task.status == "stale" for task in tasks) if not refresh_stale else 0
        ),
        "written_paths": written,
    }
