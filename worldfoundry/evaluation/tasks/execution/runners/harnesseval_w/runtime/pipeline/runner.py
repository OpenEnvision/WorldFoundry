"""Plan-selected, cache-first execution of HarnessEval skills."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..aggregate import numeric
from ..io import (
    atomic_write_json,
    exclusive_lock,
    file_digest,
    file_fingerprint,
    read_json,
)
from ..protocols import SKILLS
from .planner import load_cases
from .registry import evaluate_skill, get_skill


RUNNER_ID = "harnesseval.runner"
BUNDLE_SCHEMA = "harnesseval.metric_bundle"
SHARED_ANALYZE_SKILLS = {
    "intentional_change_verifier_vlm",
    "physical_response_verifier_vlm",
    "offscreen_evolution_verifier",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def shard_skill_tasks(
    tasks: Iterable[SkillTask], shard_count: int, shard_index: int
) -> list[SkillTask]:
    """Balance cases within every skill/family without splitting their models."""

    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard_index must satisfy 0 <= shard_index < shard_count")
    tasks = list(tasks)
    grouped_cases: dict[tuple[str, str], set[str]] = defaultdict(set)
    for task in tasks:
        grouped_cases[(task.skill_id, task.probe_family)].add(task.case_id)
    owners = {}
    for group, case_ids in sorted(grouped_cases.items()):
        seed = f"{group[0]}\0{group[1]}".encode("utf-8")
        offset = int.from_bytes(hashlib.sha256(seed).digest()[:8], "big") % shard_count
        for position, case_id in enumerate(sorted(case_ids)):
            owners[(*group, case_id)] = (offset + position) % shard_count
    return [
        task
        for task in tasks
        if owners[(task.skill_id, task.probe_family, task.case_id)] == shard_index
    ]


@dataclass(frozen=True)
class SkillTask:
    case_id: str
    model_id: str
    primary_axis: str
    probe_family: str
    skill_id: str
    selection: Mapping[str, Any]
    case: Mapping[str, Any]
    plan_path: Path
    plan_input_digest: str
    video_path: Path
    metadata_path: Path
    initial_observation_path: Path | None
    bundle_path: Path

    def key(self) -> tuple[str, str, str]:
        return self.skill_id, self.case_id, self.model_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "model_id": self.model_id,
            "primary_axis": self.primary_axis,
            "probe_family": self.probe_family,
            "skill_id": self.skill_id,
            "plan_role": self.selection.get("role"),
            "video": str(self.video_path),
            "metadata": str(self.metadata_path),
            "initial_observation": (
                str(self.initial_observation_path)
                if self.initial_observation_path
                else None
            ),
            "plan": str(self.plan_path),
            "bundle": str(self.bundle_path),
        }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"inventory row {line_number} is not an object: {path}")
        rows.append(value)
    return rows


def _plan_index(plan_root: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    index = {}
    for path in sorted(plan_root.glob("*/*.skill_plan.json")):
        plan = read_json(path)
        case_id = str(plan.get("case_id") or "")
        if not case_id:
            raise ValueError(f"plan has no case_id: {path}")
        if case_id in index:
            raise ValueError(f"duplicate plan for case {case_id}")
        validation = plan.get("validation") or {}
        if (
            validation.get("status") != "ok"
            or validation.get("selection_modified") is not False
        ):
            raise ValueError(f"plan is not a valid unmodified LLM selection: {path}")
        index[case_id] = (path.resolve(), plan)
    return index


def plan_skill_tasks(
    inventory_root: Path,
    manifests: Iterable[Path],
    plan_root: Path,
    cache_root: Path,
    *,
    skill_ids: set[str] | None = None,
    model_ids: set[str] | None = None,
    case_ids: set[str] | None = None,
    limit: int = 0,
    shard_count: int = 1,
    shard_index: int = 0,
) -> list[SkillTask]:
    """Expand every expected rollout using its unmodified case-level plan."""

    inventory_root = inventory_root.resolve(strict=True)
    plan_root = plan_root.resolve(strict=True)
    cache_root = cache_root.resolve()
    audit = read_json(inventory_root / "INPUT_AUDIT.json")
    if audit.get("status") != "passed":
        raise ValueError("HarnessEval runner requires a passed INPUT_AUDIT.json")
    rows = _read_jsonl(inventory_root / "rollout_matrix.jsonl")
    if len(rows) != audit.get("expected_rollout_count"):
        raise ValueError("rollout matrix count does not match INPUT_AUDIT.json")
    if any(row.get("status") != "ok" for row in rows):
        raise ValueError("rollout matrix contains invalid inputs")

    requested_skills = set(skill_ids or ())
    unknown = requested_skills - set(SKILLS)
    if unknown:
        raise ValueError("unknown HarnessEval skills: " + ", ".join(sorted(unknown)))
    cases = {str(case["case_id"]): case for case in load_cases(manifests)}
    plans = _plan_index(plan_root)
    tasks = []
    for row in rows:
        case_id = str(row["case_id"])
        model_id = str(row["model_id"])
        family = str(row["probe_family"])
        axis = str(row["primary_axis"])
        if model_ids and model_id not in model_ids:
            continue
        if case_ids and case_id not in case_ids:
            continue
        case = cases.get(case_id)
        if case is None:
            raise ValueError(f"inventory case is absent from manifest: {case_id}")
        plan_entry = plans.get(case_id)
        if plan_entry is None:
            raise ValueError(f"inventory case has no HarnessEval plan: {case_id}")
        plan_path, plan = plan_entry
        if str((plan.get("taxonomy") or {}).get("probe_family") or "") != family:
            raise ValueError(f"inventory/plan family mismatch for {case_id}")

        paths = row.get("paths") or {}
        video = Path(str(paths.get("video") or "")).resolve(strict=True)
        metadata = Path(str(paths.get("metadata") or "")).resolve(strict=True)
        raw_initial = paths.get("initial_observation")
        initial = Path(str(raw_initial)).resolve(strict=True) if raw_initial else None
        plan_digest = file_digest(plan_path)
        for selection in plan.get("selected_skills") or ():
            skill_id = str(selection.get("skill_id") or "")
            get_skill(skill_id)
            if requested_skills and skill_id not in requested_skills:
                continue
            tasks.append(
                SkillTask(
                    case_id=case_id,
                    model_id=model_id,
                    primary_axis=axis,
                    probe_family=family,
                    skill_id=skill_id,
                    selection=selection,
                    case=case,
                    plan_path=plan_path,
                    plan_input_digest=plan_digest,
                    video_path=video,
                    metadata_path=metadata,
                    initial_observation_path=initial,
                    bundle_path=(
                        cache_root
                        / "bundles"
                        / model_id
                        / family
                        / f"{case_id}.metrics.json"
                    ),
                )
            )
    sharded = shard_skill_tasks(tasks, shard_count, shard_index)
    return sharded[:limit] if limit else sharded


def task_summary(tasks: Iterable[SkillTask], *, details: bool = False) -> dict[str, Any]:
    tasks = list(tasks)
    by_skill = Counter(task.skill_id for task in tasks)
    by_family = Counter(task.probe_family for task in tasks)
    result: dict[str, Any] = {
        "schema_version": "harnesseval.skill_run_plan",
        "task_count": len(tasks),
        "rollout_count": len(
            {(task.model_id, task.probe_family, task.case_id) for task in tasks}
        ),
        "case_count": len({task.case_id for task in tasks}),
        "skill_counts": dict(sorted(by_skill.items())),
        "family_counts": dict(sorted(by_family.items())),
        "shared_analyze_task_count": sum(
            task.skill_id in SHARED_ANALYZE_SKILLS for task in tasks
        ),
    }
    if details:
        result["tasks"] = [task.to_dict() for task in tasks]
    return result


def _base_bundle(task: SkillTask) -> dict[str, Any]:
    if task.bundle_path.is_file():
        existing = read_json(task.bundle_path)
        expected_video = existing.get("video") or {}
        if (
            str(existing.get("case_id")) != task.case_id
            or str(existing.get("model_id")) != task.model_id
            or str(existing.get("probe_family")) != task.probe_family
            or Path(str(expected_video.get("path") or "")).resolve()
            != task.video_path
        ):
            raise ValueError(f"bundle identity mismatch: {task.bundle_path}")
        if existing.get("schema_version") != BUNDLE_SCHEMA:
            raise ValueError(f"bundle schema must be {BUNDLE_SCHEMA}: {task.bundle_path}")
        bundle = {
            "schema_version": BUNDLE_SCHEMA,
            "case_id": task.case_id,
            "model_id": task.model_id,
            "probe_family": task.probe_family,
            "video": dict(expected_video),
            "skill_results": list(existing.get("skill_results") or ()),
            "skill_runs": dict(existing.get("skill_runs") or {}),
            "provenance": dict(existing.get("provenance") or {}),
        }
    else:
        bundle = {
            "schema_version": BUNDLE_SCHEMA,
            "case_id": task.case_id,
            "model_id": task.model_id,
            "probe_family": task.probe_family,
            "video": file_fingerprint(task.video_path),
            "skill_results": [],
            "skill_runs": {},
            "provenance": {},
        }
    bundle["plan"] = {
        "path": str(task.plan_path),
        "input_digest": task.plan_input_digest,
    }
    return bundle


def publish_skill_result(task: SkillTask, response: Mapping[str, Any]) -> None:
    result = response.get("result")
    if not isinstance(result, Mapping) or result.get("skill_id") != task.skill_id:
        raise ValueError(f"{task.skill_id} returned an invalid result")
    lock_path = task.bundle_path.parent / ".locks" / f"{task.bundle_path.name}.lock"
    with exclusive_lock(
        lock_path,
        {
            "runner_id": RUNNER_ID,
            "case_id": task.case_id,
            "model_id": task.model_id,
            "skill_id": task.skill_id,
        },
        wait=True,
    ):
        bundle = _base_bundle(task)
        selected_ids = {
            str(item.get("skill_id"))
            for item in read_json(task.plan_path).get("selected_skills") or ()
        }
        by_id = {
            str(item.get("skill_id")): dict(item)
            for item in bundle.get("skill_results") or ()
            if isinstance(item, Mapping) and str(item.get("skill_id")) in selected_ids
        }
        by_id[task.skill_id] = dict(result)
        bundle["skill_results"] = [
            by_id[skill_id] for skill_id in SKILLS if skill_id in by_id
        ]
        skill_runs = {
            key: value
            for key, value in (bundle.get("skill_runs") or {}).items()
            if key in selected_ids
        }
        skill_runs[task.skill_id] = {
            "cache_path": response.get("cache_path"),
            "cache_hit": bool(response.get("cache_hit")),
            "updated_at": utc_now(),
        }
        bundle["skill_runs"] = skill_runs
        bundle["provenance"] = {
            "runner_id": RUNNER_ID,
        }
        atomic_write_json(task.bundle_path, bundle)


def _run_one(
    task: SkillTask,
    backend: Any,
    cache_root: Path,
) -> dict[str, Any]:
    view = task.to_dict()
    try:
        metadata = read_json(task.metadata_path)
        response = evaluate_skill(
            task.skill_id,
            video_path=task.video_path,
            case=task.case,
            initial_observation_path=task.initial_observation_path,
            generation_metadata=metadata,
            backend=backend,
            cache_root=cache_root,
        )
        publish_skill_result(task, response)
        result = response["result"]
        return {
            **view,
            "status": "ok",
            "cache_hit": bool(response.get("cache_hit")),
            "cache_path": response.get("cache_path"),
            "skill_status": result.get("status"),
            "score": result.get("score"),
        }
    except Exception as exc:  # noqa: BLE001 - every failed task belongs in the audit
        return {**view, "status": "failed", "error": repr(exc)}


def _run_phase(
    tasks: list[SkillTask],
    backend: Any,
    cache_root: Path,
    workers: int,
) -> list[dict[str, Any]]:
    if not tasks:
        return []
    outcomes = []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(tasks)))) as pool:
        futures = [pool.submit(_run_one, task, backend, cache_root) for task in tasks]
        for future in as_completed(futures):
            outcomes.append(future.result())
    return outcomes


def _run_skill_group(
    tasks: list[SkillTask],
    backend: Any,
    cache_root: Path,
    workers: int,
    analyze_workers: int,
) -> list[dict[str, Any]]:
    if tasks[0].skill_id not in SHARED_ANALYZE_SKILLS:
        return _run_phase(tasks, backend, cache_root, workers)

    by_case: dict[str, list[SkillTask]] = defaultdict(list)
    for task in tasks:
        by_case[task.case_id].append(task)
    first = [sorted(items, key=lambda item: item.model_id)[0] for items in by_case.values()]
    first_keys = {task.key() for task in first}
    first_outcomes = _run_phase(first, backend, cache_root, analyze_workers)
    passed_cases = {
        str(outcome["case_id"])
        for outcome in first_outcomes
        if outcome["status"] == "ok"
    }
    remaining = [
        task
        for task in tasks
        if task.key() not in first_keys and task.case_id in passed_cases
    ]
    blocked = [
        {
            **task.to_dict(),
            "status": "blocked",
            "error": "case-level analyze_agent warmup task failed",
        }
        for task in tasks
        if task.key() not in first_keys and task.case_id not in passed_cases
    ]
    return [
        *first_outcomes,
        *_run_phase(remaining, backend, cache_root, workers),
        *blocked,
    ]


def execute_skill_tasks(
    tasks: Iterable[SkillTask],
    backends: Mapping[str, Any],
    cache_root: Path,
    *,
    workers: int = 16,
    analyze_workers: int = 16,
    workers_by_skill: Mapping[str, int] | None = None,
    audit_path: Path | None = None,
    run_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute one skill at a time and publish resumable per-video bundles."""

    tasks = list(tasks)
    if workers <= 0 or analyze_workers <= 0:
        raise ValueError("worker counts must be positive")
    required = {task.skill_id for task in tasks}
    missing = required - set(backends)
    if missing:
        raise ValueError("missing backends: " + ", ".join(sorted(missing)))
    configured_workers = dict(workers_by_skill or {})
    if any(value <= 0 for value in configured_workers.values()):
        raise ValueError("per-skill worker counts must be positive")

    cache_root = cache_root.resolve()
    outcomes = []
    by_skill: dict[str, list[SkillTask]] = defaultdict(list)
    for task in tasks:
        by_skill[task.skill_id].append(task)
    for skill_id in SKILLS:
        group = sorted(
            by_skill.get(skill_id, ()),
            key=lambda task: (task.case_id, task.model_id),
        )
        if not group:
            continue
        outcomes.extend(
            _run_skill_group(
                group,
                backends[skill_id],
                cache_root,
                configured_workers.get(skill_id, workers),
                analyze_workers,
            )
        )

    outcomes.sort(key=lambda item: (item["skill_id"], item["case_id"], item["model_id"]))
    status_counts = Counter(item["status"] for item in outcomes)
    skill_status_counts = Counter(
        (item["skill_id"], item["status"]) for item in outcomes
    )
    scores = [
        float(item["score"])
        for item in outcomes
        if item["status"] == "ok" and numeric(item.get("score")) is not None
    ]
    audit = {
        "schema_version": "harnesseval.skill_run_audit",
        "created_at": utc_now(),
        "runner_id": RUNNER_ID,
        "run_context": dict(run_context or {}),
        "plan": task_summary(tasks),
        "status": "passed" if status_counts == {"ok": len(tasks)} else "failed",
        "status_counts": dict(sorted(status_counts.items())),
        "skill_status_counts": {
            f"{skill_id}/{status}": count
            for (skill_id, status), count in sorted(skill_status_counts.items())
        },
        "backends": {
            skill_id: {
                "backend_id": str(backends[skill_id].backend_id),
                "backend_version": str(backends[skill_id].version),
                "config_digest": str(backends[skill_id].config_digest),
                "execution_mode": str(backends[skill_id].execution_mode),
                "resource_mode": getattr(backends[skill_id], "resource_mode", None),
                "model_load_count": getattr(backends[skill_id], "load_count", None),
            }
            for skill_id in SKILLS
            if skill_id in required
        },
        "cache_hits": sum(bool(item.get("cache_hit")) for item in outcomes),
        "cache_misses": sum(
            item["status"] == "ok" and not item.get("cache_hit")
            for item in outcomes
        ),
        "score_min": min(scores) if scores else None,
        "score_max": max(scores) if scores else None,
        "outcomes": outcomes,
    }
    resolved_audit_path = (audit_path or cache_root / "RUN_SKILLS_AUDIT.json").resolve()
    audit["audit_path"] = str(resolved_audit_path)
    atomic_write_json(resolved_audit_path, audit)
    return audit


def prepare_drift_stage_tasks(
    tasks: Iterable[SkillTask],
    backend: Any,
    stage: str,
    *,
    workers: int = 1,
    audit_path: Path,
    run_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Prepare one cached Drift component across the complete runner task batch."""

    from ..skills.skill_drift_degradation import resolve_chunks

    tasks = list(tasks)
    if not tasks or any(task.skill_id != "drift_degradation_analyzer" for task in tasks):
        raise ValueError("--drift-stage requires only drift_degradation_analyzer tasks")
    if workers <= 0:
        raise ValueError("worker count must be positive")

    def prepare(task: SkillTask) -> dict[str, Any]:
        view = task.to_dict()
        try:
            metadata = read_json(task.metadata_path)
            chunks = resolve_chunks(task.video_path, task.case, metadata)
            result = backend.prepare_stage(
                task.video_path,
                file_fingerprint(task.video_path),
                chunks,
                stage,
            )
            return {**view, "status": "ok", **result}
        except Exception as exc:  # noqa: BLE001
            return {**view, "status": "failed", "error": repr(exc)}

    outcomes = []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(tasks)))) as pool:
        futures = [pool.submit(prepare, task) for task in tasks]
        for future in as_completed(futures):
            outcomes.append(future.result())
    outcomes.sort(key=lambda item: (item["case_id"], item["model_id"]))
    counts = Counter(item["status"] for item in outcomes)
    audit = {
        "schema_version": "harnesseval.drift_stage_audit",
        "created_at": utc_now(),
        "runner_id": RUNNER_ID,
        "stage": stage,
        "run_context": dict(run_context or {}),
        "plan": task_summary(tasks),
        "status": "passed" if counts == {"ok": len(tasks)} else "failed",
        "status_counts": dict(sorted(counts.items())),
        "cache_hits": sum(int(item.get("cache_hits", 0)) for item in outcomes),
        "cache_misses": sum(int(item.get("cache_misses", 0)) for item in outcomes),
        "outcomes": outcomes,
        "audit_path": str(audit_path.resolve()),
    }
    atomic_write_json(audit_path.resolve(), audit)
    return audit
