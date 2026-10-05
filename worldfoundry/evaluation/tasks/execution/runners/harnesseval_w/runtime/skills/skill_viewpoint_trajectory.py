"""HarnessEval Viewpoint Trajectory pose cache and WBench NavScore."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ..aggregate import numeric, skill_result
from ..io import atomic_write_json, file_digest, file_fingerprint, read_json, value_digest
from .common import MetricCacheContract, SkillBackendIdentity, skill_cache_path


SKILL_ID = "viewpoint_trajectory_verifier"
DEFINITION_VERSION = "harnesseval.viewpoint_trajectory.navscore"
CACHE_SCHEMA = "harnesseval.viewpoint_trajectory.cache"
BACKEND_OUTPUT_SCHEMA = "harnesseval.viewpoint_trajectory.backend_output"
SAMPLING = {"target_fps": 15.0, "pose_key": "cam_c2w"}
CACHE_CONTRACT = MetricCacheContract(
    version="harnesseval.viewpoint_trajectory.cache_contract",
    source_skill_ids=(SKILL_ID,),
    required_metrics=("NavScore",),
    accepts_source_skill_score=False,
    compatibility_fields=(
        "video_fingerprint",
        "pose_sha256",
        "megasam_version",
        "navigation_metric_version",
        "case_navigation_input",
    ),
)

__all__ = [
    "SKILL_ID",
    "DEFINITION_VERSION",
    "CACHE_SCHEMA",
    "CACHE_CONTRACT",
    "BACKEND_OUTPUT_SCHEMA",
    "ViewpointTrajectoryBackend",
    "cache_path_for_video",
    "pose_path_for_video",
    "navigation_case_data",
    "score_pose",
    "install_pose",
    "evaluate",
]


class ViewpointTrajectoryBackend(SkillBackendIdentity, Protocol):
    def evaluate(
        self,
        video_path: Path,
        video_fingerprint: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


def _controls(chunk: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    action = chunk.get("action")
    if isinstance(action, Mapping):
        controls = action.get("controls")
        if controls is None and action.get("type") == "navigation":
            controls = [action]
    else:
        controls = None
    controls = controls if controls is not None else chunk.get("controls")
    controls = controls if controls is not None else chunk.get("actions")
    if isinstance(controls, Mapping):
        return [controls]
    if isinstance(controls, list):
        return [item for item in controls if isinstance(item, Mapping)]
    return []


def _chunk_text(chunk: Mapping[str, Any]) -> str:
    action = chunk.get("action")
    if isinstance(action, Mapping) and action.get("text") is not None:
        return str(action["text"])
    return str(chunk.get("text") or "")


def _text_navigation_action(text: str) -> str | None:
    """Normalize explicit English camera/navigation phrases to WBench actions."""

    text = " ".join(text.lower().split())
    if not text:
        return None
    forward = bool(
        re.search(
            r"\b(?:move|go|walk|travel|continue|proceed|drive|fly)\b"
            r"[^.;]{0,48}\b(?:forward|ahead|straight)\b|\badvance\b",
            text,
        )
    )
    backward = bool(
        re.search(
            r"\b(?:move|go|walk|travel|drive|fly)\b[^.;]{0,48}"
            r"\b(?:backward|backwards)\b|\bback away\b",
            text,
        )
    )
    left = bool(
        re.search(
            r"\b(?:turn|pan|look|rotate|pivot|swing)\b[^.;]{0,64}"
            r"\b(?:left|leftward)\b",
            text,
        )
    )
    right = bool(
        re.search(
            r"\b(?:turn|pan|look|rotate|pivot|swing)\b[^.;]{0,64}"
            r"\b(?:right|rightward)\b",
            text,
        )
    )
    up = bool(re.search(r"\b(?:look|tilt|pitch)\b[^.;]{0,32}\bup(?:ward)?\b", text))
    down = bool(
        re.search(r"\b(?:look|tilt|pitch)\b[^.;]{0,32}\bdown(?:ward)?\b", text)
    )
    strafe_left = bool(
        re.search(r"\b(?:strafe|slide|step)\b[^.;]{0,32}\b(?:left|leftward)\b", text)
    )
    strafe_right = bool(
        re.search(r"\b(?:strafe|slide|step)\b[^.;]{0,32}\b(?:right|rightward)\b", text)
    )

    if forward and left:
        return "W+left"
    if forward and right:
        return "W+right"
    if forward and up:
        return "W+up"
    if forward and down:
        return "W+down"
    choices = [
        (forward, "W"),
        (backward, "S"),
        (strafe_left, "A"),
        (strafe_right, "D"),
        (left, "left"),
        (right, "right"),
        (up, "up"),
        (down, "down"),
    ]
    selected = [action for matched, action in choices if matched]
    return selected[0] if len(selected) == 1 else None


def navigation_case_data(case: Mapping[str, Any]) -> dict[str, Any]:
    """Build the exact case object consumed by WBench evaluate_navigation."""

    interaction = case.get("interaction") or {}
    action = interaction.get("action") if isinstance(interaction, Mapping) else {}
    action = action if isinstance(action, Mapping) else {}
    chunks = action.get("chunks") or []
    interactions = []
    for chunk in chunks:
        if not isinstance(chunk, Mapping):
            continue
        controls = _controls(chunk)
        for item in controls:
            token = item.get("action")
            if token is not None:
                interactions.append({"type": str(item.get("type") or ""), "action": str(token)})
        if not controls:
            text = _chunk_text(chunk)
            alias = _text_navigation_action(text)
            if text:
                interactions.append(
                    {
                        "type": "navigation" if alias else "text_chunk",
                        "action": alias or text,
                    }
                )
    if not interactions and action.get("action") is not None:
        interactions.append(
            {"type": str(action.get("type") or ""), "action": str(action["action"])}
        )
    if not interactions:
        raise ValueError("viewpoint case has no action turns")
    world = case.get("world") or {}
    source_tags = world.get("source_tags") if isinstance(world, Mapping) else {}
    source_tags = source_tags if isinstance(source_tags, Mapping) else {}
    return {
        "interactions": interactions,
        "settings": {"perspective": str(source_tags.get("perspective") or "first_person")},
    }


def cache_path_for_video(video_path: Path, cache_root: Path | None = None) -> Path:
    return skill_cache_path(video_path, "viewpoint_trajectory", cache_root)


def pose_path_for_video(video_path: Path, cache_root: Path | None = None) -> Path:
    return cache_path_for_video(video_path, cache_root).with_suffix(".pose.npz")


def _validated_poses(path: Path) -> np.ndarray:
    try:
        with np.load(path, allow_pickle=False) as payload:
            if "cam_c2w" not in payload:
                raise ValueError("MegaSAM pose cache is missing cam_c2w")
            poses = np.asarray(payload["cam_c2w"], dtype=np.float64)
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid MegaSAM pose cache {path}: {exc}") from exc
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) < 2:
        raise ValueError(f"invalid MegaSAM cam_c2w shape: {poses.shape}")
    if not np.isfinite(poses).all():
        raise ValueError("MegaSAM cam_c2w contains non-finite values")
    return poses


def _write_poses(path: Path, poses: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez(handle, cam_c2w=poses.astype(np.float32, copy=False))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def install_pose(
    video_path: Path,
    source_pose: Path,
    cache_root: Path,
) -> Path:
    """Install a validated existing MegaSAM pose into this skill's cache."""

    poses = _validated_poses(source_pose.resolve())
    destination = pose_path_for_video(video_path.resolve(), cache_root)
    if destination.is_file() and np.array_equal(_validated_poses(destination), poses):
        return destination
    _write_poses(destination, poses)
    return destination


def _backend_identity(backend: SkillBackendIdentity) -> dict[str, str]:
    identity = {
        "backend_id": str(backend.backend_id).strip(),
        "backend_version": str(backend.version).strip(),
        "config_digest": str(backend.config_digest).strip(),
    }
    missing = [name for name, value in identity.items() if not value]
    if missing:
        raise ValueError(f"viewpoint backend identity missing: {', '.join(missing)}")
    if str(backend.execution_mode).strip() != "local":
        raise ValueError("viewpoint backend execution_mode must be local")
    return identity


def _input_digest(
    backend: Mapping[str, str],
    video: Mapping[str, Any],
    case_data: Mapping[str, Any],
    pose_sha256: str,
) -> str:
    return value_digest(
        {
            "cache_schema": CACHE_SCHEMA,
            "skill_id": SKILL_ID,
            "definition_version": DEFINITION_VERSION,
            "cache_contract_version": CACHE_CONTRACT.version,
            "video": dict(video),
            "case_navigation_input": dict(case_data),
            "pose_sha256": pose_sha256,
            "backend": dict(backend),
            "sampling": SAMPLING,
        }
    )


def score_pose(path: Path, case_data: Mapping[str, Any]) -> Mapping[str, Any]:
    import numpy as np

    from ..metrics.navigation_trajectory import evaluate_navigation

    navigation_actions = {
        "W", "A", "S", "D", "left", "right", "up", "down",
        "W+A", "W+D", "S+D", "W+left", "W+right", "W+up", "W+down",
    }
    with np.load(path, allow_pickle=False) as payload:
        poses = payload["cam_c2w"]
    actions = [turn["action"] for turn in case_data["interactions"]]
    per_turn = len(poses) // len(actions)
    bounds = [
        (index * per_turn, min((index + 1) * per_turn, len(poses)))
        for index in range(len(actions))
    ]
    selected = [
        (action, bound)
        for action, bound in zip(actions, bounds, strict=True)
        if action in navigation_actions
    ]
    if not selected:
        return {"NavScore": None, "error": "no navigation actions"}
    nav_actions, nav_bounds = zip(*selected, strict=True)
    perspective = case_data.get("settings", {}).get("perspective", "first_person")
    return evaluate_navigation(poses, list(nav_bounds), list(nav_actions), perspective)


def result_from_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    names = (
        "NavScore",
        "Accuracy",
        "Consistency",
        "nATE_t",
        "nATE_r",
        "cnATE_t",
        "cnATE_r",
        "consistency_pairs",
        "ATE_t",
        "ATE_r",
        "total_path_length",
        "total_rotation_deg",
    )
    selected = {name: metrics.get(name) for name in names}
    score = None if isinstance(selected["NavScore"], bool) else numeric(selected["NavScore"])
    if score is not None and not 0.0 <= score <= 1.0:
        score = None
    return skill_result(
        SKILL_ID,
        "ok" if score is not None else "invalid",
        score,
        metrics=selected,
        diagnostics={
            "definition_version": DEFINITION_VERSION,
            "cache_contract_version": CACHE_CONTRACT.version,
            "aggregation": "harnesseval_navigation_trajectory_navscore",
        },
    )


def _load_cache(path: Path, input_digest: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = read_json(path)
    except (OSError, ValueError):
        return None
    provenance = payload.get("provenance") or {}
    if (
        payload.get("schema_version") != CACHE_SCHEMA
        or payload.get("skill_id") != SKILL_ID
        or payload.get("definition_version") != DEFINITION_VERSION
        or provenance.get("cache_contract_version") != CACHE_CONTRACT.version
        or provenance.get("input_digest") != input_digest
    ):
        return None
    return payload if result_from_metrics(payload.get("metrics") or {})["status"] == "ok" else None


def _response(path: Path, payload: Mapping[str, Any], cache_hit: bool) -> dict[str, Any]:
    return {
        "cache_hit": cache_hit,
        "cache_path": str(path),
        "pose_path": str(payload.get("pose_path") or ""),
        "result": result_from_metrics(payload.get("metrics") or {}),
        "provenance": payload.get("provenance") or {},
    }


def _backend_poses(evidence: Mapping[str, Any]) -> np.ndarray:
    if evidence.get("schema_version") != BACKEND_OUTPUT_SCHEMA:
        raise RuntimeError("viewpoint backend output schema mismatch")
    raw = evidence.get("cam_c2w")
    try:
        poses = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("viewpoint backend returned invalid cam_c2w") from exc
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) < 2:
        raise RuntimeError(f"viewpoint backend returned invalid cam_c2w shape: {poses.shape}")
    if not np.isfinite(poses).all():
        raise RuntimeError("viewpoint backend returned non-finite cam_c2w")
    return poses


def evaluate(
    video_path: Path,
    case: Mapping[str, Any],
    backend: ViewpointTrajectoryBackend,
    cache_root: Path | None = None,
) -> dict[str, Any]:
    """Evaluate one case/video, reusing MegaSAM poses before backend inference."""

    video_path = video_path.resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    score_path = cache_path_for_video(video_path, cache_root)
    pose_path = pose_path_for_video(video_path, cache_root)
    identity = _backend_identity(backend)
    video = file_fingerprint(video_path)
    case_data = navigation_case_data(case)

    if pose_path.is_file():
        _validated_poses(pose_path)
    else:
        poses = _backend_poses(backend.evaluate(video_path, video))
        _write_poses(pose_path, poses)
    pose_sha256 = file_digest(pose_path)
    digest = _input_digest(identity, video, case_data, pose_sha256)
    cached = _load_cache(score_path, digest)
    if cached is not None:
        return _response(score_path, cached, True)

    raw_metrics = score_pose(pose_path, case_data)
    metrics = {
        "NavScore": raw_metrics.get("NavScore"),
        "Accuracy": raw_metrics.get("accuracy"),
        "Consistency": raw_metrics.get("consistency"),
        "nATE_t": raw_metrics.get("nATE_t"),
        "nATE_r": raw_metrics.get("nATE_r"),
        "cnATE_t": raw_metrics.get("cnATE_t"),
        "cnATE_r": raw_metrics.get("cnATE_r"),
        "consistency_pairs": raw_metrics.get("consistency_pairs"),
        "ATE_t": raw_metrics.get("ATE_t"),
        "ATE_r": raw_metrics.get("ATE_r"),
        "total_path_length": raw_metrics.get("total_path_length"),
        "total_rotation_deg": raw_metrics.get("total_rotation_deg"),
    }
    result = result_from_metrics(metrics)
    if result["status"] != "ok":
        raise RuntimeError(str(raw_metrics.get("error") or "WBench returned no NavScore"))
    payload = {
        "schema_version": CACHE_SCHEMA,
        "skill_id": SKILL_ID,
        "definition_version": DEFINITION_VERSION,
        "pose_path": str(pose_path),
        "metrics": dict(result["metrics"]),
        "provenance": {
            "input_digest": digest,
            "cache_contract_version": CACHE_CONTRACT.version,
            "video": video,
            "case_navigation_input": case_data,
            "pose_sha256": pose_sha256,
            "backend": identity,
            "execution_mode": str(backend.execution_mode),
        },
    }
    atomic_write_json(score_path, payload)
    return _response(score_path, payload, False)
