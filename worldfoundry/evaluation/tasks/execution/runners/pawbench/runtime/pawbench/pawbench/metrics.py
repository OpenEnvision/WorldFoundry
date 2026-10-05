"""Deterministic PAWBench metrics over normalized judgment rows.

This module is intentionally independent of media decoding, PAWEval, providers,
and downloads.  ``compute_metrics`` is the complete public metric seam.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import math
from typing import Any


EXPECTED_ROLLOUTS = 50
EXPECTED_SCENES_PER_TRACK = 25
MAX_NULL_OBSERVATIONS = 30
TRACKS = ("calibration", "coverage")
OBSERVATIONS = ("outcome", "null_observation", "infrastructure_failure")


def compute_metrics(
    rows: Sequence[Mapping[str, Any]], scene_policy: Mapping[str, Any]
) -> dict[str, Any]:
    """Compute official PAWBench metrics from normalized rows and scene policy.

    Each row identifies one ``model_or_lane × track × scene × repeat_index``
    slot with a unique ``sample_id``.  Its observation is an in-schema outcome,
    a benchmark null observation, or an unresolved infrastructure failure.
    Infrastructure failures block the affected track rather than changing its
    benchmark denominator.
    """

    policies = validate_scene_policy(scene_policy)
    models = tuple(scene_policy["model_or_lanes"])
    indexed = _index_rows(rows, policies, models)

    tracks: dict[str, dict[str, Any]] = {}
    blockers: list[str] = []
    for track, scenes in policies.items():
        scene_results = [
            _compute_scene(model, track, scene, indexed[(model, track, scene["scene_id"])])
            for model in models
            for scene in scenes
        ]
        track_result = _compute_track(track, scenes, scene_results, models)
        tracks[track] = track_result
        blockers.extend(f"{track}:{blocker}" for blocker in track_result["blockers"])

    return {
        "schema_version": "pawbench.metrics/v1",
        "status": "blocked" if blockers else "ok",
        "blockers": sorted(set(blockers)),
        "tracks": tracks,
    }


def validate_scene_policy(scene_policy: Mapping[str, Any]) -> dict[str, list[Mapping[str, Any]]]:
    """Validate the released 25-calibration/25-coverage scene contract.

    This remains an internal helper: ``compute_metrics`` is the public metric
    interface.  ``evaluate`` also calls it before media extraction so an
    invalid local benchmark cannot trigger provider requests.
    """
    if not isinstance(scene_policy, Mapping):
        raise ValueError("scene policy must be an object")
    models = scene_policy.get("model_or_lanes")
    if not isinstance(models, Sequence) or isinstance(models, (str, bytes)) or not models:
        raise ValueError("scene policy model_or_lanes must be a nonempty list")
    if any(not isinstance(model, str) or not model for model in models):
        raise ValueError("scene policy model_or_lanes must contain nonempty strings")
    if len(set(models)) != len(models):
        raise ValueError("scene policy model_or_lanes must be unique")

    track_policies = scene_policy.get("tracks")
    if not isinstance(track_policies, Mapping) or set(track_policies) != set(TRACKS):
        raise ValueError("scene policy must declare exactly calibration and coverage tracks")

    validated: dict[str, list[Mapping[str, Any]]] = {}
    for track in TRACKS:
        track_policy = track_policies[track]
        if not isinstance(track_policy, Mapping):
            raise ValueError(f"{track} track policy must be an object")
        scenes = track_policy.get("scenes")
        if not isinstance(scenes, Sequence) or isinstance(scenes, (str, bytes)):
            raise ValueError(f"{track} track policy scenes must be a list")
        if len(scenes) != EXPECTED_SCENES_PER_TRACK:
            raise ValueError(f"{track} track policy must declare exactly 25 scenes")

        seen_ids: set[str] = set()
        valid_scenes: list[Mapping[str, Any]] = []
        for scene in scenes:
            if not isinstance(scene, Mapping):
                raise ValueError(f"{track} scene policy must be an object")
            scene_id = scene.get("scene_id")
            if not isinstance(scene_id, str) or not scene_id:
                raise ValueError(f"{track} scene_id must be a nonempty string")
            if scene_id in seen_ids:
                raise ValueError(f"duplicate {track} scene policy: {scene_id}")
            seen_ids.add(scene_id)
            if scene.get("track") != track:
                raise ValueError(f"scene {scene_id} has a track-policy mismatch")
            if not isinstance(scene.get("group"), str) or not scene["group"]:
                raise ValueError(f"scene {scene_id} requires a nonempty group")

            repeats = scene.get("expected_repeat_indices")
            if not isinstance(repeats, Sequence) or isinstance(repeats, (str, bytes)):
                raise ValueError(f"scene {scene_id} expected_repeat_indices must be a list")
            normalized_repeats = [_normalize_repeat(repeat) for repeat in repeats]
            if len(normalized_repeats) != EXPECTED_ROLLOUTS or len(set(normalized_repeats)) != EXPECTED_ROLLOUTS:
                raise ValueError(f"scene {scene_id} must declare exactly 50 unique repeat indices")

            if track == "calibration":
                reference = scene.get("reference_distribution")
                if not isinstance(reference, Mapping) or not reference:
                    raise ValueError(f"calibration scene {scene_id} requires reference_distribution")
                try:
                    values = [float(value) for value in reference.values()]
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"calibration scene {scene_id} reference must be numeric") from exc
                if any(not math.isfinite(value) or value < 0 for value in values) or abs(sum(values) - 1.0) > 1e-9:
                    raise ValueError(f"calibration scene {scene_id} reference must sum to one")
            else:
                support = scene.get("support_labels")
                if not isinstance(support, Sequence) or isinstance(support, (str, bytes)) or not support:
                    raise ValueError(f"coverage scene {scene_id} requires support_labels")
                if any(not isinstance(label, str) or not label for label in support):
                    raise ValueError(f"coverage scene {scene_id} support_labels must contain nonempty strings")
                if len(set(support)) != len(support):
                    raise ValueError(f"coverage scene {scene_id} support_labels must be unique")
            valid_scenes.append(scene)
        validated[track] = valid_scenes
    return validated


def _normalize_repeat(value: Any) -> int | str:
    if isinstance(value, bool) or value in (None, ""):
        raise ValueError("row repeat_index must be an integer or nonempty string")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return value
    raise ValueError("row repeat_index must be an integer or nonempty string")


def _row_key(row: Mapping[str, Any]) -> tuple[str, str, str, str, int | str]:
    required = ("sample_id", "scene_id", "track", "model_or_lane", "repeat_index")
    missing = [field for field in required if row.get(field) in (None, "")]
    if missing:
        raise ValueError(f"row key is missing fields: {missing}")
    sample_id, scene_id, track, model, repeat = (row[field] for field in required)
    if not all(isinstance(value, str) and value for value in (sample_id, scene_id, track, model)):
        raise ValueError("row key identity fields must be nonempty strings")
    return sample_id, scene_id, track, model, _normalize_repeat(repeat)


def _index_rows(
    rows: Sequence[Mapping[str, Any]],
    policies: Mapping[str, Sequence[Mapping[str, Any]]],
    models: Sequence[str],
) -> dict[tuple[str, str, str], list[Mapping[str, Any]]]:
    expected_slots = {
        (model, track, scene["scene_id"], _normalize_repeat(repeat))
        for model in models
        for track, scenes in policies.items()
        for scene in scenes
        for repeat in scene["expected_repeat_indices"]
    }
    policy_by_scene = {
        (track, scene["scene_id"]): scene for track, scenes in policies.items() for scene in scenes
    }
    indexed: dict[tuple[str, str, str, str, int | str], Mapping[str, Any]] = {}
    observed_slots: dict[tuple[str, str, str, int | str], tuple[str, str, str, str, int | str]] = {}
    sample_ids: dict[tuple[str, str, str], set[str]] = {}

    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("every metric row must be an object")
        key = _row_key(row)
        slot = (key[3], key[2], key[1], key[4])
        if slot not in expected_slots:
            raise ValueError(f"unexpected metric row key: {key}")
        if key in indexed or slot in observed_slots:
            raise ValueError(f"expected row keys contain a duplicate slot: {key}")
        scene_sample_ids = sample_ids.setdefault((key[3], key[2], key[1]), set())
        if key[0] in scene_sample_ids:
            raise ValueError(f"sample_id is reused within a scene: {key[0]!r}")
        _validate_observation(row, policy_by_scene[(key[2], key[1])])
        indexed[key] = row
        observed_slots[slot] = key
        scene_sample_ids.add(key[0])

    missing = sorted(expected_slots - set(observed_slots), key=lambda key: tuple(map(str, key)))
    if missing:
        raise ValueError(f"expected row keys are missing: {missing[:3]}")

    return {
        (model, track, scene["scene_id"]): [
            indexed[observed_slots[(model, track, scene["scene_id"], _normalize_repeat(repeat))]]
            for repeat in scene["expected_repeat_indices"]
        ]
        for model in models
        for track, scenes in policies.items()
        for scene in scenes
    }


def _validate_observation(row: Mapping[str, Any], scene: Mapping[str, Any]) -> None:
    observation = row.get("observation")
    if observation not in OBSERVATIONS:
        raise ValueError(f"unsupported row observation: {observation!r}")
    label = row.get("outcome_label")
    if observation == "outcome":
        if not isinstance(label, str) or not label:
            raise ValueError("outcome rows require a nonempty outcome_label")
        allowed = scene["reference_distribution"] if scene["track"] == "calibration" else scene["support_labels"]
        if label not in allowed:
            raise ValueError(f"outcome label outside scene ontology: {label!r}")
    elif label is not None:
        raise ValueError(f"{observation} rows cannot carry an outcome_label")
    elif observation == "infrastructure_failure" and not isinstance(row.get("failure_code"), str):
        raise ValueError("infrastructure_failure rows require a failure_code")


def _compute_scene(
    model: str, track: str, scene: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    null_observations = sum(row["observation"] == "null_observation" for row in rows)
    infrastructure_failures = sum(row["observation"] == "infrastructure_failure" for row in rows)
    valid_labels = [row["outcome_label"] for row in rows if row["observation"] == "outcome"]
    result = {
        "model_or_lane": model,
        "track": track,
        "scene_id": scene["scene_id"],
        "group": scene["group"],
        "expected_rollouts": EXPECTED_ROLLOUTS,
        "valid_outcomes": len(valid_labels),
        "null_observations": null_observations,
        "infrastructure_failures": infrastructure_failures,
    }
    if infrastructure_failures:
        return {**result, "status": "blocked", "metric": None}
    if null_observations > MAX_NULL_OBSERVATIONS:
        return {**result, "status": "fail", "metric": None}

    if track == "calibration":
        counts = Counter(valid_labels)
        reference = scene["reference_distribution"]
        tvd = 0.5 * sum(abs(counts[label] / len(valid_labels) - float(probability)) for label, probability in reference.items())
        metric = {"name": "calibration_tvd_percent", "value": round(tvd * 100, 12)}
    else:
        support = set(scene["support_labels"])
        metric = {"name": "coverage_percent", "value": round(len(set(valid_labels) & support) / len(support) * 100, 12)}
    return {**result, "status": "pass", "metric": metric}


def _compute_track(
    track: str,
    scene_specs: Sequence[Mapping[str, Any]],
    scenes: Sequence[Mapping[str, Any]],
    models: Sequence[str],
) -> dict[str, Any]:
    blockers = [
        f"infrastructure_failure:{scene['model_or_lane']}:{scene['scene_id']}"
        for scene in scenes
        if scene["status"] == "blocked"
    ]
    return {
        "status": "blocked" if blockers else "ok",
        "blockers": sorted(blockers),
        "models": {
            model: _compute_model_track(track, scene_specs, [scene for scene in scenes if scene["model_or_lane"] == model])
            for model in models
        },
    }


def _compute_model_track(
    track: str, scene_specs: Sequence[Mapping[str, Any]], scenes: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    blockers = [f"infrastructure_failure:{scene['scene_id']}" for scene in scenes if scene["status"] == "blocked"]
    passing = [scene for scene in scenes if scene["status"] == "pass"]
    groups = {}
    for group in dict.fromkeys(scene["group"] for scene in scene_specs):
        group_scenes = [scene for scene in scenes if scene["group"] == group]
        group_passing = [scene for scene in group_scenes if scene["status"] == "pass"]
        groups[group] = {
            "passing_scenes": len(group_passing),
            "scene_denominator": len(group_scenes),
            "metric": None if blockers else _average_metric(track, group_passing),
        }
    return {
        "status": "blocked" if blockers else "ok",
        "blockers": sorted(blockers),
        "scenes": list(scenes),
        "groups": groups,
        "track_average": None if blockers else _average_metric(track, passing),
        "scene_pass_rate": None if blockers else {
            "passing_scenes": len(passing),
            "scene_denominator": EXPECTED_SCENES_PER_TRACK,
            "value": round(len(passing) / EXPECTED_SCENES_PER_TRACK * 100, 12),
        },
    }


def _average_metric(track: str, scenes: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    if not scenes:
        return None
    return {
        "name": "calibration_tvd_percent" if track == "calibration" else "coverage_percent",
        "value": round(sum(scene["metric"]["value"] for scene in scenes) / len(scenes), 12),
    }
