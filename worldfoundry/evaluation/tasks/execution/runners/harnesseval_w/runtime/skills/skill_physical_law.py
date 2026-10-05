"""Physical-law diagnostics defined by the HarnessEval canonical specification."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ..aggregate import mean_valid, numeric, skill_result
from ..io import atomic_write_json, file_fingerprint, read_json, value_digest
from .common import MetricCacheContract, SkillBackendIdentity, skill_cache_path


SKILL_ID = "physical_law_validator"
DEFINITION_VERSION = "harnesseval.physical_law"
SIGNAL_CACHE_SCHEMA = "harnesseval.physical_law.signals"
RESULT_CACHE_SCHEMA = "harnesseval.physical_law.result"
BACKEND_OUTPUT_SCHEMA = "harnesseval.physical_law.backend_output"
ENGINE_MODEL_BY_SCENE = {
    "projectile_field": "projectile_2d",
    "newton_cradle": "newton_cradle_transfer",
    "block_push_table": "block_collision_friction_2d",
    "domino_table": "domino_chain_topple",
    "rolling_incline": "incline_rolling_friction",
    "pendulum_setup": "pendulum_period_2d",
    "water_pouring_setup": "fluid_drain_torricelli",
    "lab_workshop": "incline_rolling_friction",
    "billiards_table": "billiards_collision_2d",
}
SIGNAL_KEYS = (
    "motion",
    "cx",
    "cy",
    "onset_col",
    "onset_row",
    "left_vy",
    "right_vy",
    "active_cols",
)
CACHE_CONTRACT = MetricCacheContract(
    version="harnesseval.physical_law.cache_contract",
    source_skill_ids=(SKILL_ID,),
    required_metrics=("PhysicalLawScore",),
    accepts_source_skill_score=False,
    compatibility_fields=(
        "video_fingerprint",
        "engine_spec",
        "signal_extraction_version",
        "selected_laws",
        "definition_version",
    ),
)

__all__ = [
    "SKILL_ID",
    "DEFINITION_VERSION",
    "SIGNAL_CACHE_SCHEMA",
    "RESULT_CACHE_SCHEMA",
    "BACKEND_OUTPUT_SCHEMA",
    "ENGINE_MODEL_BY_SCENE",
    "CACHE_CONTRACT",
    "PhysicalLawBackend",
    "engine_spec",
    "selected_law_names",
    "score_signals",
    "cache_paths",
    "evaluate",
]


class PhysicalLawBackend(SkillBackendIdentity, Protocol):
    def infer(self, video_path: Path) -> Mapping[str, Any]: ...


def engine_spec(
    case: Mapping[str, Any], action_id: str | None = None
) -> dict[str, Any]:
    non_model = case.get("non_model_facing") or {}
    explicit = non_model.get("physical_engine") or non_model.get("law_engine") or {}
    if isinstance(explicit, Mapping) and explicit.get("model_id"):
        return {
            "source": "case_metadata",
            "engine_model_id": str(explicit["model_id"]),
            "sample_id": str(
                explicit.get("sample_id")
                or action_id
                or ((case.get("interaction") or {}).get("action") or {}).get("action_id")
                or ""
            ),
            "gt": dict(explicit.get("gt") or {}),
            "law_checks": list(explicit.get("law_checks") or []),
        }
    source_tags = (case.get("world") or {}).get("source_tags") or {}
    scene = str(source_tags.get("scene") or "")
    case_action_id = ((case.get("interaction") or {}).get("action") or {}).get(
        "action_id"
    )
    return {
        "source": "scene_mapping",
        "scene": scene,
        "engine_model_id": ENGINE_MODEL_BY_SCENE.get(scene),
        "sample_id": str(action_id or case_action_id or ""),
        "gt": {},
        "law_checks": [],
    }


def _array(signals: Mapping[str, Any], key: str) -> np.ndarray:
    values = [np.nan if value is None else value for value in signals.get(key) or []]
    return np.asarray(values, dtype=np.float64)


def _validated_signals(value: Mapping[str, Any]) -> dict[str, Any]:
    if value.get("schema_version") != BACKEND_OUTPUT_SCHEMA:
        raise RuntimeError("physical-law backend output schema mismatch")
    raw = value.get("signals")
    if raw is None and value.get("status") == "no_signal":
        return {}
    if not isinstance(raw, Mapping):
        raise RuntimeError("physical-law backend returned no signals")
    signals = {key: _array(raw, key) for key in SIGNAL_KEYS}
    n = int(raw.get("n") or 0)
    if n < 2 or any(len(signals[key]) != n for key in ("motion", "cx", "cy", "left_vy", "right_vy")):
        raise RuntimeError("physical-law backend returned invalid temporal signals")
    if any(len(signals[key]) != 12 for key in ("onset_col", "onset_row", "active_cols")):
        raise RuntimeError("physical-law backend returned invalid spatial signals")
    signals["n"] = n
    return signals


def _interp(values: Any) -> np.ndarray | None:
    array = np.asarray(values, dtype=np.float64)
    good = np.isfinite(array)
    if good.sum() < 3:
        return None
    indices = np.arange(len(array))
    return np.interp(indices, indices[good], array[good])


def _projectile_arc(sig: Mapping[str, Any], gt: Mapping[str, Any]) -> tuple[float, str]:
    del gt
    cx, cy = _interp(sig["cx"]), _interp(sig["cy"])
    if cx is None or cy is None or np.std(cx) < 0.05:
        return 0.0, "insufficient horizontal travel for an arc"
    order = np.argsort(cx)
    x, y = cx[order], cy[order]
    a, b, c = np.polyfit(x, y, 2)
    predicted = a * x * x + b * x + c
    r2 = 1 - np.sum((y - predicted) ** 2) / (
        np.sum((y - y.mean()) ** 2) + 1e-9
    )
    score = np.clip(0.5 * max(float(r2), 0.0) + 0.5 * (a > 0), 0.0, 1.0)
    return float(score), f"R2={r2:.2f} a={'+' if a > 0 else '-'}"


def _periodic_motion(sig: Mapping[str, Any], gt: Mapping[str, Any]) -> tuple[float, str]:
    del gt
    best = 0.0
    for key in ("cx", "cy"):
        series = _interp(sig[key])
        if series is None or np.std(series) < 0.02:
            continue
        spectrum = np.abs(np.fft.rfft((series - series.mean()) * np.hanning(len(series))))
        if len(spectrum) < 4:
            continue
        spectrum[0] = 0
        best = max(best, float(spectrum.max() / (spectrum.sum() + 1e-9)))
    return min(4 * best, 1.0), f"spectral_concentration={best:.3f}"


def _bounce_peak_decay(sig: Mapping[str, Any], gt: Mapping[str, Any]) -> tuple[float, str]:
    del gt
    cy = _interp(sig["cy"])
    if cy is None:
        return 0.0, "no track"
    height = 1 - cy
    peaks = [
        height[index]
        for index in range(1, len(height) - 1)
        if height[index] >= height[index - 1] and height[index] > height[index + 1]
        and height[index] > height.min() + 0.03
    ]
    if len(peaks) < 2:
        return 0.3, f"only {len(peaks)} peak(s) detected"
    decreasing = sum(
        peaks[index] <= peaks[index - 1] + 0.02
        for index in range(1, len(peaks))
    )
    return decreasing / (len(peaks) - 1), f"{decreasing}/{len(peaks) - 1} decreasing"


def _event_order(
    sig: Mapping[str, Any],
    gt: Mapping[str, Any],
    expect_reach_right: bool = True,
) -> tuple[float, str]:
    del gt
    onset = np.asarray(sig["onset_col"], dtype=np.float64)
    fired = np.flatnonzero(np.isfinite(onset))
    monotonicity = 0.0
    if len(fired) >= 2:
        times = onset[fired]
        monotonicity = float(np.mean(np.diff(times) >= -0.05))
    right = onset[len(onset) // 2 :]
    right_reach = min(1.5 * float(np.isfinite(right).sum()) / max(len(right), 1), 1.0)
    reach = right_reach if expect_reach_right else 1 - right_reach
    return 0.5 * monotonicity + 0.5 * reach, f"monotonicity={monotonicity:.2f} right_reach={right_reach:.2f}"


def _monotonic_motion(
    sig: Mapping[str, Any], gt: Mapping[str, Any], direction: str = "any"
) -> tuple[float, str]:
    del gt
    cy = _interp(sig["cy"])
    if cy is None:
        return 0.4, "no dominant track; weak evidence"
    delta = np.diff(cy)
    down = float(np.mean(delta > -0.005))
    up = float(np.mean(delta < 0.005))
    score = down if direction == "down" else up if direction == "up" else max(down, up)
    return score, f"monotone_fraction={score:.2f} ({direction})"


def _opposite_vertical_motion(sig: Mapping[str, Any], gt: Mapping[str, Any]) -> tuple[float, str]:
    del gt
    left = float(np.nanmean(sig["left_vy"]))
    right = float(np.nanmean(sig["right_vy"]))
    opposite = float(left * right < 0)
    strength = min(3 * (abs(left) + abs(right)), 1.0)
    return opposite * (0.5 + 0.5 * strength), f"left_vy={left:+.2f} right_vy={right:+.2f}"


def _straight_after_release(sig: Mapping[str, Any], gt: Mapping[str, Any]) -> tuple[float, str]:
    del gt
    cx, cy = _interp(sig["cx"]), _interp(sig["cy"])
    if cx is None or cy is None:
        return 0.3, "insufficient late motion"
    x, y = cx[len(cx) // 2 :], cy[len(cy) // 2 :]
    if len(x) < 4 or (np.std(x) < 0.02 and np.std(y) < 0.02):
        return 0.3, "insufficient late motion"
    slope, offset = np.linalg.lstsq(
        np.vstack([x, np.ones_like(x)]).T, y, rcond=None
    )[0]
    residual = float(np.sqrt(np.mean((y - (slope * x + offset)) ** 2)))
    return max(0.0, 1 - 20 * residual), f"late_line_residual={residual:.3f}"


def _rate_increasing(sig: Mapping[str, Any], gt: Mapping[str, Any]) -> tuple[float, str]:
    del gt
    motion = np.asarray(sig["motion"], dtype=np.float64)
    if len(motion) < 4:
        return 0.0, "too short"
    half = len(motion) // 2
    early, late = float(motion[:half].mean()), float(motion[half:].mean())
    score = 1.0 if late > 1.1 * early else np.clip(late / (early + 1e-9) - 0.5, 0, 1)
    return float(score), f"early={early:.0f} late={late:.0f}"


def _descend_and_settle(sig: Mapping[str, Any], gt: Mapping[str, Any]) -> tuple[float, str]:
    del gt
    cy = _interp(sig["cy"])
    motion = np.asarray(sig["motion"], dtype=np.float64)
    if cy is None or np.ptp(cy) < 0.06:
        return 0.2, "insufficient vertical travel"
    down = float(np.mean(np.diff(cy) > -0.01))
    half = len(motion) // 2
    settled = 1.0 if motion[half:].mean() < 0.9 * motion[:half].mean() else 0.5
    return 0.6 * down + 0.4 * settled, f"down={down:.2f} settled={settled:.1f}"


def _decelerate_to_stop(sig: Mapping[str, Any], gt: Mapping[str, Any]) -> tuple[float, str]:
    del gt
    cx = _interp(sig["cx"])
    motion = np.asarray(sig["motion"], dtype=np.float64)
    if cx is None or np.ptp(cx) < 0.08:
        return 0.2, "insufficient travel"
    half = len(motion) // 2
    early, late = float(motion[:half].mean()), float(motion[half:].mean())
    deceleration = 1.0 if late < 0.9 * early else max(0.0, 1 - late / (early + 1e-9))
    quarter = max(len(cx) // 4, 2)
    late_std = float(np.std(cx[-quarter:]))
    plateau = 1.0 if late_std < 0.03 else max(0.0, 1 - 10 * late_std)
    return 0.5 * (deceleration + plateau), f"deceleration={deceleration:.2f} plateau={plateau:.2f}"


def _horizontal_reversal(sig: Mapping[str, Any], gt: Mapping[str, Any]) -> tuple[float, str]:
    del gt
    cx = _interp(sig["cx"])
    if cx is None or np.ptp(cx) < 0.10:
        return 0.2, "insufficient travel"
    maximum, minimum = int(np.argmax(cx)), int(np.argmin(cx))
    turn = max(maximum, minimum) if min(maximum, minimum) < len(cx) * 0.2 else min(maximum, minimum)
    position = turn / len(cx)
    return float(0.2 < position < 0.9), f"turning_point={position:.2f}"


def _collision_energy_loss(sig: Mapping[str, Any], gt: Mapping[str, Any]) -> tuple[float, str]:
    del gt
    motion = np.asarray(sig["motion"], dtype=np.float64)
    if len(motion) < 6:
        return 0.0, "too short"
    peak_index = int(np.argmax(motion))
    peak = float(motion[peak_index])
    after = float(motion[peak_index + 1 :].mean()) if peak_index + 1 < len(motion) else peak
    score = np.clip(1 - after / (peak + 1e-9), 0, 1)
    return float(score), f"peak={peak_index / len(motion):.2f} post_peak_ratio={after / (peak + 1e-9):.2f}"


LawCheck = Callable[[Mapping[str, Any], Mapping[str, Any]], tuple[float, str]]


def _selected_laws(model_id: str, sample_id: str) -> list[tuple[str, LawCheck]]:
    laws: list[tuple[str, LawCheck]] = []
    if model_id in {"projectile_2d", "bank_shot_bucket_3d", "drop_vs_horizontal_launch"}:
        laws.append(("projectile_arc", _projectile_arc))
    if model_id in {"pendulum_period_2d", "spring_mass_vertical", "newton_cradle_transfer", "wrecking_pendulum_impact"}:
        laws.append(("periodic_motion", _periodic_motion))
    if model_id == "ball_bounce_restitution":
        laws.append(("bounce_peak_decay", _bounce_peak_decay))
    if model_id == "domino_chain_topple":
        laws.append(("event_order", _event_order))
    if model_id == "domino_gap_stop":
        cross = "cross" in sample_id
        laws.append(("event_order", lambda s, g: _event_order(s, g, cross)))
    if model_id in {"candle_burn_shrink", "ice_melt_puddle", "fluid_drain_torricelli"}:
        laws.append(("monotonic_motion", lambda s, g: _monotonic_motion(s, g, "down")))
    if model_id == "water_displacement_rise":
        laws.append(("monotonic_motion", lambda s, g: _monotonic_motion(s, g, "up")))
    if model_id == "freeze_surface_spread":
        laws.append(("monotonic_motion", _monotonic_motion))
    if model_id in {"balloon_vs_ball_drop", "seesaw_torque_balance", "atwood_pulley_2mass", "balance_scale_pans"}:
        laws.append(("opposite_vertical_motion", _opposite_vertical_motion))
    if model_id == "centripetal_release_tangent":
        laws.append(("straight_after_release", _straight_after_release))
    if model_id == "boiling_bubble_rate":
        laws.append(("rate_increasing", _rate_increasing))
    if model_id in {"incline_rolling_friction", "buoyancy_density_tank"}:
        laws.append(("descend_and_settle", _descend_and_settle))
    if model_id == "skid_brake_stop_distance":
        laws.append(("decelerate_to_stop", _decelerate_to_stop))
    if model_id in {"wall_bounce_angle_2d", "bank_shot_bucket_3d"}:
        laws.append(("horizontal_reversal", _horizontal_reversal))
    if model_id in {"elastic_vs_inelastic_carts", "block_collision_friction_2d"}:
        laws.append(("collision_energy_loss", _collision_energy_loss))
    return laws


def selected_law_names(model_id: str, sample_id: str = "") -> list[str]:
    return [name for name, _ in _selected_laws(model_id, sample_id)]


def score_signals(
    model_id: str,
    sample_id: str,
    signals: Mapping[str, Any],
    gt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    laws = []
    for name, check in _selected_laws(model_id, sample_id):
        try:
            score, detail = check(signals, gt or {})
            value = round(float(np.clip(score, 0, 1)), 4)
        except Exception as exc:  # noqa: BLE001 - retain per-law failure evidence
            value, detail = None, f"error: {exc!r}"
        laws.append({"law": name, "score": value, "detail": detail})
    mean_score = mean_valid(item["score"] for item in laws)
    score = round(mean_score, 4) if mean_score is not None else None
    return {"laws": laws, "law_score": score, "status": "ok" if score is not None else "flow_not_applicable"}


def cache_paths(video_path: Path, cache_root: Path | None = None) -> dict[str, Path]:
    return {
        "signals": skill_cache_path(video_path, "physical_law/signals", cache_root),
        "result": skill_cache_path(video_path, "physical_law/results", cache_root),
    }


def _identity(backend: SkillBackendIdentity) -> dict[str, str]:
    identity = {
        "backend_id": str(backend.backend_id).strip(),
        "backend_version": str(backend.version).strip(),
        "config_digest": str(backend.config_digest).strip(),
    }
    missing = [key for key, value in identity.items() if not value]
    if missing:
        raise ValueError(f"physical-law backend identity missing: {', '.join(missing)}")
    return identity


def _load(path: Path, schema: str, digest: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = read_json(path)
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("schema_version") != schema or payload.get("input_digest") != digest:
        return None
    return payload


def _result(
    status: str,
    law_score: float | None,
    laws: list[dict[str, Any]],
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    scored = [item["score"] for item in laws if numeric(item.get("score")) is not None]
    return skill_result(
        SKILL_ID,
        status,
        law_score,
        metrics={
            "PhysicalLawScore": law_score,
            "law_score": law_score,
            "law_count": len(laws),
            "scored_law_count": len(scored),
            "mean_scored_law_score": mean_valid(scored),
        },
        diagnostics={
            "definition_version": DEFINITION_VERSION,
            "engine_spec": dict(spec),
            "laws": laws,
            "reliability_assumptions": [
                "fixed camera",
                "one dominant moving subject",
                "static background",
                "sufficient subject texture",
                "moderate inter-frame displacement",
            ],
        },
        notes=[
            "This diagnostic uses only final.md optical-flow laws; unavailable laws are not forced to zero."
        ],
    )


def _response(
    paths: Mapping[str, Path],
    payload: Mapping[str, Any],
    cache_hit: bool,
    has_signals: bool,
) -> dict[str, Any]:
    return {
        "cache_hit": cache_hit,
        "cache_path": str(paths["result"]),
        "signal_cache_path": str(paths["signals"]) if has_signals else None,
        "result": payload["result"],
        "provenance": payload.get("provenance") or {},
    }


def evaluate(
    video_path: Path,
    case: Mapping[str, Any],
    backend: PhysicalLawBackend,
    cache_root: Path | None = None,
    *,
    action_id: str | None = None,
) -> dict[str, Any]:
    """Evaluate one video with cacheable raw signals and deterministic laws."""

    video_path = video_path.resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    family = str((case.get("taxonomy") or {}).get("probe_family") or "")
    if family != "physical_transition":
        raise ValueError(f"physical-law skill does not accept family {family!r}")
    identity = _identity(backend)
    video = file_fingerprint(video_path, include_sha256=True)
    spec = engine_spec(case, action_id)
    model_id = str(spec.get("engine_model_id") or "")
    sample_id = str(spec.get("sample_id") or "")
    paths = cache_paths(video_path, cache_root)
    common = {
        "skill_id": SKILL_ID,
        "definition_version": DEFINITION_VERSION,
        "cache_contract_version": CACHE_CONTRACT.version,
        "video": video,
        "engine_spec": spec,
        "backend": identity,
    }

    if not model_id:
        result_digest = value_digest({**common, "schema": RESULT_CACHE_SCHEMA})
        cached = _load(paths["result"], RESULT_CACHE_SCHEMA, result_digest)
        if cached is None:
            result = _result("not_applicable", None, [], spec)
            cached = {
                "schema_version": RESULT_CACHE_SCHEMA,
                "input_digest": result_digest,
                "skill_id": SKILL_ID,
                "definition_version": DEFINITION_VERSION,
                "result": result,
                "provenance": {"backend": identity, "video": video},
            }
            atomic_write_json(paths["result"], cached)
            return _response(paths, cached, False, False)
        return _response(paths, cached, True, False)

    selected = selected_law_names(model_id, sample_id)
    if not selected:
        result_digest = value_digest({**common, "schema": RESULT_CACHE_SCHEMA, "selected_laws": []})
        cached = _load(paths["result"], RESULT_CACHE_SCHEMA, result_digest)
        if cached is None:
            laws = [{"law": "flow_not_applicable", "score": None}]
            result = _result("flow_not_applicable", None, laws, spec)
            cached = {
                "schema_version": RESULT_CACHE_SCHEMA,
                "input_digest": result_digest,
                "skill_id": SKILL_ID,
                "definition_version": DEFINITION_VERSION,
                "result": result,
                "provenance": {"backend": identity, "video": video},
            }
            atomic_write_json(paths["result"], cached)
            return _response(paths, cached, False, False)
        return _response(paths, cached, True, False)

    signal_digest = value_digest(
        {"schema": SIGNAL_CACHE_SCHEMA, "video": video, "backend": identity}
    )
    signal_cache = _load(paths["signals"], SIGNAL_CACHE_SCHEMA, signal_digest)
    signal_hit = signal_cache is not None
    if signal_cache is None:
        output = dict(backend.infer(video_path))
        signals = _validated_signals(output)
        signal_cache = {
            "schema_version": SIGNAL_CACHE_SCHEMA,
            "input_digest": signal_digest,
            "skill_id": SKILL_ID,
            "backend_output_schema": output.get("schema_version"),
            "status": output.get("status") or ("ok" if signals else "no_signal"),
            "signals": output.get("signals"),
            "sampling": output.get("sampling") or {},
            "raw_output": output.get("raw_output") or {},
            "provenance": {"backend": identity, **dict(output.get("provenance") or {})},
        }
        signal_cache["evidence_digest"] = value_digest(signal_cache)
        atomic_write_json(paths["signals"], signal_cache)
    signals = _validated_signals(
        {
            "schema_version": signal_cache.get("backend_output_schema"),
            "signals": signal_cache.get("signals"),
            "status": signal_cache.get("status"),
        }
    )
    result_digest = value_digest(
        {
            **common,
            "schema": RESULT_CACHE_SCHEMA,
            "selected_laws": selected,
            "signal_evidence_digest": signal_cache.get("evidence_digest"),
        }
    )
    cached = _load(paths["result"], RESULT_CACHE_SCHEMA, result_digest)
    if cached is not None:
        return _response(paths, cached, True, True)

    if not signals:
        result = _result("no_signal", None, [], spec)
    else:
        scored = score_signals(model_id, sample_id, signals, spec.get("gt") or {})
        result = _result(scored["status"], scored["law_score"], scored["laws"], spec)
    payload = {
        "schema_version": RESULT_CACHE_SCHEMA,
        "input_digest": result_digest,
        "skill_id": SKILL_ID,
        "definition_version": DEFINITION_VERSION,
        "result": result,
        "provenance": {
            "backend": identity,
            "video": video,
            "signal_cache": str(paths["signals"]),
            "signal_cache_hit": signal_hit,
            "signal_evidence_digest": signal_cache.get("evidence_digest"),
        },
    }
    atomic_write_json(paths["result"], payload)
    return _response(paths, payload, False, True)
