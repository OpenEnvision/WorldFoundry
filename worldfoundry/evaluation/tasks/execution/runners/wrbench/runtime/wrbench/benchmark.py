"""Benchmark task expansion helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wrbench.datasets import (
    PROMPT_PROFILE_TI2V_ACTIVE,
    PROMPT_PROFILE_T2V_LAYOUT_ANCHOR,
    load_jsonl,
    load_natural25_t2v_event_tails,
    load_natural25_t2v_layout_anchors,
    natural25_variants_path,
    resolve_variant_prompt,
    resolve_variant_prompt_profile,
)

NATURAL25_CAMERA_COMBOS: tuple[str, ...] = (
    "static",
    "yaw_LR",
    "pan_LR",
    "yaw_RL",
    "pan_RL",
)


@dataclass(frozen=True)
class Natural25CameraSpec:
    """Configured camera entry for a Natural-25 benchmark scope."""

    label: str
    preset: str
    camera_type: str
    stress_axis: str
    stress_yaw_deg: float | None


@dataclass(frozen=True)
class Natural25CameraScope:
    """Config-backed Natural-25 camera scope."""

    scope_id: str
    variant_oov_gap: str
    expected_task_count: int
    cameras: tuple[Natural25CameraSpec, ...]


@dataclass(frozen=True)
class Natural25CameraTask:
    """One Natural-25 semantic variant paired with one camera combo."""

    output_id: str
    variant_id: str
    family_id: str
    reasoning_tier: str
    divergence_id: str | None
    camera: str
    preset: str
    camera_type: str
    stress_axis: str
    stress_yaw_deg: float | None
    camera_scope_id: str
    oov_gap: str
    event_delta: str
    world_state_prompt: str
    expected_state: str
    prompt_profile_id: str
    ti2v_prompt: str
    prompt: str


def _scope_camera_spec(row: dict[str, Any], *, index: int) -> Natural25CameraSpec:
    expected_keys = {"label", "preset", "camera_type", "stress_axis", "stress_yaw_deg"}
    keys = set(row)
    if keys != expected_keys:
        raise ValueError(f"camera scope entry {index} keys must be {sorted(expected_keys)}, got {sorted(keys)}")
    label = str(row["label"])
    preset = str(row["preset"])
    camera_type = str(row["camera_type"])
    stress_axis = str(row["stress_axis"])
    raw_yaw = row["stress_yaw_deg"]
    stress_yaw_deg = None if raw_yaw is None else float(raw_yaw)
    if not label:
        raise ValueError(f"camera scope entry {index} has empty label")
    if preset not in NATURAL25_CAMERA_COMBOS:
        raise ValueError(f"camera scope entry {index} has unsupported preset {preset!r}")
    if camera_type not in NATURAL25_CAMERA_COMBOS:
        raise ValueError(f"camera scope entry {index} has unsupported camera_type {camera_type!r}")
    if stress_axis == "yaw":
        if camera_type not in {"yaw_LR", "yaw_RL"} or stress_yaw_deg is None:
            raise ValueError(f"camera scope entry {index} yaw stress requires yaw camera_type and stress_yaw_deg")
    elif stress_axis == "static":
        if camera_type != "static" or stress_yaw_deg is not None:
            raise ValueError(f"camera scope entry {index} static stress requires static camera_type and null stress_yaw_deg")
    else:
        raise ValueError(f"camera scope entry {index} has unsupported stress_axis {stress_axis!r}")
    return Natural25CameraSpec(
        label=label,
        preset=preset,
        camera_type=camera_type,
        stress_axis=stress_axis,
        stress_yaw_deg=stress_yaw_deg,
    )


def load_natural25_camera_scope(path: str | Path) -> Natural25CameraScope:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    expected_keys = {"schema_version", "scope_id", "variant_filter", "expected_task_count", "cameras"}
    keys = set(payload)
    if keys != expected_keys:
        raise ValueError(f"camera scope keys must be {sorted(expected_keys)}, got {sorted(keys)}")
    if int(payload["schema_version"]) != 1:
        raise ValueError(f"unsupported camera scope schema_version {payload['schema_version']!r}")
    variant_filter = payload["variant_filter"]
    if set(variant_filter) != {"oov_gap"}:
        raise ValueError("camera scope variant_filter must contain exactly oov_gap")
    cameras = tuple(
        _scope_camera_spec(row, index=index)
        for index, row in enumerate(payload["cameras"])
    )
    labels = [camera.label for camera in cameras]
    if len(labels) != len(set(labels)):
        raise ValueError(f"camera scope has duplicate labels: {labels}")
    return Natural25CameraScope(
        scope_id=str(payload["scope_id"]),
        variant_oov_gap=str(variant_filter["oov_gap"]),
        expected_task_count=int(payload["expected_task_count"]),
        cameras=cameras,
    )


def _legacy_camera_spec(camera: str) -> Natural25CameraSpec:
    if camera not in NATURAL25_CAMERA_COMBOS:
        raise ValueError(f"unknown Natural-25 camera combo: {camera!r}")
    return Natural25CameraSpec(
        label=str(camera),
        preset=str(camera),
        camera_type=str(camera),
        stress_axis="legacy",
        stress_yaw_deg=None,
    )


def _natural25_camera_tasks_for_specs(
    *,
    specs: tuple[Natural25CameraSpec, ...],
    scope_id: str,
    variant_oov_gap: str,
    variants_path: str | Path | None,
    prompt_profile: str,
) -> list[Natural25CameraTask]:
    path = Path(variants_path) if variants_path is not None else natural25_variants_path()
    prompt_profile_id = resolve_variant_prompt_profile(prompt_profile)
    layout_anchors = (
        load_natural25_t2v_layout_anchors()
        if prompt_profile_id == PROMPT_PROFILE_T2V_LAYOUT_ANCHOR
        else None
    )
    t2v_event_tails = (
        load_natural25_t2v_event_tails()
        if prompt_profile_id == PROMPT_PROFILE_T2V_LAYOUT_ANCHOR
        else None
    )
    tasks: list[Natural25CameraTask] = []
    for row in load_jsonl(path):
        if row["oov_gap"] != variant_oov_gap:
            continue
        variant_id = str(row["variant_id"])
        ti2v_prompt = str(row.get("ti2v_prompt") or "").strip()
        prompt = resolve_variant_prompt(
            row,
            prompt_profile=prompt_profile_id,
            layout_anchors=layout_anchors,
            t2v_event_tails=t2v_event_tails,
        )
        world_state_prompt = str(row.get("world_state_prompt") or "").strip()
        expected_state = str(row.get("expected_state") or "").strip()
        if not ti2v_prompt:
            raise ValueError(f"variant {variant_id!r} missing ti2v_prompt")
        if not world_state_prompt:
            raise ValueError(f"variant {variant_id!r} missing world_state_prompt")
        if not expected_state:
            raise ValueError(f"variant {variant_id!r} missing expected_state")
        for spec in specs:
            tasks.append(
                Natural25CameraTask(
                    output_id=f"{variant_id}__{spec.label}",
                    variant_id=variant_id,
                    family_id=str(row["family_id"]),
                    reasoning_tier=str(row["reasoning_tier"]),
                    divergence_id=row.get("divergence_id"),
                    camera=spec.label,
                    preset=spec.preset,
                    camera_type=spec.camera_type,
                    stress_axis=spec.stress_axis,
                    stress_yaw_deg=spec.stress_yaw_deg,
                    camera_scope_id=scope_id,
                    oov_gap=str(row["oov_gap"]),
                    event_delta=str(row.get("event_delta") or ""),
                    world_state_prompt=world_state_prompt,
                    expected_state=expected_state,
                    prompt_profile_id=prompt_profile_id,
                    ti2v_prompt=ti2v_prompt,
                    prompt=prompt,
                )
            )
    return tasks


def natural25_camera_tasks_from_scope(
    *,
    camera_scope: Natural25CameraScope,
    variants_path: str | Path | None = None,
    prompt_profile: str = PROMPT_PROFILE_TI2V_ACTIVE,
) -> list[Natural25CameraTask]:
    tasks = _natural25_camera_tasks_for_specs(
        specs=camera_scope.cameras,
        scope_id=camera_scope.scope_id,
        variant_oov_gap=camera_scope.variant_oov_gap,
        variants_path=variants_path,
        prompt_profile=prompt_profile,
    )
    if len(tasks) != camera_scope.expected_task_count:
        raise ValueError(
            f"camera scope {camera_scope.scope_id!r} expected {camera_scope.expected_task_count} tasks, got {len(tasks)}"
        )
    return tasks


def natural25_camera_tasks(
    *,
    variants_path: str | Path | None = None,
    cameras: tuple[str, ...] | list[str] | None = None,
    prompt_profile: str = PROMPT_PROFILE_TI2V_ACTIVE,
) -> list[Natural25CameraTask]:
    """Expand the Natural-25 camera benchmark into per-video tasks.

    The bundled variant file contains multiple ``oov_gap`` prompt forms. Camera
    benchmark generation uses the 100 ``oov_gap == "none"`` semantic variants
    and expands each into the five canonical camera controls, producing 500
    videos for models that support all controls.
    """

    selected_cameras = tuple(cameras) if cameras is not None else NATURAL25_CAMERA_COMBOS
    unknown = sorted(set(selected_cameras) - set(NATURAL25_CAMERA_COMBOS))
    if unknown:
        raise ValueError(f"unknown Natural-25 camera combo(s): {unknown}")
    return _natural25_camera_tasks_for_specs(
        specs=tuple(_legacy_camera_spec(camera) for camera in selected_cameras),
        scope_id="natural25_legacy_camera_combos",
        variant_oov_gap="none",
        variants_path=variants_path,
        prompt_profile=prompt_profile,
    )
