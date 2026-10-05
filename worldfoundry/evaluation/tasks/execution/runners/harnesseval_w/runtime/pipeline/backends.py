"""Construct HarnessEval skill backends from one explicit JSON configuration."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from ..protocols import SKILLS
from ..skill_backend import appearance_consistency
from ..skill_backend import drift_degradation
from ..skill_backend import intentional_change_vlm
from ..skill_backend import motion_quality
from ..skill_backend import offscreen_evolution_vlm
from ..skill_backend import physical_law
from ..skill_backend import physical_plausibility
from ..skill_backend import physical_response_vlm
from ..skill_backend import render_quality
from ..skill_backend import return_consistency
from ..skill_backend import viewpoint_trajectory


OBSERVATION_MODULES = {
    "render_quality_inspector": render_quality,
    "motion_quality_inspector": motion_quality,
    "appearance_consistency_inspector": appearance_consistency,
    "physical_plausibility_inspector": physical_plausibility,
    "viewpoint_trajectory_verifier": viewpoint_trajectory,
}
VLM_MODULES = {
    "intentional_change_verifier_vlm": intentional_change_vlm,
    "physical_response_verifier_vlm": physical_response_vlm,
    "offscreen_evolution_verifier": offscreen_evolution_vlm,
}


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"backend config {label} must be an object")
    return dict(value)


def _skill_config(config: Mapping[str, Any], skill_id: str) -> dict[str, Any]:
    defaults = _mapping(config.get("defaults"), "defaults")
    skills = _mapping(config.get("skills"), "skills")
    if skill_id not in skills:
        raise ValueError(f"backend config is missing skill {skill_id}")
    return {**defaults, **_mapping(skills[skill_id], f"skills.{skill_id}")}


def _required(value: Mapping[str, Any], key: str, label: str) -> Any:
    result = value.get(key)
    if result is None or (isinstance(result, str) and not result.strip()):
        raise ValueError(f"backend config {label} requires {key}")
    return result


def _path(
    value: Mapping[str, Any], key: str, label: str, config_root: Path
) -> Path:
    path = Path(str(_required(value, key, label))).expanduser()
    return (path if path.is_absolute() else config_root / path).resolve()


def _optional_path(
    value: Mapping[str, Any], key: str, config_root: Path
) -> Path | None:
    raw = value.get(key)
    if raw is None:
        return None
    path = Path(str(raw)).expanduser()
    return (path if path.is_absolute() else config_root / path).resolve()


def _build_observation(
    skill_id: str, values: Mapping[str, Any], config_root: Path
) -> Any:
    module = OBSERVATION_MODULES[skill_id]
    label = f"skills.{skill_id}"
    mode = str(_required(values, "mode", label))
    if mode != "local":
        raise ValueError(f"backend config {label}.mode must be local")

    if skill_id == "viewpoint_trajectory_verifier":
        return module.LocalBackend(
            weights_root=_path(values, "weights_root", label, config_root),
            device=str(values.get("device", "cuda")),
            frame_batch_size=int(values.get("frame_batch_size", 1)),
        )
    common = {
        "weights_root": _path(values, "weights_root", label, config_root),
        "device": str(values.get("device", "cuda")),
    }
    if skill_id == "physical_plausibility_inspector":
        common["model_path"] = _path(values, "model_path", label, config_root)
    if skill_id in {"render_quality_inspector", "motion_quality_inspector"}:
        common["cpu_workers"] = int(values.get("cpu_workers", 8))
    return module.LocalBackend(**common)


def _api_key(values: Mapping[str, Any], config_root: Path) -> str | None:
    environment_name = values.get("api_key_env")
    if environment_name:
        return os.environ.get(str(environment_name)) or None
    path = _optional_path(values, "api_key_file", config_root)
    if path is None or not path.is_file():
        return None
    key = path.read_text(encoding="utf-8").strip()
    return key or None


def _build_vlm(skill_id: str, values: Mapping[str, Any], config_root: Path) -> Any:
    label = f"skills.{skill_id}"
    if str(_required(values, "mode", label)) != "api":
        raise ValueError(f"backend config {label}.mode must be api")
    return VLM_MODULES[skill_id].OpenAICompatibleBackend(
        base_url=str(_required(values, "base_url", label)),
        model=str(_required(values, "model", label)),
        api_key=_api_key(values, config_root),
        wire_api=str(values.get("wire_api", "responses")),
        timeout=float(values.get("timeout", 90.0)),
        retries=int(values.get("retries", 1)),
    )


def _build_clip(config: Mapping[str, Any], config_root: Path) -> Any:
    values = _mapping(config.get("clip"), "clip")
    mode = str(_required(values, "mode", "clip"))
    if mode == "local":
        return return_consistency.LocalClipEmbedder(
            device=str(values.get("device", "cuda")),
            download_root=_optional_path(values, "download_root", config_root),
        )
    raise ValueError("backend config clip.mode must be local")


def build_backends(
    skill_ids: Iterable[str],
    config: Mapping[str, Any],
    *,
    cache_root: Path,
    config_root: Path,
) -> dict[str, Any]:
    """Build only requested backends; constructors must remain inference-lazy."""

    required = set(skill_ids)
    unknown = required - set(SKILLS)
    if unknown:
        raise ValueError("unknown HarnessEval skills: " + ", ".join(sorted(unknown)))
    cache_root = cache_root.resolve()
    config_root = config_root.resolve()
    built: dict[str, Any] = {}
    clip: Any | None = None

    def observation(skill_id: str) -> Any:
        if skill_id not in built:
            built[skill_id] = _build_observation(
                skill_id, _skill_config(config, skill_id), config_root
            )
        return built[skill_id]

    def clip_backend() -> Any:
        nonlocal clip
        if clip is None:
            clip = _build_clip(config, config_root)
        return clip

    for skill_id in SKILLS:
        if skill_id not in required:
            continue
        values = _skill_config(config, skill_id)
        if skill_id in OBSERVATION_MODULES:
            observation(skill_id)
        elif skill_id in VLM_MODULES:
            built[skill_id] = _build_vlm(skill_id, values, config_root)
        elif skill_id == "physical_law_validator":
            if values.get("mode") != "local":
                raise ValueError("physical_law_validator mode must be local")
            built[skill_id] = physical_law.LocalBackend()
        elif skill_id == "drift_degradation_analyzer":
            mode = str(_required(values, "mode", f"skills.{skill_id}"))
            backend_type = {
                "local": drift_degradation.LocalBackend,
                "staged_local": drift_degradation.LocalStagedBackend,
            }.get(mode)
            if backend_type is None:
                raise ValueError(
                    "drift_degradation_analyzer mode must be local or staged_local"
                )
            embedder = clip_backend()
            children = (
                observation("render_quality_inspector"),
                observation("motion_quality_inspector"),
                observation("physical_plausibility_inspector"),
                embedder,
            )
            if mode == "staged_local":
                built[skill_id] = backend_type(*children, cache_root=cache_root)
            else:
                built[skill_id] = backend_type(*children)
        elif skill_id == "return_consistency_verifier":
            embedder = clip_backend()
            mode = str(values.get("mode", "local"))
            if mode != "local":
                raise ValueError("return_consistency_verifier mode must be local")
            built[skill_id] = return_consistency.LocalBackend(embedder)
        else:  # pragma: no cover - SKILLS and the factory are checked together
            raise AssertionError(skill_id)
    return {skill_id: built[skill_id] for skill_id in SKILLS if skill_id in required}
