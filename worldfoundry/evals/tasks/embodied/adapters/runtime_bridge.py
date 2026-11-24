"""Shared helpers for bridging synthesis runtimes into embodied policy adapters."""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast


def normalize_model_id(model_id: str) -> str:
    return str(model_id or "").strip().lower().replace("_", "-")


def first_image(obs: Mapping[str, Any]) -> Any:
    images = cast(Mapping[str, Any], obs.get("images")) if isinstance(obs.get("images"), Mapping) else {}
    for source, key in (
        (images, "agentview"),
        (images, "primary"),
        (images, "front"),
        (images, "base_camera"),
        (images, "wrist"),
        (obs, "image"),
        (obs, "agentview"),
    ):
        if key in source and source[key] is not None:
            return source[key]
    return None


def load_pipeline_target(model_id: str) -> str | None:
    from worldfoundry.zoo.bindings import resolve_pipeline_route

    route = resolve_pipeline_route(model_id=normalize_model_id(model_id))
    if route is None:
        return None
    return str(route[0])


def load_synthesis_class(model_id: str) -> type[Any]:
    pipeline_target = load_pipeline_target(model_id)
    if not pipeline_target:
        raise ValueError(f"no pipeline binding found for model_id={model_id!r}")
    module_name, _, class_name = pipeline_target.partition(":")
    if not module_name or not class_name:
        raise ValueError(f"invalid pipeline target {pipeline_target!r} for model_id={model_id!r}")
    pipeline_cls = getattr(importlib.import_module(module_name), class_name)
    return pipeline_cls.SYNTHESIS_CLS


def extract_action_values(result: Mapping[str, Any]) -> Any:
    for key in ("actions", "action"):
        value = result.get(key)
        if value not in (None, ""):
            return value

    artifact_path = result.get("artifact_path") or result.get("path") or result.get("output_path")
    if artifact_path:
        path = Path(str(artifact_path))
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, Mapping):
                for key in ("actions", "action"):
                    value = payload.get(key)
                    if value not in (None, ""):
                        return value
    raise RuntimeError(f"policy result did not contain actions: {dict(result)}")


__all__ = [
    "extract_action_values",
    "first_image",
    "load_pipeline_target",
    "load_synthesis_class",
    "normalize_model_id",
]
