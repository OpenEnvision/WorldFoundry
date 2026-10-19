"""Studio catalog ordering, filtering, and aggregate stats."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.evaluation.utils import REPO_ROOT
from worldfoundry.runtime.inference_catalog import ASSET_GATED_WORLD_RUNTIME_MODEL_IDS
from worldfoundry.studio.inference.catalog import LIVE_CONTROL_TAGS, CatalogEntry, discover_catalog, template_id_hint

CATALOG_RUNTIME_COMPLETENESS_CHECKS_ENV = "WORLDFOUNDRY_STUDIO_CHECK_RUNTIME_COMPLETENESS"


LIVE_CONTROL_TEMPLATE_IDS = frozenset({"interactive-world"})
NON_LIVE_CONTROL_TEMPLATE_IDS = frozenset(
    {
        "embodied-policy",
        "visual-action",
        "depth-geometry",
        "hosted-api",
        "conditioned-video",
        "video-to-video",
        "text-video",
    }
)


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on", "strict", "preflight"}


def _catalog_runtime_completeness_checks_enabled() -> bool:
    return _env_flag(CATALOG_RUNTIME_COMPLETENESS_CHECKS_ENV)


@lru_cache(maxsize=1)
def _three_d_four_d_runtime_spec_tables() -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    from worldfoundry.base_models.three_dimensions.three_d_four_d.runtime import (
        IN_TREE_RUNTIME_DIRS,
        THREE_D_FOUR_D_RUNTIME_SPECS,
    )

    return IN_TREE_RUNTIME_DIRS, THREE_D_FOUR_D_RUNTIME_SPECS


def _has_complete_in_tree_runtime(entry: CatalogEntry) -> bool:
    """Return False for catalog rows that would only create placeholder jobs."""
    task_type = entry.default_task_type.strip().lower().replace("_", "-")
    if task_type == "unsupported-inference":
        return False
    if bool(entry.default_call_kwargs.get("plan_only")):
        return False
    if not _catalog_runtime_completeness_checks_enabled():
        return True

    in_tree_runtime_dirs, three_d_four_d_runtime_specs = _three_d_four_d_runtime_spec_tables()
    spec = three_d_four_d_runtime_specs.get(entry.model_id.lower().replace("_", "-"))
    if spec is None:
        return True
    if spec.command_kind == "unsupported_inference":
        return False
    if not spec.entrypoint:
        return False

    in_tree_parts = in_tree_runtime_dirs.get(spec.model_id)
    if not in_tree_parts:
        return True
    return (REPO_ROOT / "worldfoundry" / Path(*in_tree_parts) / spec.entrypoint).is_file()


def _supports_live_controls(entry: CatalogEntry) -> bool:
    """Return True when the entry exposes streamable token-driven live controls.

    Args:
        entry: Catalog row describing the runnable pipeline.
    """
    template_id = template_id_hint(entry)
    if template_id in NON_LIVE_CONTROL_TEMPLATE_IDS:
        return False
    return (
        entry.supports_stream
        and template_id in LIVE_CONTROL_TEMPLATE_IDS
        and (
            entry.runtime_kind in {"two_stage_3dgs", "pointcloud_nav", "worldfm"}
            or bool(entry.default_interactions)
            or any(tag in LIVE_CONTROL_TAGS for tag in entry.tags)
        )
    )


def _studio_catalog(entries: Sequence[CatalogEntry] | None = None) -> tuple[CatalogEntry, ...]:
    """Return Studio pipelines sorted for the UI sidebar.

    Prefers entries that support live controls and streaming, then category order.

    Args:
        entries: Optional pre-built catalog rows; defaults to ``discover_catalog()``.
    """
    active_entries = discover_catalog() if entries is None else entries
    studio_entries = [
        entry
        for entry in active_entries
        if entry.model_id not in ASSET_GATED_WORLD_RUNTIME_MODEL_IDS and _has_complete_in_tree_runtime(entry)
    ]
    category_order = {
        "Video Generation": 0,
        "Video-to-Video": 1,
        "3D Scene": 2,
        "Depth / Geometry": 3,
        "Visual Action": 4,
        "Embodied Action": 5,
        "Remote API": 6,
    }
    studio_entries.sort(
        key=lambda entry: (
            0 if _supports_live_controls(entry) else 1,
            0 if entry.supports_stream else 1,
            category_order.get(entry.category, 99),
            entry.display_name.lower(),
            entry.model_id,
        )
    )
    return tuple(studio_entries)
