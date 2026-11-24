"""Studio model catalog: pipeline discovery, curated per-model overrides, and lookup.

Layout (lower layers never import higher ones):

- ``entry``: the ``CatalogEntry`` data model.
- ``resolve``: checkpoint, local-path, and interpreter resolution shared by all defaults.
- ``params``: parameter templates and variant folding shared across model families.
- ``models``: curated overrides and model-specific defaults, one module per family.
- ``discovery``: AST pipeline discovery, entry building, and ``find_entry`` lookup.
"""

from __future__ import annotations

from .discovery import (
    LIVE_CONTROL_TAGS,
    STUDIO_HIDDEN_CATALOG_MODEL_IDS,
    discover_catalog,
    find_entry,
    find_runtime_entry,
    template_id_hint,
)
from .entry import CatalogEntry
from .models import CURATED_OVERRIDES
from .models.video_generation import SANA_DEFAULT_IMAGE_VARIANT_ID
from .models.world_models import lingbot_world_fast_load_kwargs
from .params import COGVIDEOX_DEFAULT_VARIANT_ID, COGVIDEOX_STUDIO_PARENT_ID, cogvideox_runtime_model_id

__all__ = [
    "COGVIDEOX_DEFAULT_VARIANT_ID",
    "COGVIDEOX_STUDIO_PARENT_ID",
    "CURATED_OVERRIDES",
    "LIVE_CONTROL_TAGS",
    "SANA_DEFAULT_IMAGE_VARIANT_ID",
    "STUDIO_HIDDEN_CATALOG_MODEL_IDS",
    "CatalogEntry",
    "cogvideox_runtime_model_id",
    "discover_catalog",
    "find_entry",
    "find_runtime_entry",
    "lingbot_world_fast_load_kwargs",
    "template_id_hint",
]
