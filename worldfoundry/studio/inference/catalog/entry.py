"""Catalog data model: the resolved ``CatalogEntry`` and AST pipeline records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CatalogEntry:
    model_id: str
    display_name: str
    module_path: str
    class_name: str
    family: str
    category: str
    summary: str
    call_params: tuple[str, ...] = field(default_factory=tuple)
    input_params: tuple[str, ...] = field(default_factory=tuple)
    stream_params: tuple[str, ...] = field(default_factory=tuple)
    load_params: tuple[str, ...] = field(default_factory=tuple)
    supports_stream: bool = False
    supports_from_pretrained: bool = False
    supports_api_init: bool = False
    runtime_kind: str = "default"
    default_backend: str = "auto"
    default_model_ref: str = ""
    default_endpoint: str = ""
    default_prompt: str = ""
    default_input_path: str = ""
    default_task_type: str = ""
    default_interactions: tuple[str, ...] = field(default_factory=tuple)
    default_load_kwargs: dict[str, Any] = field(default_factory=dict)
    default_call_kwargs: dict[str, Any] = field(default_factory=dict)
    extra_variants: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    suggested_task_types: tuple[str, ...] = field(default_factory=tuple)
    aliases: tuple[str, ...] = field(default_factory=tuple)
    tags: tuple[str, ...] = field(default_factory=tuple)
    env_hints: tuple[str, ...] = field(default_factory=tuple)
    notes: str = ""


@dataclass(frozen=True)
class _AstPipelineInfo:
    model_id: str
    display_name: str
    module_path: str
    class_name: str
    family: str
    call_params: tuple[str, ...]
    stream_params: tuple[str, ...]
    load_params: tuple[str, ...]
    supports_stream: bool
    supports_from_pretrained: bool
    supports_api_init: bool
    base_names: tuple[str, ...]
