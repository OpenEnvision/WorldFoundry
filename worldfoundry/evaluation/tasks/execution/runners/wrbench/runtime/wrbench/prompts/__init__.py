"""Prompt generation for wrbench benchmarks."""

from wrbench.prompts.camera_text import (
    API_CAMERA_MOTIONS,
    CAMERA_CLAUSES,
    assemble_ti2v_prompt,
    build_api_prompt_preview_row,
    build_api_prompt_preview_rows,
    build_prompt_to_send,
    camera_clause,
    preset_camera_text,
)
from wrbench.prompts.scene import enrich_family_with_t2i_scene, generate_t2i_scene
from wrbench.prompts.task import (
    generate_ti2v_content_llm,
    generate_ti2v_variants_llm,
    generate_variants_deterministic,
    load_jsonl,
    write_jsonl,
)

__all__ = [
    "API_CAMERA_MOTIONS",
    "CAMERA_CLAUSES",
    "assemble_ti2v_prompt",
    "build_api_prompt_preview_row",
    "build_api_prompt_preview_rows",
    "build_prompt_to_send",
    "camera_clause",
    "enrich_family_with_t2i_scene",
    "generate_t2i_scene",
    "generate_ti2v_content_llm",
    "generate_ti2v_variants_llm",
    "generate_variants_deterministic",
    "load_jsonl",
    "preset_camera_text",
    "write_jsonl",
]
