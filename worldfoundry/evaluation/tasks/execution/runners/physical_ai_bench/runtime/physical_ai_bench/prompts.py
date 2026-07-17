"""Load PAI-Bench judge prompts from the checked-in data asset tree."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import yaml

from worldfoundry.evaluation.tasks.execution.framework.benchmark_assets import bundled_benchmark_asset


@lru_cache(maxsize=None)
def prompt_config(name: str) -> dict[str, Any]:
    path = bundled_benchmark_asset("physical-ai-bench", "prompts", f"{name}.yaml")
    if not path.is_file():
        raise FileNotFoundError(f"PAI-Bench prompt config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("template"), str):
        raise ValueError(f"invalid PAI-Bench prompt config: {path}")
    return payload


def render_prompt(name: str, **values: Any) -> str:
    return str(prompt_config(name)["template"]).format(**values).strip()


__all__ = ["prompt_config", "render_prompt"]
