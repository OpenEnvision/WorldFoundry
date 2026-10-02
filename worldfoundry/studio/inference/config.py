"""Inference process topology and runtime environment configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping

LINGBOT_WORLD_MODEL_ID = "lingbot-world"
LINGBOT_WORLD_V2_MODEL_ID = "lingbot-world-v2"
MATRIX_GAME3_MODEL_ID = "matrix-game-3"
LONGVIE2_MODEL_ID = "longvie-2"
HELIOS_MODEL_ID = "helios"
DREAMX_WORLD_MODEL_ID = "dreamx-world-5b-cam"
LINGBOT_VARIANT_FAST = "fast"
LINGBOT_FAST_NUM_PROCS_ENV_KEYS = (
    "WORLDFOUNDRY_REALTIME_NPROC_PER_NODE",
    "WORLDFOUNDRY_REALTIME_NPROC",
    "WORLDFOUNDRY_STUDIO_LINGBOT_TORCHRUN_NPROC",
    "WM_LINGBOTWORLDFAST_NUM_PROCS",
)
LINGBOT_FAST_USE_SP_ENV = "WORLDFOUNDRY_LINGBOT_FAST_USE_SP"
def _torchrun_world_size() -> int:
    try:
        return max(int(os.getenv("WORLD_SIZE", "1") or "1"), 1)
    except Exception:
        return 1


def _cuda_visible_device_count(value: str | None = None) -> int:
    text = os.getenv("CUDA_VISIBLE_DEVICES", "") if value is None else str(value or "")
    text = text.strip()
    if not text:
        return 0
    return len([item for item in text.split(",") if item.strip()])


def resolve_lingbot_fast_num_procs(*, visible_count: int | None = None) -> int:
    """Resolve LingBot-World-Fast torchrun process count using WMFactory semantics."""

    if visible_count is None:
        visible_count = _cuda_visible_device_count()
        if visible_count <= 0:
            try:
                import torch

                visible_count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
            except Exception:
                visible_count = 0
    visible_count = max(int(visible_count or 0), 0)
    # The public LingBot 14B recipe is eight-way FSDP + Ulysses.  Four ranks
    # are the compact supported topology because 40 attention heads divide
    # evenly; prefer all eight ranks when the host exposes them.
    default = 8 if visible_count >= 8 else (4 if visible_count >= 4 else (2 if visible_count >= 2 else 1))
    for key in LINGBOT_FAST_NUM_PROCS_ENV_KEYS:
        raw = os.getenv(key, "").strip()
        if not raw:
            continue
        try:
            nproc = int(raw)
        except ValueError as exc:
            raise ValueError(f"{key} must be an integer, got {raw!r}.") from exc
        if nproc < 1:
            raise ValueError(f"{key} must be >= 1.")
        if visible_count and nproc > visible_count:
            raise ValueError(f"{key}={nproc} exceeds visible CUDA devices ({visible_count}).")
        return nproc
    return default


def lingbot_fast_sequence_parallel_enabled(*, world_size: int | None = None) -> bool:
    """Return whether LingBot-World-Fast should use sequence parallel inference."""

    raw = os.getenv(LINGBOT_FAST_USE_SP_ENV, "").strip().lower()
    if raw:
        return raw in {"1", "true", "yes", "on"}
    size = _torchrun_world_size() if world_size is None else max(int(world_size or 1), 1)
    return size > 1


@dataclass(frozen=True)
class WMFactoryInteractiveModelSpec:
    model_id: str
    wmfactory_model_id: str
    env_prefix: str
    preferred_visible_devices: int | None = None
    use_dual_device_hint: bool = False
    supported_process_counts: tuple[int, ...] = ()


WMFACTORY_INTERACTIVE_MODEL_SPECS: tuple[WMFactoryInteractiveModelSpec, ...] = (
    WMFactoryInteractiveModelSpec("matrix-game-2", "matrixgame", "MATRIXGAME"),
    WMFactoryInteractiveModelSpec(
        MATRIX_GAME3_MODEL_ID,
        "matrixgame3",
        "MATRIXGAME3",
        preferred_visible_devices=4,
        supported_process_counts=(1, 2, 4, 8),
    ),
    WMFactoryInteractiveModelSpec("yume", "yume", "YUME"),
    WMFactoryInteractiveModelSpec("yume-1p5", "yume", "YUME"),
    WMFactoryInteractiveModelSpec("diamond", "diamond", "DIAMOND"),
    WMFactoryInteractiveModelSpec("oasis-500m", "open-oasis", "OPENOASIS"),
    WMFactoryInteractiveModelSpec("vid2world", "vid2world", "VID2WORLD"),
    WMFactoryInteractiveModelSpec(
        "infinite-world",
        "infinite-world",
        "INFINITEWORLD",
        preferred_visible_devices=2,
        use_dual_device_hint=True,
    ),
    WMFactoryInteractiveModelSpec(
        "hunyuan-worldplay",
        "worldplay",
        "WORLDPLAY",
        preferred_visible_devices=8,
        supported_process_counts=(1, 2, 3, 4, 6, 8),
    ),
    WMFactoryInteractiveModelSpec("mineworld", "mineworld", "MINEWORLD"),
    WMFactoryInteractiveModelSpec(
        "lingbot-world",
        "lingbot-world-fast",
        "LINGBOTWORLDFAST",
        preferred_visible_devices=8,
        supported_process_counts=(1, 4, 8),
    ),
    WMFactoryInteractiveModelSpec(
        "lingbot-world-v2",
        "lingbot-world-v2",
        "LINGBOTWORLDV2",
        preferred_visible_devices=8,
        supported_process_counts=(1, 4, 8),
    ),
)

_WMFACTORY_SPEC_BY_MODEL_ID = {spec.model_id: spec for spec in WMFACTORY_INTERACTIVE_MODEL_SPECS}
_WMFACTORY_MODEL_ALIASES = {
    "matrixgame": "matrix-game-2",
    "matrixgame2": "matrix-game-2",
    "matrix-game2": "matrix-game-2",
    "matrixgame3": "matrix-game-3",
    "matrix-game3": "matrix-game-3",
    "yume-1.5": "yume-1p5",
    "yume1.5": "yume-1p5",
    "open-oasis": "oasis-500m",
    "openoasis": "oasis-500m",
    "open_oasis": "oasis-500m",
    "infiniteworld": "infinite-world",
    "infinite_world": "infinite-world",
    "worldplay": "hunyuan-worldplay",
    "hy-worldplay": "hunyuan-worldplay",
    "hyworldplay": "hunyuan-worldplay",
    "lingbot-world-fast": "lingbot-world",
    "lingbotworld-fast": "lingbot-world",
    "lingbotworldfast": "lingbot-world",
}


def wmfactory_interactive_model_spec(
    model_id: str,
    *,
    load_kwargs: Mapping[str, Any] | None = None,
) -> WMFactoryInteractiveModelSpec | None:
    """Return the WMFactory compatibility spec for a WorldFoundry interactive model."""

    key = (model_id or "").strip().lower()
    key = _WMFACTORY_MODEL_ALIASES.get(key, key)
    spec = _WMFACTORY_SPEC_BY_MODEL_ID.get(key)
    if spec is None:
        return None
    if spec.model_id == LINGBOT_WORLD_MODEL_ID:
        runtime_variant = str((load_kwargs or {}).get("runtime_variant") or "").strip().lower()
        if runtime_variant and runtime_variant != LINGBOT_VARIANT_FAST:
            return None
    return spec


def env_first(*names: str) -> str:
    """Return the first non-empty environment value from ordered keys.

    Args:
        names: Environment variable names in priority order.
    """
    for name in names:
        value = os.getenv(name)
        if value is not None and value != "":
            return value
    return ""
