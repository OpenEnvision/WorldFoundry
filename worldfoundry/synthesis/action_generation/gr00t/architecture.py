"""GR00T checkpoint architecture inspection."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def load_checkpoint_architecture(checkpoint_dir: str | Path) -> dict[str, Any]:
    """Read GR00T architecture metadata from a local Hugging Face checkpoint.

    Args:
        checkpoint_dir: Directory containing the GR00T config.json and processor_config.json files.
    """
    root = Path(checkpoint_dir).expanduser().resolve()
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    processor = json.loads((root / "processor_config.json").read_text(encoding="utf-8"))
    modality_configs = dict(processor.get("processor_kwargs", {}).get("modality_configs", {}))
    return {
        "architectures": config.get("architectures", []),
        "model_type": config.get("model_type"),
        "model_name": config.get("model_name"),
        "processor_class": processor.get("processor_class"),
        "embodiments": sorted(str(key) for key in modality_configs),
        "action_horizon": config.get("action_horizon"),
        "max_action_dim": config.get("max_action_dim"),
        "max_state_dim": config.get("max_state_dim"),
        "num_inference_timesteps": config.get("num_inference_timesteps"),
        "diffusion_model": _compact_mapping(config.get("diffusion_model_cfg", {})),
    }


def load_embodiment_ids(checkpoint_dir: str | Path) -> dict[str, int]:
    """Read the checkpoint embodiment id mapping.

    Args:
        checkpoint_dir: Directory containing embodiment_id.json.
    """
    root = Path(checkpoint_dir).expanduser().resolve()
    data = json.loads((root / "embodiment_id.json").read_text(encoding="utf-8"))
    return {str(key): int(value) for key, value in data.items()}


def _compact_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): item for key, item in value.items()}
