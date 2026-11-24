from pathlib import Path

import torch
import yaml
from easydict import EasyDict

WORLDFOUNDRY_DATA_ROOT = Path(__file__).resolve().parents[5] / "data"
LINGBOT_WORLD_RUNTIME_CONFIG_ROOT = WORLDFOUNDRY_DATA_ROOT / "models" / "runtime" / "configs" / "lingbot_world"


def _load_yaml_config(name: str) -> dict:
    path = LINGBOT_WORLD_RUNTIME_CONFIG_ROOT / name
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping config in {path}")
    return payload


def _torch_dtype(name: str):
    try:
        return getattr(torch, name)
    except AttributeError as exc:
        raise ValueError(f"Unsupported torch dtype in LingBot World YAML: {name}") from exc


wan_shared_cfg = EasyDict()
_shared_payload = _load_yaml_config("shared.yaml")
for key in ("t5_dtype", "param_dtype"):
    if key in _shared_payload:
        _shared_payload[key] = _torch_dtype(_shared_payload[key])
wan_shared_cfg.update(_shared_payload)
