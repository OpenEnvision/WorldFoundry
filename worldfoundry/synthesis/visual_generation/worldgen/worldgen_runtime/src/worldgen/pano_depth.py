import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from worldfoundry.base_models.three_dimensions.depth.depth_anything.runtime_v2 import (
    DepthAnything2Representation,
)

from .utils.general_utils import pano_unit_rays

MAX_DISTANCE = 20.0  # Scale the relative depth prior so max distance = 20 meters

def _depth_anything2_checkpoint_name(encoder: str) -> str:
    return f"depth_anything_v2_{encoder}.pth"


def _has_depth_anything2_checkpoint(path: str | Path, encoder: str) -> bool:
    candidate = Path(path).expanduser()
    checkpoint_name = _depth_anything2_checkpoint_name(encoder)
    if candidate.is_file():
        return candidate.name == checkpoint_name or candidate.name.startswith("depth_anything_v2_")
    return (candidate / checkpoint_name).is_file() or (candidate / "checkpoints" / checkpoint_name).is_file()


def _resolve_depth_anything2_encoder(encoder: str | None = None) -> str:
    return (
        encoder
        or os.environ.get("WORLDFOUNDRY_DEPTH_ANYTHING2_ENCODER")
        or os.environ.get("WORLDFOUNDRY_DA2_ENCODER")
        or "vitl"
    )


def _resolve_depth_anything2_model_path(
    model_path: str | None = None,
    encoder: str = "vitl",
) -> str | None:
    if model_path:
        return model_path

    for env_name in ("WORLDFOUNDRY_DEPTH_ANYTHING2_MODEL_PATH", "WORLDFOUNDRY_DA2_MODEL_PATH"):
        env_value = os.environ.get(env_name)
        if env_value and _has_depth_anything2_checkpoint(env_value, encoder):
            return env_value

    ckpt_root_value = os.environ.get("WORLDFOUNDRY_CKPT_DIR")
    if not ckpt_root_value:
        return None

    ckpt_root = Path(ckpt_root_value).expanduser()
    candidate_names_by_encoder = {
        "vits": ("Depth-Anything-V2-Small", "Prior-Depth-Anything"),
        "vitb": ("Depth-Anything-V2-Base", "Prior-Depth-Anything"),
        "vitl": ("Depth-Anything-V2-Large", "Prior-Depth-Anything"),
        "vitg": ("Depth-Anything-V2-Giant", "Prior-Depth-Anything"),
    }
    for candidate_name in candidate_names_by_encoder.get(encoder, ()):
        candidate = ckpt_root / candidate_name
        if (candidate / _depth_anything2_checkpoint_name(encoder)).is_file():
            return str(candidate)
    return None


def build_depth_model(
    device: torch.device | str = "cuda",
    model_path: str | None = None,
    encoder: str | None = None,
    input_size: int = 518,
) -> DepthAnything2Representation:
    """Build WorldGen's depth prior through WorldFoundry' in-tree DA2 integration."""
    resolved_encoder = _resolve_depth_anything2_encoder(encoder)
    resolved_model_path = _resolve_depth_anything2_model_path(
        model_path=model_path,
        encoder=resolved_encoder,
    )
    return DepthAnything2Representation.from_pretrained(
        pretrained_model_path=resolved_model_path,
        encoder=resolved_encoder,
        device=str(device),
        default_input_size=input_size,
    )


def _predict_depth(model: DepthAnything2Representation, image: Image.Image):
    depth = model.get_representation(
        {
            "image": image,
            "color_order": "rgb",
            "input_size": model.default_input_size,
        }
    )["depth"]

    distance = depth.float()
    max_distance = distance.max().clamp_min(1e-6)
    distance = distance / max_distance * MAX_DISTANCE
    h, w = distance.shape
    rays = pano_unit_rays(h, w, model.device)  # (H, W, 3)

    rgb_out = torch.tensor(np.array(image.resize((w, h))), device=model.device)

    results = {
        "rgb": rgb_out,       # (H, W, 3)
        "depth": distance,    # (H, W)
        "distance": distance, # (H, W)
        "rays": rays          # (H, W, 3)
    }

    return results


def pred_pano_depth(model: DepthAnything2Representation, image: Image.Image):
    return _predict_depth(model, image)


def pred_depth(model: DepthAnything2Representation, image: Image.Image):
    return _predict_depth(model, image)

if __name__ == "__main__":
    model = build_depth_model()
    image = Image.open("data/background/timeless_desert.png")
    predictions = pred_pano_depth(model, image)
    print(predictions)
