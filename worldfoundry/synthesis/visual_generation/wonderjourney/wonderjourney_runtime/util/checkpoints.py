from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def checkpoint_roots() -> tuple[Path, ...]:
    primary = Path(
        os.environ.get(
            "WORLDFOUNDRY_CKPT_DIR",
            Path(__file__).resolve().parents[7] / "ckpt",
        )
    ).expanduser()
    roots = [primary]
    alternate_name = {"ckpt": "ckpts", "ckpts": "ckpt"}.get(primary.name)
    if alternate_name:
        roots.append(primary.with_name(alternate_name))
    return tuple(roots)


def local_model_ref(default: str, *relative_paths: str) -> str:
    for root in checkpoint_roots():
        for relative_path in relative_paths:
            candidate = root / relative_path
            if candidate.exists():
                return str(candidate)
    return default


def require_existing_checkpoint(*relative_paths: str) -> str:
    checked: list[Path] = []
    for root in checkpoint_roots():
        for relative_path in relative_paths:
            candidate = root / relative_path
            checked.append(candidate)
            if candidate.is_file():
                return str(candidate)
    rendered = "\n  - ".join(str(path) for path in checked)
    raise FileNotFoundError(f"WonderJourney checkpoint is missing; checked:\n  - {rendered}")


def diffusers_fp16_load_kwargs(model_ref: str, *, subfolder: str | None = None) -> dict[str, Any]:
    model_path = Path(model_ref).expanduser()
    if not model_path.is_dir():
        return {"revision": "fp16"} if subfolder is None else {}

    if subfolder == "vae":
        required = (model_path / "vae" / "diffusion_pytorch_model.fp16.safetensors",)
    else:
        required = (
            model_path / "text_encoder" / "model.fp16.safetensors",
            model_path / "unet" / "diffusion_pytorch_model.fp16.safetensors",
            model_path / "vae" / "diffusion_pytorch_model.fp16.safetensors",
        )
    if all(path.is_file() for path in required):
        return {"variant": "fp16", "use_safetensors": True}
    return {}


def oneformer_local_load_kwargs(model_ref: str, *, processor: bool = False) -> dict[str, Any]:
    model_path = Path(model_ref).expanduser()
    if not model_path.is_dir():
        return {}
    kwargs: dict[str, Any] = {"local_files_only": True}
    if processor and (model_path / "coco_panoptic.json").is_file():
        kwargs["repo_path"] = str(model_path)
    return kwargs
