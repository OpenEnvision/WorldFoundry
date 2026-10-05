"""In-process Visual Chronometer PhyFPS prediction runtime."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from worldfoundry.evaluation.tasks.execution.framework.runner_common import resolve_env_path, VIDEO_SUFFIXES
from worldfoundry.evaluation.tasks.execution.runners.phyfps_bench_gen.phyfps_metrics import VideoPhyFPSRecord

DEFAULT_CONFIG_REL = Path("inference/configs/config_fps.yaml")
DEFAULT_CKPT_REL = Path("inference/ckpts/vc_common_10_60fps.ckpt")
HF_REPO_ID = "xiangbog/Visual_Chronometer"
HF_CKPT_FILENAME = "vc_common_10_60fps.ckpt"

IN_TREE_VISUAL_CHRONOMETER_ROOT = Path(__file__).resolve().parent / "runtime" / "visual_chronometer"


@dataclass(frozen=True)
class VisualChronometerPaths:
    chronometer_root: Path
    config_path: Path
    ckpt_path: Path




def resolve_chronometer_root(explicit: Path | None = None) -> Path | None:
    for candidate in (
        explicit,
        resolve_env_path("WORLDFOUNDRY_VISUAL_CHRONOMETER_ROOT"),
        IN_TREE_VISUAL_CHRONOMETER_ROOT,
    ):
        if candidate is not None and candidate.is_dir():
            return candidate.expanduser().resolve()
    return None


def resolve_predictor_paths(
    *,
    chronometer_root: Path | None = None,
    config_path: Path | None = None,
    ckpt_path: Path | None = None,
) -> VisualChronometerPaths:
    root = chronometer_root or resolve_chronometer_root()
    if root is None:
        raise FileNotFoundError(
            "Visual Chronometer root is missing. Set WORLDFOUNDRY_VISUAL_CHRONOMETER_ROOT "
            "for full official PhyFPS prediction."
        )
    resolved_config = config_path or (root / DEFAULT_CONFIG_REL)
    resolved_ckpt = ckpt_path or (root / DEFAULT_CKPT_REL)
    if not resolved_config.is_file():
        raise FileNotFoundError(f"Visual Chronometer config not found: {resolved_config}")
    return VisualChronometerPaths(
        chronometer_root=root,
        config_path=resolved_config.resolve(),
        ckpt_path=resolved_ckpt.resolve(),
    )


def list_generated_videos(video_dir: Path) -> list[Path]:
    if not video_dir.is_dir():
        return []
    return sorted(
        path
        for path in video_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )


def ensure_inference_import_path(chronometer_root: Path) -> Path:
    inference_root = chronometer_root / "inference"
    if not inference_root.is_dir():
        raise FileNotFoundError(f"Visual Chronometer inference directory not found: {inference_root}")
    path_str = str(inference_root.resolve())
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
    return inference_root


def download_checkpoint_if_needed(ckpt_path: Path) -> Path:
    if ckpt_path.is_file():
        return ckpt_path
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "Visual Chronometer checkpoint is missing and huggingface_hub is not installed. "
            f"Install huggingface_hub or place the checkpoint at {ckpt_path}."
        ) from exc
    downloaded = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_CKPT_FILENAME,
        local_dir=str(ckpt_path.parent),
    )
    return Path(downloaded)


def load_predictor_model(
    *,
    paths: VisualChronometerPaths,
    device: str,
) -> Any:
    ensure_inference_import_path(paths.chronometer_root)
    import torch
    from omegaconf import OmegaConf
    from utils.common_utils import instantiate_from_config  # type: ignore[import-not-found]  # noqa: WPS433

    ckpt_path = download_checkpoint_if_needed(paths.ckpt_path)
    config = OmegaConf.load(paths.config_path)
    config.model.params.freeze_encoder = False
    if "ckpt_path" in config.model.params:
        config.model.params.ckpt_path = None
    model = instantiate_from_config(config.model)
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def predict_video_file(
    *,
    model: Any,
    video_path: Path,
    chronometer_root: Path,
    device: str,
    clip_length: int = 30,
    stride: int = 4,
    resolution: int = 216,
) -> VideoPhyFPSRecord:
    ensure_inference_import_path(chronometer_root)
    from predict import predict_video  # type: ignore[import-not-found]  # noqa: WPS433

    segment_results, avg_fps, _total_frames = predict_video(
        model,
        str(video_path),
        device,
        clip_length=clip_length,
        stride=stride,
        resolution=resolution,
    )
    segment_values = tuple(float(item["predicted_phyfps"]) for item in segment_results)
    return VideoPhyFPSRecord(
        video=video_path.name,
        avg_phyfps=float(avg_fps),
        segment_phyfps=segment_values,
    )


def predict_video_directory(
    *,
    video_dir: Path,
    chronometer_root: Path | None = None,
    config_path: Path | None = None,
    ckpt_path: Path | None = None,
    device: str = "cuda:0",
    clip_length: int = 30,
    stride: int = 4,
    resolution: int = 216,
) -> list[VideoPhyFPSRecord]:
    paths = resolve_predictor_paths(
        chronometer_root=chronometer_root,
        config_path=config_path,
        ckpt_path=ckpt_path,
    )
    model = load_predictor_model(paths=paths, device=device)
    records: list[VideoPhyFPSRecord] = []
    for video_path in list_generated_videos(video_dir):
        records.append(
            predict_video_file(
                model=model,
                video_path=video_path,
                chronometer_root=paths.chronometer_root,
                device=device,
                clip_length=clip_length,
                stride=stride,
                resolution=resolution,
            )
        )
    return records
