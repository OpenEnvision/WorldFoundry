from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from worldfoundry.base_models.three_dimensions.point_clouds.lyra.utils import (
    configure_lyra_runtime_env,
    ensure_path_exists,
    resolve_checkpoint_root,
    resolve_repo_root,
    resolve_required_paths,
)
from worldfoundry.core.io.paths import scratch_directory
from worldfoundry.core.media.codecs.video import save_video_frames, video_tensor_to_uint8_frames

DEFAULT_DA3_MODEL_NAME = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"


class Lyra2Representation:
    """Wrapper for Lyra-2 VIPE + DA3 + Gaussian reconstruction."""

    def __init__(
        self,
        repo_root: str,
        device: str = "cuda",
        da3_model_name: str = DEFAULT_DA3_MODEL_NAME,
        da3_model_path_custom: str | None = None,
        no_vipe: bool = False,
        force: bool = False,
        render_fps: float | None = None,
        render_chunk_size: int = 1,
        vipe_overrides: list[str] | None = None,
    ):
        self.repo_root = repo_root
        self.device = device
        self.da3_model_name = da3_model_name
        self.da3_model_path_custom = da3_model_path_custom
        self.no_vipe = bool(no_vipe)
        self.force = bool(force)
        self.render_fps = render_fps
        self.render_chunk_size = int(render_chunk_size)
        self.vipe_overrides = list(vipe_overrides) if vipe_overrides else None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path,
        device=None,
        da3_model_name: str = DEFAULT_DA3_MODEL_NAME,
        da3_model_path_custom: str | None = None,
        no_vipe: bool = False,
        force: bool = False,
        render_fps: float | None = None,
        render_chunk_size: int = 1,
        vipe_overrides: list[str] | None = None,
        **kwargs,
    ):
        repo_root = resolve_repo_root(pretrained_model_path)
        weights_root = resolve_checkpoint_root(pretrained_model_path, repo_root=repo_root)
        configure_lyra_runtime_env()
        required_paths = resolve_required_paths(
            repo_root,
            da3_model_path_custom=da3_model_path_custom,
            weights_root=weights_root,
        )
        ensure_path_exists(required_paths["da3_model_path_custom"], "Lyra-2 DA3 checkpoint")
        return cls(
            repo_root=repo_root,
            device=device or "cuda",
            da3_model_name=da3_model_name,
            da3_model_path_custom=required_paths["da3_model_path_custom"],
            no_vipe=no_vipe,
            force=force,
            render_fps=render_fps,
            render_chunk_size=render_chunk_size,
            vipe_overrides=vipe_overrides,
        )

    def _ensure_video_path(self, data: dict) -> str:
        if data.get("video_path"):
            return str(Path(data["video_path"]).expanduser().resolve())

        video = data.get("video")
        if video is None and "video_tensor" in data and torch.is_tensor(data["video_tensor"]):
            video = video_tensor_to_uint8_frames(data["video_tensor"])

        if video is None:
            raise ValueError("Lyra2Representation expects either 'video_path', 'video', or 'video_tensor'.")

        video = video_tensor_to_uint8_frames(video) if torch.is_tensor(video) else np.asarray(video)
        if video.ndim != 4:
            raise ValueError(f"Expected video with shape [T,H,W,C], got {tuple(video.shape)}")

        fps = int(data.get("fps", 16))
        output_dir = data.get("output_dir")
        if output_dir:
            output_dir_path = Path(output_dir).expanduser().resolve()
            output_dir_path.mkdir(parents=True, exist_ok=True)
            video_path = output_dir_path / "generated.mp4"
        else:
            video_path = scratch_directory("lyra2_repr_") / "generated.mp4"
        save_video_frames(video, str(video_path), fps=fps)
        return str(video_path)

    def get_representation(self, data: dict[str, object]) -> dict[str, object]:
        video_path = self._ensure_video_path(data)
        output_dir = data.get("output_dir")
        if output_dir is None:
            output_dir = str(Path(video_path).with_name(f"{Path(video_path).stem}_gs_ours"))
        output_dir = str(Path(output_dir).expanduser().resolve())

        from worldfoundry.synthesis.visual_generation.lyra_2.reconstruction import reconstruct_video_gaussians

        outputs = reconstruct_video_gaussians(
            video_path,
            output_dir,
            da3_model_path_custom=self.da3_model_path_custom,
            da3_model_name=self.da3_model_name,
            device=self.device,
            force=self.force,
            no_vipe=self.no_vipe,
            vipe_overrides=self.vipe_overrides,
            render_fps=self.render_fps,
            render_chunk_size=self.render_chunk_size,
        )
        return {
            "video_path": video_path,
            "output_dir": str(outputs.output_dir),
            "reconstructed_scene_path": str(outputs.reconstructed_scene_path),
            "render_video_path": str(outputs.render_video_path),
            "camera_path": str(outputs.camera_path),
            "vipe_prediction_path": (
                None if outputs.vipe_prediction_path is None else str(outputs.vipe_prediction_path)
            ),
        }
