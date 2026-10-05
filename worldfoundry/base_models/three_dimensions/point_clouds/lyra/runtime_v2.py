from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from worldfoundry.base_models.three_dimensions.point_clouds.lyra.utils import (
    configure_lyra_runtime_env,
    ensure_path_exists,
    ensure_repo_on_path,
    prepare_lyra2_runtime_root,
    resolve_checkpoint_root,
    resolve_repo_root,
    resolve_required_paths,
)
from worldfoundry.core.io import save_video_frames, video_tensor_to_uint8_frames

DEFAULT_DA3_MODEL_NAME = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"


class Lyra2Representation:
    """Wrapper for Lyra-2 VIPE + DA3 + Gaussian reconstruction."""

    def __init__(
        self,
        repo_root: str,
        runtime_root: str,
        device: str = "cuda",
        da3_model_name: str = DEFAULT_DA3_MODEL_NAME,
        da3_model_path_custom: Optional[str] = None,
        no_vipe: bool = False,
        force: bool = False,
        render_fps: Optional[float] = None,
        render_chunk_size: int = 1,
        vipe_overrides: Optional[list[str]] = None,
    ):
        self.repo_root = repo_root
        self.runtime_root = runtime_root
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
        da3_model_path_custom: Optional[str] = None,
        no_vipe: bool = False,
        force: bool = False,
        render_fps: Optional[float] = None,
        render_chunk_size: int = 1,
        vipe_overrides: Optional[list[str]] = None,
        **kwargs,
    ):
        repo_root = resolve_repo_root(pretrained_model_path)
        weights_root = resolve_checkpoint_root(pretrained_model_path, repo_root=repo_root)
        runtime_root = prepare_lyra2_runtime_root(repo_root, weights_root=weights_root)
        configure_lyra_runtime_env()
        ensure_repo_on_path(runtime_root)
        required_paths = resolve_required_paths(
            repo_root,
            da3_model_path_custom=da3_model_path_custom,
            weights_root=weights_root,
        )
        ensure_path_exists(required_paths["da3_model_path_custom"], "Lyra-2 DA3 checkpoint")
        return cls(
            repo_root=repo_root,
            runtime_root=runtime_root,
            device=device or "cuda",
            da3_model_name=da3_model_name,
            da3_model_path_custom=required_paths["da3_model_path_custom"],
            no_vipe=no_vipe,
            force=force,
            render_fps=render_fps,
            render_chunk_size=render_chunk_size,
            vipe_overrides=vipe_overrides,
        )

    def _ensure_video_path(self, data: Dict) -> str:
        if data.get("video_path"):
            return str(Path(data["video_path"]).expanduser().resolve())

        video = data.get("video")
        if video is None and "video_tensor" in data and torch.is_tensor(data["video_tensor"]):
            video = video_tensor_to_uint8_frames(data["video_tensor"])

        if video is None:
            raise ValueError("Lyra2Representation expects either 'video_path', 'video', or 'video_tensor'.")

        if torch.is_tensor(video):
            video = video_tensor_to_uint8_frames(video)
        else:
            video = np.asarray(video)
        if video.ndim != 4:
            raise ValueError(f"Expected video with shape [T,H,W,C], got {tuple(video.shape)}")

        fps = int(data.get("fps", 16))
        output_dir = data.get("output_dir")
        if output_dir:
            output_dir_path = Path(output_dir).expanduser().resolve()
            output_dir_path.mkdir(parents=True, exist_ok=True)
            video_path = output_dir_path / "generated.mp4"
        else:
            tmp_dir = Path(tempfile.mkdtemp(prefix="lyra2_repr_"))
            video_path = tmp_dir / "generated.mp4"
        save_video_frames(video, str(video_path), fps=fps)
        return str(video_path)

    def get_representation(self, data: Dict[str, object]) -> Dict[str, object]:
        video_path = self._ensure_video_path(data)
        output_dir = data.get("output_dir")
        if output_dir is None:
            output_dir = str(Path(video_path).with_name(f"{Path(video_path).stem}_gs_ours"))
        output_dir = str(Path(output_dir).expanduser().resolve())

        command = [
            sys.executable,
            "-m",
            "lyra_2._src.inference.vipe_da3_gs_recon",
            "--input_video_path",
            video_path,
            "--output_dir",
            output_dir,
            "--device",
            self.device,
            "--da3_model_name",
            self.da3_model_name,
            "--da3_model_path_custom",
            str(self.da3_model_path_custom),
            "--render_chunk_size",
            str(self.render_chunk_size),
        ]
        if self.no_vipe:
            command.append("--no_vipe")
        if self.force:
            command.append("--force")
        if self.render_fps is not None:
            command.extend(["--render_fps", str(self.render_fps)])
        if self.vipe_overrides:
            command.append("--vipe_overrides")
            command.extend(self.vipe_overrides)

        raise RuntimeError(
            "Lyra-2 reconstruction is blocked because this representation path would run a subprocess. "
            f"Use the in-process Lyra-2 synthesis runtime or port `{command[2]}` into an in-process adapter."
        )

        output_dir_path = Path(output_dir)
        return {
            "video_path": video_path,
            "output_dir": str(output_dir_path),
            "reconstructed_scene_path": str(output_dir_path / "reconstructed_scene.ply"),
            "render_video_path": str(output_dir_path / "gs_trajectory.mp4"),
            "camera_path": str(output_dir_path / "cameras.npz"),
            "vipe_prediction_path": str(output_dir_path / "vipe_predictions.npz"),
        }
