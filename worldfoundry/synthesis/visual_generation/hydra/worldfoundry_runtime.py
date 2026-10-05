"""Independent in-tree runtime for HyDRA."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.runtime.in_tree_cli import ensure_in_tree_runtime, execute_in_tree, require_path

_PROJECT_ROOT = Path(__file__).resolve().parents[4]


class HydraRuntime:
    """Invoke the pinned HyDRA inference entrypoint from the bundled source tree."""

    SOURCE_REVISION = "48652becf0030edbaabde4d04f33bb33ad380410"

    def __init__(
        self,
        *,
        checkpoint_path: str | Path | None = None,
        base_model_path: str | Path | None = None,
        python_executable: str | Path | None = None,
        device: str = "cuda",
        env: Mapping[str, Any] | None = None,
    ) -> None:
        self.repo_root = ensure_in_tree_runtime(self.bundled_repo_root(), package_file=__file__)
        self.checkpoint_path = Path(
            checkpoint_path
            or os.environ.get("WORLDFOUNDRY_HYDRA_CHECKPOINT")
            or Path(os.environ.get("WORLDFOUNDRY_HFD_ROOT", "cache/hfd")) / "H-EmbodVis--HyDRA" / "hydra.ckpt"
        ).expanduser()
        self.base_model_path = Path(
            base_model_path
            or os.environ.get("WORLDFOUNDRY_WAN21_1P3B_ROOT")
            or Path(os.environ.get("WORLDFOUNDRY_HFD_ROOT", "cache/hfd")) / "Wan-AI--Wan2.1-T2V-1.3B"
        ).expanduser()
        self.python_executable = str(python_executable or sys.executable)
        self.device = device
        self.env = dict(env or {})

    @staticmethod
    def bundled_repo_root() -> Path:
        return Path(__file__).resolve().parent / "hydra_runtime"

    @classmethod
    def from_pretrained(cls, pretrained_model_path: Any = None, **kwargs: Any) -> "HydraRuntime":
        if isinstance(pretrained_model_path, Mapping):
            options = dict(pretrained_model_path)
            options.update(kwargs)
            checkpoint = options.pop("checkpoint_path", options.pop("model_path", None))
            return cls(checkpoint_path=checkpoint, **options)
        return cls(checkpoint_path=pretrained_model_path, **kwargs)

    def plan(self, *, video_path: Any, camera_json: Any, output_path: Any, **_: Any) -> dict[str, Any]:
        return {
            "model_id": "hydra",
            "repo_root": str(self.repo_root),
            "source_revision": self.SOURCE_REVISION,
            "checkpoint_path": str(self.checkpoint_path),
            "base_model_path": str(self.base_model_path),
            "video_path": str(video_path),
            "camera_json": str(camera_json),
            "output_path": str(output_path),
        }

    def predict(
        self,
        *,
        prompt: str,
        video_path: Any,
        camera_json: Any,
        output_path: Any,
        num_frames: int = 77,
        fps: int = 15,
        width: int = 832,
        height: int = 480,
        cfg_scale: float = 5.0,
        num_inference_steps: int = 50,
        seed: int = 42,
        return_dict: bool = True,
        **_: Any,
    ) -> Any:
        video = require_path(video_path, "HyDRA condition video", kind="file")
        camera = require_path(camera_json, "HyDRA camera JSON", kind="file")
        checkpoint = require_path(self.checkpoint_path, "HyDRA checkpoint", kind="file")
        base = require_path(self.base_model_path, "Wan2.1-T2V-1.3B base model", kind="dir")
        output = Path(output_path).expanduser().resolve()
        input_dir = output.parent / f".{output.stem}_hydra_inputs"
        input_dir.mkdir(parents=True, exist_ok=True)
        caption = input_dir / "caption.txt"
        caption.write_text(prompt.strip() + "\n", encoding="utf-8")
        command = [
            self.python_executable,
            "infer_hydra.py",
            "--cond_video",
            video,
            "--cond_json",
            camera,
            "--caption_txt",
            caption,
            "--ckpt_path",
            checkpoint,
            "--base_dit_path",
            base / "diffusion_pytorch_model.safetensors",
            "--base_text_encoder_path",
            base / "models_t5_umt5-xxl-enc-bf16.pth",
            "--base_vae_path",
            base / "Wan2.1_VAE.pth",
            "--output_path",
            output,
            "--device",
            self.device,
            "--height",
            str(height),
            "--width",
            str(width),
            "--cond_frames",
            str(num_frames),
            "--fps",
            str(fps),
            "--num_inference_steps",
            str(num_inference_steps),
            "--cfg_scale",
            str(cfg_scale),
            "--seed",
            str(seed),
        ]
        result = execute_in_tree(
            command,
            cwd=self.repo_root,
            output_path=output,
            env=self.env,
            python_paths=(_PROJECT_ROOT, self.repo_root),
        )
        return result if return_dict else result.get("video")


__all__ = ["HydraRuntime"]
