"""Independent in-tree runtime for VerseCrafter camera control."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from worldfoundry.core.io.paths import checkpoint_root_path, hfd_root_path, resolve_data_path
from worldfoundry.runtime.in_tree_cli import ensure_in_tree_runtime, execute_in_tree, require_path


class VerseCrafterRuntime:
    """Run VerseCrafter depth, 4D-control rendering, and diffusion stages."""

    SOURCE_REVISION = "008693b52aa74367afb34d183046fecf88100bdc"
    GPU_MEMORY_MODES = (
        "model_full_load",
        "model_full_load_and_qfloat8",
        "model_cpu_offload",
        "model_cpu_offload_and_qfloat8",
        "sequential_cpu_offload",
    )

    def __init__(
        self,
        *,
        checkpoint_path: Any = None,
        base_model_path: Any = None,
        moge_model_path: Any = None,
        python_executable: Any = None,
        device: str = "cuda",
        gpu_memory_mode: str = "model_cpu_offload",
        env: Mapping[str, Any] | None = None,
    ) -> None:
        self.repo_root = ensure_in_tree_runtime(self.bundled_repo_root(), package_file=__file__)
        hfd = hfd_root_path()
        checkpoints = checkpoint_root_path()
        self.checkpoint_path = Path(
            checkpoint_path or hfd / "TencentARC--VerseCrafter"
        ).expanduser()
        self.base_model_path = Path(base_model_path or checkpoints / "Wan2.1-T2V-14B").expanduser()
        self.moge_model_path = Path(moge_model_path or hfd / "Ruicheng--moge-2-vitl-normal").expanduser()
        self.python_executable = str(python_executable or sys.executable)
        self.device = device
        normalized_memory_mode = str(gpu_memory_mode).strip().lower()
        if normalized_memory_mode not in self.GPU_MEMORY_MODES:
            raise ValueError(
                "VerseCrafter gpu_memory_mode must be one of "
                f"{', '.join(self.GPU_MEMORY_MODES)}; got {gpu_memory_mode!r}"
            )
        self.gpu_memory_mode = normalized_memory_mode
        self.env = dict(env or {})

    @staticmethod
    def bundled_repo_root() -> Path:
        return Path(__file__).resolve().parent / "versecrafter_runtime"

    @classmethod
    def from_pretrained(cls, pretrained_model_path: Any = None, **kwargs: Any) -> "VerseCrafterRuntime":
        return cls(checkpoint_path=pretrained_model_path, **kwargs)

    @staticmethod
    def _torchrun_prefix(python_executable: str) -> list[str]:
        """Return a collision-free one-rank launcher for concurrent Studio jobs."""

        return [
            python_executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=1",
        ]

    def plan(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "model_id": "versecrafter",
            "repo_root": str(self.repo_root),
            "source_revision": self.SOURCE_REVISION,
            "checkpoint_path": str(self.checkpoint_path),
            "base_model_path": str(self.base_model_path),
            "moge_model_path": str(self.moge_model_path),
            "gpu_memory_mode": self.gpu_memory_mode,
            "inputs": {key: str(value) for key, value in kwargs.items()},
        }

    def _python_paths(self) -> tuple[Path, ...]:
        worldfoundry_root = Path(__file__).resolve().parents[3]
        return (
            worldfoundry_root.parent,
            self.repo_root,
            worldfoundry_root / "synthesis" / "visual_generation" / "video_x_fun" / "video_x_fun_runtime",
        )

    def _process_env(self) -> tuple[dict[str, str], str]:
        env = {str(key): str(value) for key, value in self.env.items()}
        effective_device = self.device
        if self.device.startswith("cuda:") and "CUDA_VISIBLE_DEVICES" not in env:
            env["CUDA_VISIBLE_DEVICES"] = self.device.split(":", 1)[1]
            effective_device = "cuda"
        python_path = os.pathsep.join(str(path.resolve()) for path in self._python_paths())
        inherited = os.environ.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(part for part in (python_path, inherited) if part)
        return env, effective_device

    def _run_stage(
        self,
        name: str,
        command: Sequence[Any],
        *,
        log_dir: Path,
        env: Mapping[str, str],
    ) -> None:
        process_env = os.environ.copy()
        process_env.update(env)
        completed = subprocess.run(
            [str(item) for item in command],
            cwd=self.repo_root,
            env=process_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        log_path = log_dir / f"{name}.log"
        log_path.write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(f"VerseCrafter stage {name!r} exited with code {completed.returncode}; see {log_path}")

    def predict(
        self,
        *,
        prompt: str,
        image_path: Any,
        trajectory_npz: Any = None,
        output_path: Any,
        width: int = 832,
        height: int = 480,
        num_frames: int = 81,
        fps: int = 16,
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
        seed: int = 2025,
        return_dict: bool = True,
        **_: Any,
    ) -> Any:
        image = require_path(image_path, "VerseCrafter input image", kind="file")
        output = Path(output_path).expanduser().resolve()
        input_dir = output.parent / f".{output.stem}_versecrafter_inputs"
        input_dir.mkdir(parents=True, exist_ok=True)
        default_trajectory = trajectory_npz is None or not str(trajectory_npz).strip()
        if default_trajectory:
            trajectory = self._write_default_trajectory(input_dir / "camera_trajectory.npz", num_frames)
        else:
            trajectory = require_path(trajectory_npz, "VerseCrafter camera trajectory", kind="file")
        checkpoint = require_path(self.checkpoint_path, "VerseCrafter checkpoint", kind="dir")
        for relative in (
            "config.json",
            "diffusion_pytorch_model.safetensors.index.json",
            "diffusion_pytorch_model-00001-of-00004.safetensors",
            "diffusion_pytorch_model-00002-of-00004.safetensors",
            "diffusion_pytorch_model-00003-of-00004.safetensors",
            "diffusion_pytorch_model-00004-of-00004.safetensors",
        ):
            require_path(checkpoint / relative, f"VerseCrafter asset {relative}", kind="file")
        base = require_path(self.base_model_path, "Wan2.1-T2V-14B base model", kind="dir")
        for relative in (
            "config.json",
            "diffusion_pytorch_model.safetensors.index.json",
            "Wan2.1_VAE.pth",
            "models_t5_umt5-xxl-enc-bf16.pth",
        ):
            require_path(base / relative, f"Wan2.1-T2V-14B asset {relative}", kind="file")
        moge = require_path(self.moge_model_path, "MoGe-v2 checkpoint", kind="dir")
        require_path(moge / "model.pt", "MoGe-v2 weights", kind="file")
        depth_dir = input_dir / "estimated_depth"
        render_dir = input_dir / "rendering_4D_maps"
        native_output = input_dir / "outputs"
        mask_dir = input_dir / "empty_masks"
        for directory in (depth_dir, render_dir, native_output, mask_dir):
            directory.mkdir(parents=True, exist_ok=True)

        ellipsoid = input_dir / "empty_3D_gaussian_trajectory.json"
        ellipsoid.write_text(
            json.dumps(
                {
                    "metadata": {"num_frames": int(num_frames), "num_objects": 0, "obj_id_to_color_idx": {}},
                    "frames": [
                        {"frame_index": index, "objects": []}
                        for index in range(int(num_frames))
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        env, effective_device = self._process_env()
        self._run_stage(
            "depth",
            [
                self.python_executable,
                "inference/moge-v2_infer.py",
                "--input",
                image,
                "--output",
                depth_dir,
                "--pretrained",
                moge,
                "--device",
                effective_device,
                "--maps",
            ],
            log_dir=input_dir,
            env=env,
        )
        depth_npz = require_path(depth_dir / "depth_intrinsics.npz", "VerseCrafter estimated depth", kind="file")
        render_command: list[Any] = [
            self.python_executable,
            "inference/rendering_4D_control_maps.py",
            "--png_path",
            image,
            "--npz_path",
            depth_npz,
            "--mask_dir",
            mask_dir,
            "--trajectory_npz",
            trajectory,
            "--ellipsoid_json",
            ellipsoid,
            "--output_dir",
            render_dir,
            "--device",
            effective_device,
            "--fps",
            str(fps),
            "--target_height",
            str(height),
            "--target_width",
            str(width),
            "--use_fp16",
        ]
        self._run_stage("render_4d", render_command, log_dir=input_dir, env=env)
        for filename in (
            "background_RGB.mp4",
            "background_depth.mp4",
            "3D_gaussian_RGB.mp4",
            "3D_gaussian_depth.mp4",
            "merged_mask.mp4",
        ):
            require_path(render_dir / filename, f"VerseCrafter control map {filename}", kind="file")
        if default_trajectory:
            self._validate_default_control_maps(render_dir)

        command = [
            *self._torchrun_prefix(self.python_executable),
            "inference/versecrafter_inference.py",
            "--transformer_path",
            checkpoint,
            "--base_model_path",
            base,
            "--config_path",
            resolve_data_path(
                "models", "runtime", "configs", "versecrafter", "wan2.1", "wan_civitai.yaml"
            ),
            "--rendering_maps_path",
            render_dir,
            "--input_image_path",
            image,
            "--save_path",
            native_output,
            "--sample_size",
            f"{int(height)},{int(width)}",
            "--video_length",
            str(num_frames),
            "--fps",
            str(fps),
            "--num_inference_steps",
            str(num_inference_steps),
            "--guidance_scale",
            str(guidance_scale),
            "--seed",
            str(seed),
            "--ulysses_degree",
            "1",
            "--ring_degree",
            "1",
            "--gpu_memory_mode",
            self.gpu_memory_mode,
            "--prompt",
            prompt,
        ]
        result = execute_in_tree(
            command,
            cwd=self.repo_root,
            output_path=output,
            search_roots=(native_output,),
            env=env,
            python_paths=self._python_paths(),
            preferred_names=("generated_video_0.mp4",),
        )
        return result if return_dict else result.get("video")

    @staticmethod
    def _validate_default_control_maps(render_dir: Path) -> None:
        """Reject the empty 4D map produced by a misoriented default camera."""

        import cv2

        captures = []
        try:
            frames = []
            for filename in ("merged_mask.mp4", "background_depth.mp4"):
                capture = cv2.VideoCapture(str(render_dir / filename))
                captures.append(capture)
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise RuntimeError(f"VerseCrafter could not decode control map {filename}")
                frames.append(frame)
        finally:
            for capture in captures:
                capture.release()
        valid_background = np.mean(frames[0][..., 0] < 128)
        has_depth = np.any(frames[1][..., 0] > 2)
        if valid_background < 0.01 or not has_depth:
            raise RuntimeError(
                "VerseCrafter default trajectory rendered an empty background control map"
            )

    @staticmethod
    def _write_default_trajectory(path: Path, num_frames: int) -> Path:
        """Write a deterministic, gentle camera translation for demo inference."""

        frame_count = max(1, int(num_frames))
        extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None], frame_count, axis=0)
        # The renderer unprojects depth in OpenCV camera coordinates, then
        # transforms the point cloud into Blender world coordinates.  Its
        # trajectory loader flips the camera's local Y/Z axes before taking
        # the inverse.  An identity Blender pose therefore looks away from
        # the point cloud and renders an empty (gray) background.  Set the
        # Blender camera basis so frame zero maps back to the input view.
        extrinsics[:, :3, :3] = np.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        if frame_count > 1:
            extrinsics[:, 0, 3] = np.linspace(0.0, 0.12, frame_count, dtype=np.float32)
            extrinsics[:, 2, 3] = np.linspace(0.0, -0.04, frame_count, dtype=np.float32)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, extrinsics=extrinsics)
        return path


__all__ = ["VerseCrafterRuntime"]
