"""WorldFoundry subprocess adapter for the bundled official Evoke runtime."""

from __future__ import annotations

import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.core.io.paths import resolve_local_hf_model_path
from worldfoundry.runtime.in_tree_cli import ensure_in_tree_runtime, execute_in_tree, require_path

SOURCE_REVISION = "74d268516d95c8fceadd2378f91a73f9f187042b"
CHECKPOINT_REPO = "AlayaLab/Evoke"
CHECKPOINT_REVISION = "7fa34ecef85754fde6f08996b1ece9d195dcd2f4"
VIGEO_REPO = "pkqbajng/ViGeo1.1"
VIGEO_REVISION = "49103e6eeab888bae974251d3578b496bec711d7"
SUPPORTED_MODES = frozenset({"auto", "t2v", "i2v", "v2v"})
OFFLINE_ENV = {
    "DIFFUSERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}

_COMPATIBILITY_PROBE = r"""
import json
from importlib.metadata import version

from packaging.version import Version

versions = {
    "diffusers": version("diffusers"),
    "torch": version("torch"),
    "transformers": version("transformers"),
}
if Version(versions["diffusers"]) < Version("0.39.0"):
    raise RuntimeError(f"Evoke requires diffusers>=0.39.0, found {versions['diffusers']}")
if not (Version("2.5") <= Version(versions["torch"]) < Version("2.12")):
    raise RuntimeError(f"the WorldFoundry Evoke port supports torch>=2.5,<2.12, found {versions['torch']}")
if not (Version("4.57") <= Version(versions["transformers"]) < Version("5")):
    raise RuntimeError(
        "the shared WorldFoundry environment requires transformers>=4.57,<5, "
        f"found {versions['transformers']}"
    )

import cv2
import torch
from diffusers import AutoencoderKLWan
from diffusers.models._modeling_parallel import ContextParallelInput, ContextParallelOutput
from diffusers.models.attention import AttentionMixin, AttentionModuleMixin
from diffusers.models.cache_utils import CacheMixin
from transformers import AutoTokenizer, UMT5EncoderModel

if not torch.cuda.is_available():
    raise RuntimeError("torch.cuda.is_available() is false")
print(json.dumps(versions, sort_keys=True))
"""


def _as_single_path(value: Any, label: str) -> Any:
    """Accept a file-backed input or a one-item standard-pipeline sequence."""
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"Evoke accepts exactly one {label} path, got {len(value)}")
        value = value[0]
    if value is not None and not isinstance(value, (str, os.PathLike)):
        raise TypeError(f"Evoke {label} input must be a local file path, got {type(value).__name__}")
    return value


def _cuda_visible_devices(device: Any, inherited: str | None) -> str:
    """Resolve a CUDA device without importing torch in the lightweight adapter."""
    normalized = str(device or "cuda").strip().lower()
    if normalized == "cuda":
        return inherited or "0"
    if normalized.startswith("cuda:"):
        suffix = normalized.split(":", 1)[1].strip()
    elif normalized.isdigit():
        suffix = normalized
    else:
        raise ValueError(f"Evoke requires a CUDA device, got {device!r}")
    if not suffix.isdigit():
        raise ValueError(f"invalid Evoke CUDA device: {device!r}")
    if inherited:
        visible = [item.strip() for item in inherited.split(",") if item.strip()]
        index = int(suffix)
        if index < len(visible):
            return visible[index]
    return suffix


@lru_cache(maxsize=8)
def _probe_runtime(python_executable: str, cuda_visible_devices: str) -> tuple[bool, str]:
    env = {
        **os.environ,
        **OFFLINE_ENV,
        "CUDA_VISIBLE_DEVICES": cuda_visible_devices,
    }
    try:
        completed = subprocess.run(
            [python_executable, "-c", _COMPATIBILITY_PROBE],
            env=env,
            capture_output=True,
            text=True,
            # Cold imports from the shared DolphinFS environment can take several minutes
            # (the Evoke dependency probe took 329 seconds in a local validation run).
            # This is a readiness probe, not an inference SLA.
            timeout=600,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    detail = (completed.stdout if completed.returncode == 0 else completed.stderr).strip()
    return completed.returncode == 0, detail


class EvokeRuntime:
    """Run the official post-distillation Evoke recipe from bundled source."""

    SOURCE_REVISION = SOURCE_REVISION
    CHECKPOINT_REVISION = CHECKPOINT_REVISION
    VIGEO_REVISION = VIGEO_REVISION

    def __init__(
        self,
        *,
        checkpoint_path: Any = None,
        base_model_path: Any = None,
        transformer_path: Any = None,
        vigeo_path: Any = None,
        runtime_root: Any = None,
        python_executable: Any = None,
        device: str = "cuda",
        env: Mapping[str, Any] | None = None,
    ) -> None:
        requested_runtime = Path(runtime_root).expanduser() if runtime_root else self.bundled_repo_root()
        self.repo_root = ensure_in_tree_runtime(requested_runtime, package_file=__file__)
        self.checkpoint_source = checkpoint_path or CHECKPOINT_REPO
        self.base_model_source = base_model_path
        self.transformer_source = transformer_path
        self.vigeo_source = vigeo_path or VIGEO_REPO
        self.python_executable = str(python_executable or sys.executable)
        self.device = str(device)
        self.env = {str(key): str(value) for key, value in dict(env or {}).items()}

    @staticmethod
    def bundled_repo_root() -> Path:
        return Path(__file__).resolve().parent / "evoke_runtime"

    @classmethod
    def from_pretrained(cls, pretrained_model_path: Any = None, **kwargs: Any) -> "EvokeRuntime":
        return cls(checkpoint_path=pretrained_model_path, **kwargs)

    def _cuda_visible_devices(self) -> str:
        inherited = self.env.get("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"))
        return _cuda_visible_devices(self.device, inherited)

    def runtime_compatibility(self) -> dict[str, Any]:
        """Probe the selected interpreter for the APIs used by Evoke inference."""
        cuda_visible_devices = self._cuda_visible_devices()
        ready, detail = _probe_runtime(self.python_executable, cuda_visible_devices)
        return {
            "ready": ready,
            "python_executable": self.python_executable,
            "cuda_visible_devices": cuda_visible_devices,
            "detail": detail,
            "environment": "_unified",
        }

    def _require_runtime_compatibility(self) -> None:
        compatibility = self.runtime_compatibility()
        if compatibility["ready"]:
            return
        raise RuntimeError(
            "Evoke cannot run in the selected Python environment. Install or update the shared "
            "WorldFoundry environment with "
            "`bash scripts/setup/model_env_install.sh --model evoke`, then launch the runner from "
            f"that environment. Probe detail: {compatibility['detail']}"
        )

    def _resolve_model_paths(self, *, use_geometric_state: bool) -> tuple[Path, Path, Path | None]:
        resolver_env = {**os.environ, **self.env}
        checkpoint_root = resolve_local_hf_model_path(
            self.checkpoint_source,
            revision=CHECKPOINT_REVISION,
            env=resolver_env,
        )
        base = resolve_local_hf_model_path(
            self.base_model_source or checkpoint_root / "evoke-base",
            env=resolver_env,
        )
        transformer = resolve_local_hf_model_path(
            self.transformer_source or checkpoint_root / "evoke" / "stage3_post_distillation",
            env=resolver_env,
        )
        for relative in ("vae", "scheduler", "text_encoder", "tokenizer"):
            require_path(base / relative, f"Evoke base component {relative}", kind="dir")
        require_path(transformer / "transformer", "Evoke post-distillation transformer", kind="dir")
        vigeo = None
        if use_geometric_state:
            vigeo = resolve_local_hf_model_path(
                self.vigeo_source,
                required_files=("vigeo.pt",),
                revision=VIGEO_REVISION,
                env=resolver_env,
            )
            require_path(vigeo / "vigeo.pt", "ViGeo1.1 weights", kind="file")
        return base, transformer, vigeo

    @staticmethod
    def _resolve_pose(
        trajectory_npz: Any = None,
        pose_path: Any = None,
        camera_trajectory: Any = None,
    ) -> Path | None:
        values = [value for value in (trajectory_npz, pose_path, camera_trajectory) if value is not None]
        if not values:
            return None
        resolved = [require_path(_as_single_path(value, "camera trajectory"), "Evoke camera trajectory", kind="file") for value in values]
        if any(path != resolved[0] for path in resolved[1:]):
            raise ValueError("Evoke camera trajectory aliases refer to different files")
        return resolved[0]

    @staticmethod
    def _resolve_mode(mode: str | None, image_path: Path | None, video_path: Path | None) -> str:
        normalized = str(mode or "auto").strip().lower()
        if normalized not in SUPPORTED_MODES:
            raise ValueError(f"unsupported Evoke mode {mode!r}; expected one of {sorted(SUPPORTED_MODES)}")
        if normalized == "auto":
            return "v2v" if video_path is not None else "i2v" if image_path is not None else "t2v"
        return normalized

    def plan(self, **kwargs: Any) -> dict[str, Any]:
        image_value = _as_single_path(kwargs.get("image_path"), "image")
        video_value = _as_single_path(kwargs.get("video_path"), "video")
        image = Path(image_value).expanduser().resolve() if image_value is not None else None
        video = Path(video_value).expanduser().resolve() if video_value is not None else None
        mode = self._resolve_mode(kwargs.get("mode") or kwargs.get("sample_type"), image, video)
        pose = next(
            (
                kwargs.get(key)
                for key in ("trajectory_npz", "pose_path", "camera_trajectory")
                if kwargs.get(key) is not None
            ),
            None,
        )
        geometric = bool(pose) if kwargs.get("use_geometric_state") is None else bool(kwargs["use_geometric_state"])
        return {
            "model_id": "evoke",
            "repo_root": str(self.repo_root),
            "source_revision": SOURCE_REVISION,
            "checkpoint_source": str(self.checkpoint_source),
            "checkpoint_revision": CHECKPOINT_REVISION,
            "vigeo_source": str(self.vigeo_source) if geometric else None,
            "vigeo_revision": VIGEO_REVISION if geometric else None,
            "mode": mode,
            "use_geometric_state": geometric,
            "environment": "_unified",
        }

    def preflight(self, *, use_geometric_state: bool = False) -> dict[str, Any]:
        missing: list[str] = []
        paths: dict[str, str] = {}
        try:
            base, transformer, vigeo = self._resolve_model_paths(use_geometric_state=use_geometric_state)
            paths = {
                "base_model_path": str(base),
                "transformer_path": str(transformer),
                "vigeo_path": str(vigeo) if vigeo else "",
            }
        except (FileNotFoundError, ValueError) as exc:
            missing.append(str(exc))
        compatibility = self.runtime_compatibility()
        if not compatibility["ready"]:
            missing.append(str(compatibility["detail"]))
        script = self.repo_root / "scripts" / "inference" / "infer_single.py"
        if not script.is_file():
            missing.append(f"bundled Evoke inference entrypoint not found: {script}")
        return {
            "status": "ready" if not missing else "blocked",
            "runtime_root": str(self.repo_root),
            "runtime_script": str(script),
            "source_revision": SOURCE_REVISION,
            "paths": paths,
            "compatibility": compatibility,
            "diagnostics": missing,
        }

    def predict(
        self,
        *,
        prompt: str,
        output_path: Any,
        image_path: Any = None,
        video_path: Any = None,
        trajectory_npz: Any = None,
        pose_path: Any = None,
        camera_trajectory: Any = None,
        mode: str = "auto",
        sample_type: str | None = None,
        use_geometric_state: bool | None = None,
        width: int = 640,
        height: int = 384,
        num_chunks: int = 4,
        num_frames: int | None = None,
        fps: int = 24,
        seed: int = 42,
        num_inference_steps: int = 3,
        guidance_scale: float = 1.0,
        ref_seconds: float = 5.0,
        start_seconds: float = 0.0,
        lingbot_pose_source_fps: int = 24,
        lingbot_pose_source_resolution: Sequence[int] = (480, 832),
        return_dict: bool = True,
        **_: Any,
    ) -> Any:
        image_value = _as_single_path(image_path, "image")
        video_value = _as_single_path(video_path, "video")
        image = require_path(image_value, "Evoke input image", kind="file") if image_value is not None else None
        video = require_path(video_value, "Evoke input video", kind="file") if video_value is not None else None
        pose = self._resolve_pose(trajectory_npz, pose_path, camera_trajectory)
        resolved_mode = self._resolve_mode(sample_type or mode, image, video)

        if resolved_mode == "t2v" and (image is not None or video is not None):
            raise ValueError("Evoke t2v mode does not accept image or video conditioning")
        if resolved_mode == "i2v" and image is None:
            raise ValueError("Evoke i2v mode requires an image path")
        if resolved_mode == "i2v" and video is not None:
            raise ValueError("Evoke i2v mode does not accept a video path")
        if resolved_mode == "v2v" and video is None:
            raise ValueError("Evoke v2v mode requires a video path")

        geometric = pose is not None if use_geometric_state is None else bool(use_geometric_state)
        if geometric and pose is None:
            raise ValueError("Evoke geometric state requires trajectory_npz/pose_path/camera_trajectory")
        if resolved_mode == "t2v" and (pose is not None or geometric):
            raise ValueError("Evoke geometric state is unavailable in t2v mode")
        if resolved_mode == "v2v" and image is not None and not geometric:
            raise ValueError("Evoke warp-free v2v accepts only video conditioning; remove image_path")

        width = int(width)
        height = int(height)
        if width <= 0 or width % 64:
            raise ValueError("Evoke width must be a positive multiple of 64")
        if height <= 0 or height % 16:
            raise ValueError("Evoke height must be a positive multiple of 16")
        chunks = int(num_chunks)
        if chunks <= 0:
            raise ValueError("Evoke num_chunks must be positive")
        frames = int(num_frames) if num_frames is not None else 33 * chunks
        if frames <= 0:
            raise ValueError("Evoke num_frames must be positive")
        if int(num_inference_steps) != 3 or float(guidance_scale) != 1.0:
            raise ValueError(
                "Evoke stage3_post_distillation is a fixed three-step, CFG-free recipe "
                "(num_inference_steps=3, guidance_scale=1.0)"
            )
        if len(tuple(lingbot_pose_source_resolution)) != 2:
            raise ValueError("lingbot_pose_source_resolution must contain height and width")

        if output_path is None:
            raise ValueError("Evoke output_path is required")
        if not isinstance(output_path, (str, os.PathLike)):
            raise TypeError(f"Evoke output_path must be a local file path, got {type(output_path).__name__}")
        output = Path(output_path).expanduser().resolve()
        if output.exists() and output.is_dir():
            raise ValueError(f"Evoke output_path points to a directory: {output}")
        run_dir = output.parent / f".{output.stem}_evoke"
        run_dir.mkdir(parents=True, exist_ok=True)
        base, transformer, vigeo = self._resolve_model_paths(use_geometric_state=geometric)
        self._require_runtime_compatibility()

        command: list[Any] = [
            self.python_executable,
            "scripts/inference/infer_single.py",
            "--ckpt_path",
            base,
            "--transformer_path",
            transformer,
            "--is_enable_stage2",
            "--stage2_num_stages",
            "3",
            "--stage2_steps",
            "1",
            "1",
            "1",
            "--stage2_stage_range",
            "0",
            "0.3333333333333333",
            "0.6666666666666666",
            "1",
            "--stage2_warp_compression_mode",
            "fixed_mem",
            "--height",
            str(height),
            "--width",
            str(width),
            "--num_frames",
            str(frames),
            "--fps",
            str(int(fps)),
            "--num_inference_steps",
            "3",
            "--guidance_scale",
            "1.0",
            "--seed",
            str(int(seed)),
            "--vae_decode_type",
            "persistent",
            "--no_raw_sink_frames",
            "--image_noise_sigma_min",
            "0",
            "--image_noise_sigma_max",
            "0",
            "--sample_type",
            resolved_mode,
            "--prompt",
            str(prompt or ""),
            "--output_folder",
            run_dir,
        ]
        if image is not None:
            command.extend(["--image_path", image])
        if video is not None:
            command.extend(
                [
                    "--video_path",
                    video,
                    "--ref_seconds",
                    str(float(ref_seconds)),
                    "--start_seconds",
                    str(float(start_seconds)),
                ]
            )
        if geometric:
            assert pose is not None and vigeo is not None
            source_height, source_width = (int(value) for value in lingbot_pose_source_resolution)
            command.extend(
                [
                    "--use_geometric_state",
                    "--lingbot_pose_path",
                    pose,
                    "--lingbot_pose_source_fps",
                    str(int(lingbot_pose_source_fps)),
                    "--lingbot_pose_source_resolution",
                    str(source_height),
                    str(source_width),
                    "--lingbot_pose_type",
                    "vipe",
                    "--visibility_aware_noise",
                    "--warp_noise_sigma_invisible",
                    "1.0",
                    "--warp_noise_sigma_min",
                    "0",
                    "--warp_noise_sigma_max",
                    "0.135",
                    "--visible_token_threshold",
                    "0.5",
                    "--prefix_idx_mode",
                    "zero",
                    "--warp_rope_mode",
                    "overlap_noise",
                    "--warp_lag_chunks",
                    "0",
                    "--geo_recon_backend",
                    "da3",
                    "--geo_cloud_update_n",
                    "12",
                    "--geo_da3_process_res",
                    "644",
                    "--geo_depth_backend",
                    "vigeo",
                    "--geo_vigeo_weights",
                    vigeo,
                    "--geo_vigeo_mode",
                    "chunk",
                    "--geo_vigeo_scale_mode",
                    "auto",
                    "--geo_vigeo_scale_value",
                    "0",
                    "--geo_vigeo_depth_median_target",
                    "5",
                    "--geo_vigeo_anchor_windows",
                    "4",
                    "--geo_vigeo_cache_keep_frames",
                    "6",
                    "--geo_vigeo_intr_source",
                    "gt",
                    "--geo_da3_render_mode",
                    "backward_zbuf",
                    "--geo_bw_fill_iters",
                    "12",
                    "--geo_warp_stage0_only",
                    "--max_deg_per_chunk",
                    "10",
                    "--max_trans_per_chunk",
                    "3",
                    "--cap_mode",
                    "clamp",
                    "--pose_smooth_win",
                    "5",
                    "--pose_extend_mode",
                    "relative_replay",
                    "--joystick_hud",
                    "off",
                ]
            )

        runtime_env = {
            **os.environ,
            **self.env,
            **OFFLINE_ENV,
            "CUDA_VISIBLE_DEVICES": self._cuda_visible_devices(),
        }
        project_root = Path(__file__).resolve().parents[4]
        result = execute_in_tree(
            command,
            cwd=self.repo_root,
            output_path=output,
            search_roots=(run_dir,),
            preferred_names=("geo_pred.mp4",),
            env=runtime_env,
            python_paths=(project_root, self.repo_root),
        )
        metadata = result.setdefault("metadata", {})
        metadata.update(
            {
                "model_id": "evoke",
                "source_revision": SOURCE_REVISION,
                "checkpoint_revision": CHECKPOINT_REVISION,
                "vigeo_revision": VIGEO_REVISION if geometric else None,
                "mode": resolved_mode,
                "use_geometric_state": geometric,
                "num_chunks": chunks,
                "requested_num_frames": frames,
                "environment": "_unified",
            }
        )
        return result if return_dict else result.get("video")


__all__ = [
    "CHECKPOINT_REPO",
    "CHECKPOINT_REVISION",
    "EvokeRuntime",
    "SOURCE_REVISION",
    "VIGEO_REPO",
    "VIGEO_REVISION",
]
