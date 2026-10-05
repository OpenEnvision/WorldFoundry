"""Run the pinned UniWorld-View image/video novel-view inference entry point.

The upstream project has a separate Torch/Diffusers environment and several
large checkpoint bundles.  Keep its own inference implementation authoritative;
this adapter owns WorldFoundry input validation, paths and artifact handling.
"""

from __future__ import annotations

import os
import json
import shutil
import subprocess
import sys
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from worldfoundry.core.io import artifact_root_path
from worldfoundry.core.io.paths import checkpoint_root_candidates, conda_envs_root_path, official_runtime_repo_path
from worldfoundry.pipelines.pipeline_utils import PipelineABC
from worldfoundry.runtime.assets import expand_worldfoundry_path, local_file_is_ready


_WEIGHT_DEFAULTS = {
    "transformer_path": ("Drexubery--UniView",),
    "model_name": ("Wan-AI--Wan2.1-VACE-14B-diffusers",),
    "blip_path": ("Salesforce--blip2-opt-2.7b",),
    "moge_path": ("Ruicheng--moge-2-vitl-normal", "model.pt"),
    "sam2_checkpoint": ("facebook--sam2-hiera-large", "sam2_hiera_large.pt"),
    "segnet_path": ("Carve--tracer_b7", "tracer_b7.pth"),
    "stream3r_path": ("yslan--STream3R",),
    "lora_path": ("Kijai--WanVideo_comfy", "Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors"),
}
_OFFICIAL_LAYOUT = {
    "transformer_path": ("UniView",),
    "model_name": ("Wan2.1-VACE-14B-diffusers",),
    "blip_path": ("blip2-opt-2.7b",),
    "moge_path": ("moge", "model.pt"),
    "sam2_checkpoint": ("sam2", "sam2_hiera_large.pt"),
    "segnet_path": ("tracer_b7.pth",),
    "stream3r_path": ("STream3R",),
    "lora_path": ("loras", "Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors"),
}
_DIRECTORY_WEIGHTS = frozenset({"transformer_path", "model_name", "blip_path", "stream3r_path"})
_LOAD_KEYS = frozenset({"repo_root", "python_executable", "checkpoint_root", *_WEIGHT_DEFAULTS})
_SOURCE_FILES = ("inference.py", "demo.py", "configs/infer_config.py", "extern/STream3R/stream3r/models/stream3r.py")
_UPSTREAM_ENTRY = Path(__file__).with_name("upstream_entry.py")


def _path(value: str | Path) -> Path:
    return expand_worldfoundry_path(value).expanduser().resolve()


def _python_path(value: str | Path) -> Path:
    # Resolving the `bin/python` symlink escapes a venv and loses pyvenv.cfg.
    return Path(os.path.abspath(expand_worldfoundry_path(value).expanduser()))


def _complete_hf_weights(directory: Path) -> bool:
    """Reject a partial Hugging Face shard download before model loading."""
    if not directory.is_dir():
        return False
    # HF repositories may publish both safetensors and PyTorch .bin weights.
    # Transformers uses one format, so the unused format need not be downloaded.
    for suffix in (".safetensors", ".bin"):
        files = list(directory.glob(f"*{suffix}"))
        indexes = list(directory.glob(f"*{suffix}.index.json"))
        if not files and not indexes:
            continue
        try:
            if indexes:
                names = set().union(*(
                    json.loads(index.read_text(encoding="utf-8"))["weight_map"].values()
                    for index in indexes
                ))
                return bool(names) and all(
                    local_file_is_ready(directory / name)
                    for name in names
                )
            return not any("-of-" in path.name for path in files) and all(
                local_file_is_ready(path) for path in files
            )
        except (OSError, ValueError, KeyError, TypeError):
            return False
    return False


class UniWorldViewPipeline(PipelineABC):
    """WorldFoundry interface to the official UniWorld-View CLI."""

    MODEL_ID = "uniworld-view"

    def __init__(
        self,
        *,
        repo_root: str | Path,
        python_executable: str | Path,
        weights: Mapping[str, Path],
        device: str,
    ) -> None:
        super().__init__(model_id=self.MODEL_ID, device=device)
        self.repo_root = _path(repo_root)
        self.python_executable = str(_python_path(python_executable))
        self.weights = dict(weights)
        self._check_source()

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path | Mapping[str, Any] | None = None,
        required_components: Mapping[str, Any] | None = None,
        device: str = "cuda:0",
        **kwargs: Any,
    ) -> "UniWorldViewPipeline":
        options = dict(model_path) if isinstance(model_path, Mapping) else {}
        if model_path is not None and not isinstance(model_path, Mapping):
            options["repo_root"] = model_path
        options.update(dict(required_components or {}))
        options.update(kwargs)
        cls._strip_framework_loading_options(options)
        for key in ("runtime_profile", "variant_id", "pipeline_binding", "lazy"):
            options.pop(key, None)
        model_ref = options.pop("model_path", None)
        if model_ref is not None:
            options.setdefault("repo_root", model_ref)
        unknown = set(options) - _LOAD_KEYS
        if unknown:
            raise ValueError(f"unsupported UniWorld-View loading options: {sorted(unknown)}")
        repo_root = _path(options.pop("repo_root", None) or official_runtime_repo_path(
            "UniWorld-View", specific_env="UNIWORLD_VIEW_REPO"
        ))
        dedicated_python = conda_envs_root_path() / "uniworld-view-cu124" / "bin" / "python"
        python_executable = (
            options.pop("python_executable", None)
            or os.getenv("UNIWORLD_VIEW_PYTHON")
            or (dedicated_python if dedicated_python.is_file() else sys.executable)
        )
        checkpoint_root = options.pop("checkpoint_root", None)
        root = _path(checkpoint_root) if checkpoint_root is not None else repo_root / "checkpoints"
        defaults = {}
        for key, parts in _WEIGHT_DEFAULTS.items():
            official_path = root.joinpath(*_OFFICIAL_LAYOUT[key])
            if checkpoint_root is not None or official_path.exists():
                defaults[key] = official_path
            else:
                candidates = checkpoint_root_candidates(*parts)
                defaults[key] = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
        weights = {key: _path(options.pop(key, defaults[key])) for key in _WEIGHT_DEFAULTS}
        return cls(repo_root=repo_root, python_executable=python_executable, weights=weights, device=device)

    def _check_source(self) -> None:
        missing = [str(self.repo_root / relative) for relative in _SOURCE_FILES if not (self.repo_root / relative).is_file()]
        if missing:
            raise FileNotFoundError(
                "UniWorld-View official source is incomplete. Clone the pinned upstream repository "
                "and STream3R under extern/STream3R, then set UNIWORLD_VIEW_REPO or repo_root. "
                f"Missing: {missing}"
            )
        if not Path(self.python_executable).is_file():
            raise FileNotFoundError(f"UniWorld-View Python executable does not exist: {self.python_executable}")

    def _check_weights(self, *, mode: str, geometry_backend: str, render_method: str) -> None:
        required = set(_WEIGHT_DEFAULTS)
        if mode == "single_view" or render_method != "warp":
            required.remove("sam2_checkpoint")
        if mode == "single_view" or geometry_backend == "mosca":
            required.remove("stream3r_path")
        missing = []
        for key in sorted(required):
            target = self.weights[key]
            if key in _DIRECTORY_WEIGHTS:
                ok = target.is_dir()
                if key == "stream3r_path":
                    ok = ok and (target / "config.json").is_file() and (target / "model.safetensors").is_file()
                elif key == "model_name":
                    ok = (
                        ok
                        and _complete_hf_weights(target / "vae")
                        and _complete_hf_weights(target / "text_encoder")
                        and (target / "tokenizer").is_dir()
                        and any((target / "tokenizer").iterdir())
                    )
                else:
                    ok = ok and (target / "config.json").is_file() and _complete_hf_weights(target)
            else:
                ok = local_file_is_ready(target)
            if not ok:
                missing.append(f"{key}={target}")
        if missing:
            raise FileNotFoundError(
                "UniWorld-View requires official local checkpoints. Download with upstream "
                "checkpoints/download_hf.sh --stream3r (for dynamic_view), or override the paths. "
                "Missing/incomplete: " + ", ".join(missing)
            )
        if render_method in {"hybrid", "mesh"}:
            check = subprocess.run(
                [self.python_executable, "-c", "from pytorch3d.renderer import MeshRasterizer"],
                cwd=self.repo_root, capture_output=True, text=True, check=False, timeout=90,
            )
            if check.returncode:
                tail = "\n".join(check.stderr.splitlines()[-5:])
                raise RuntimeError(
                    "UniWorld-View hybrid/mesh renderer requires PyTorch3D in its separate Python environment. "
                    "Install with the upstream extern/install_pytorch3d.sh script. " + tail
                )

    def __call__(
        self,
        prompt: str = "",
        images: str | Path | None = None,
        video: str | Path | None = None,
        *,
        image_path: str | Path | None = None,
        video_path: str | Path | None = None,
        mode: str | None = None,
        geometry_backend: str = "stream3r",
        render_method: str = "hybrid",
        height: int = 480,
        width: int = 832,
        keep_aspect_ratio: bool = True,
        num_frames: int = 81,
        num_inference_steps: int = 8,
        guidance_scale: float = 4.0,
        fps: int = 16,
        seed: int = 43,
        stride: int = 1,
        traj_type: str = "custom",
        d_phi: float = 50.0,
        d_theta: float = 0.0,
        x_offset: float = 0.0,
        y_offset: float = 0.0,
        z_offset: float = 0.0,
        radius_scale: float = 1.0,
        low_gpu_memory_mode: bool = False,
        mosca_ws: str | Path | None = None,
        output_path: str | Path | None = None,
        timeout_seconds: int = 21600,
        return_dict: bool = False,
        plan_only: bool = False,
    ) -> Any:
        if prompt and str(prompt).strip():
            raise ValueError("UniWorld-View upstream inference generates its own BLIP2 caption and ignores --prompt; leave prompt empty")
        has_image = images is not None or image_path is not None
        has_video = video is not None or video_path is not None
        if has_image == has_video:
            raise ValueError("UniWorld-View requires one image or one video input")
        is_video = has_video
        # Studio supplies the staged input as `images`/`video` and retains the
        # original upload path as `image_path`/`video_path`. Prefer the staged
        # object; the two aliases can legitimately be different file paths.
        selected = (video if video is not None else video_path) if is_video else (
            images if images is not None else image_path
        )
        is_path = isinstance(selected, (str, Path))
        if is_video and not is_path:
            raise TypeError("UniWorld-View video input must be a local file path")
        mode = mode or ("dynamic_view" if is_video else "single_view")
        if mode not in {"single_view", "dynamic_view"}:
            raise ValueError("mode must be single_view or dynamic_view")
        if (mode == "dynamic_view") != is_video:
            raise ValueError("dynamic_view requires video; single_view requires image")
        if geometry_backend not in {"stream3r", "mosca"}:
            raise ValueError("geometry_backend must be stream3r or mosca")
        if render_method not in {"warp", "hybrid", "mesh"}:
            raise ValueError("render_method must be warp, hybrid or mesh")
        if min(height, width) < 64 or height % 16 or width % 16:
            raise ValueError("height and width must be at least 64 and divisible by 16")
        if num_frames < 1 or (num_frames - 1) % 4:
            raise ValueError("num_frames must be 4n+1 for the upstream Wan video VAE")
        if mode == "dynamic_view" and num_frames < 13:
            raise ValueError("dynamic_view requires at least 13 frames so UniView stage2 has its 10 reference frames")
        if num_inference_steps < 1 or stride < 1 or fps < 1 or timeout_seconds < 1:
            raise ValueError("frame count, steps, stride, fps and timeout must be positive")
        if mode == "dynamic_view" and geometry_backend == "mosca" and mosca_ws is None:
            raise ValueError("mosca geometry backend requires a precomputed mosca_ws")
        if mosca_ws is not None and not _path(mosca_ws).is_dir():
            raise FileNotFoundError(f"MoSca workspace does not exist: {_path(mosca_ws)}")

        target = _path(output_path) if output_path else artifact_root_path() / "pipeline_eval" / "uniworld-view.mp4"
        run_root = target.parent / f".{target.stem}-uniworld-view-{uuid.uuid4().hex[:12]}"
        exp_name = "generation"
        generated = run_root / exp_name / "diffusion_correct.mp4"
        source = _path(selected) if is_path else run_root / "reference.png"
        if is_path and not source.is_file():
            raise FileNotFoundError(f"UniWorld-View input does not exist: {source}")
        command = [
            self.python_executable, "-u", str(_UPSTREAM_ENTRY),
            "--image_dir", str(source), "--out_dir", str(run_root), "--exp_name", exp_name,
            "--mode", mode, "--device", self.device, "--geometry_backend", geometry_backend,
            "--render_method", render_method,
            "--height", str(height), "--width", str(width), "--video_length", str(num_frames),
            "--ddim_steps", str(num_inference_steps), "--diffusion_guidance_scale", str(guidance_scale),
            "--fps", str(fps), "--seed", str(seed), "--stride", str(stride),
            "--traj_type", traj_type, "--d_phi", str(d_phi), "--d_theta", str(d_theta),
            "--x_offset", str(x_offset), "--y_offset", str(y_offset), "--z_offset", str(z_offset),
            "--radius_scale", str(radius_scale), "--prompt", str(prompt or ""),
            "--low_gpu_memory_mode", str(low_gpu_memory_mode).lower(),
        ]
        for key, value in self.weights.items():
            command.extend((f"--{key}", str(value)))
        if mosca_ws is not None:
            command.extend(("--mosca_ws", str(_path(mosca_ws))))
        if not keep_aspect_ratio:
            command.append("--no_keep_aspect_ratio")
        if plan_only:
            return {"status": "planned", "command": command, "cwd": str(self.repo_root), "artifact_path": str(target)}
        self._check_weights(mode=mode, geometry_backend=geometry_backend, render_method=render_method)
        if not is_path:
            from PIL import Image
            import numpy as np

            value = selected
            if isinstance(value, Image.Image):
                reference = value.convert("RGB")
            elif isinstance(value, np.ndarray):
                reference = Image.fromarray(value).convert("RGB")
            else:
                raise TypeError("UniWorld-View image input must be a path, PIL image or numpy array")
            run_root.mkdir(parents=True, exist_ok=True)
            reference.save(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        log_path = target.with_suffix(".uniworld-view.log")
        env = os.environ.copy()
        # The official checkout inserts its own vendored source paths. Keep
        # WorldFoundry's interpreter paths out of the dedicated environment.
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        env["PYTHONNOUSERSITE"] = "1"
        with log_path.open("w", encoding="utf-8") as log:
            try:
                result = subprocess.run(
                    command, cwd=self.repo_root, env=env, stdout=log, stderr=subprocess.STDOUT,
                    check=False, timeout=timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError(f"UniWorld-View exceeded {timeout_seconds}s; see {log_path}") from exc
        if result.returncode or not generated.is_file() or generated.stat().st_size == 0:
            tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-25:])
            raise RuntimeError(f"UniWorld-View inference failed (exit {result.returncode}); see {log_path}\n{tail}")
        shutil.copy2(generated, target)
        record = {"status": "success", "artifact_path": str(target), "video": str(target), "log_path": str(log_path), "run_dir": str(run_root / exp_name)}
        return record if return_dict else str(target)

    def run_pipeline_invocation(self, invocation: Any) -> Mapping[str, Any]:
        options = dict(invocation.pipeline_kwargs)
        options.pop("operator_kwargs", None)
        input_path = options.pop("input_path", None) or invocation.request.inputs.get("input_path")
        image = invocation.image
        video = invocation.video
        if input_path is not None and image is None and video is None:
            if options.get("mode") == "dynamic_view" or Path(str(input_path)).suffix.lower() in {".mp4", ".mov", ".avi", ".webm", ".mkv"}:
                video = input_path
            else:
                image = input_path
        return self(
            prompt=invocation.prompt, images=image, video=video,
            output_path=invocation.output_path, return_dict=True, **options,
        )


__all__ = ["UniWorldViewPipeline"]
