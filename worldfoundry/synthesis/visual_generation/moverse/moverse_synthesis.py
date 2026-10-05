"""In-tree headless-pipeline adapter for MoVerse."""

from __future__ import annotations

import json
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from worldfoundry.core.io.paths import resolve_data_path, resolve_local_hf_model_path
from worldfoundry.runtime.in_tree_cli import execute_in_tree
from worldfoundry.synthesis.base_synthesis import BaseSynthesis

OFFICIAL_SOURCE_REPO = "https://github.com/Orange-3DV-Team/MoVerse"
UPSTREAM_REVISION = "ae146303a826dfa7af2fea81fc492c9bc39651e8"
DEFAULT_MOVERSE_REPO = "Orange-3DV-Team/MoVerse"
DEFAULT_FLUX_REPO = "black-forest-labs/FLUX.1-Fill-dev"
DEFAULT_DA3_REPO = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
DEFAULT_WAN_REPO = "Wan-AI/Wan2.1-T2V-1.3B"
REQUIRED_ASSET_FILES = {
    "moverse_path": (
        "gimbal360/autolevel.pth",
        "gimbal360/pytorch_lora_weights.safetensors",
        "gaussian_predictor_checkpoint.pth",
        "self_forcing_video.pt",
        "taew2_1.pth",
    ),
    "flux_path": ("model_index.json",),
    "da3_path": ("config.json", "model.safetensors"),
    "wan_path": (
        "Wan2.1_VAE.pth",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "google/umt5-xxl/tokenizer_config.json",
        "google/umt5-xxl/spiece.model",
    ),
}


def runtime_root() -> Path:
    """Return the vendored MoVerse infer-only source root."""

    return Path(__file__).resolve().parent / "moverse_runtime"


class MoVerseSynthesis(BaseSynthesis):
    """Run image -> panorama -> depth -> Gaussian scaffold -> video."""

    def __init__(self, *, device: str = "cuda", options: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        self.device = device
        self.options = dict(options or {})
        self.model_id = "moverse"
        self.model_name = "MoVerse"

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Any = None,
        args: Any = None,
        device: str | None = None,
        **kwargs: Any,
    ) -> "MoVerseSynthesis":
        del args
        options = dict(pretrained_model_path) if isinstance(pretrained_model_path, Mapping) else {}
        if pretrained_model_path is not None and not isinstance(pretrained_model_path, Mapping):
            options["moverse_path"] = str(pretrained_model_path)
        options.update(kwargs)
        options.pop("model_id", None)
        return cls(device=str(device or options.pop("device", "cuda")), options=options)

    def predict(
        self,
        prompt: str = "",
        images: Any = None,
        video: Any = None,
        interactions: Any = None,
        output_path: str | Path | None = None,
        fps: int | None = None,
        plan_only: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del video
        options = {**self.options, **kwargs}
        run_dir = Path(options.get("run_dir") or tempfile.mkdtemp(prefix="moverse_")).expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        stage_dir = run_dir / "pipeline"
        stage_dir.mkdir(parents=True, exist_ok=True)
        output = _output_path(output_path, run_dir / "moverse.mp4")
        image = _materialize_image(images, run_dir)
        trajectory = _trajectory_path(options, interactions)
        back_prompt = str(options.get("back_prompt") or "")
        num_frames = int(options.get("num_frames", 161))
        output_fps = int(fps or options.get("fps", 16))

        unresolved_assets = {
            "moverse_path": str(options.get("moverse_path") or DEFAULT_MOVERSE_REPO),
            "flux_path": str(options.get("flux_path") or DEFAULT_FLUX_REPO),
            "da3_path": str(options.get("da3_path") or DEFAULT_DA3_REPO),
            "wan_path": str(options.get("wan_path") or DEFAULT_WAN_REPO),
        }
        resolved_assets, asset_errors = _resolve_assets(unresolved_assets, required=not plan_only)
        python_executable = str(options.get("python_executable") or sys.executable)
        config_path = resolve_data_path(
            "models", "runtime", "configs", "moverse", "inference", "cond_inference_multi.yaml"
        )
        default_config_path = resolve_data_path(
            "models", "runtime", "configs", "moverse", "default_config.yaml"
        )
        moverse_root = Path(resolved_assets["moverse_path"])
        command = [
            python_executable,
            str(runtime_root() / "scripts" / "infer.py"),
            "--image",
            image,
            "--prompt",
            prompt,
            "--output_dir",
            str(stage_dir),
            "--gimbal360_dir",
            str(moverse_root / "gimbal360"),
            "--flux_model_path",
            resolved_assets["flux_path"],
            "--da3_model_dir",
            resolved_assets["da3_path"],
            "--gaussian_checkpoint",
            str(moverse_root / "gaussian_predictor_checkpoint.pth"),
            "--video_checkpoint",
            str(moverse_root / "self_forcing_video.pt"),
            "--taehv_checkpoint",
            str(moverse_root / "taew2_1.pth"),
            "--config_path",
            str(config_path),
            "--default_config_path",
            str(default_config_path),
            "--num_frames",
            str(num_frames),
            "--fps",
            str(output_fps),
            "--python_executable",
            python_executable,
        ]
        if back_prompt:
            command.extend(["--back_prompt", back_prompt])
        if trajectory:
            command.extend(["--traj", trajectory])

        input_errors = _input_errors(image, prompt, trajectory, required=not plan_only)
        package_root = Path(__file__).resolve().parents[3]
        required_runtime_files = (
            runtime_root() / "scripts" / "infer.py",
            config_path,
            default_config_path,
            package_root
            / "base_models"
            / "three_dimensions"
            / "depth"
            / "depth_anything"
            / "depth_anything_v3"
            / "api.py",
            package_root
            / "base_models"
            / "three_dimensions"
            / "point_clouds"
            / "sharp"
            / "__init__.py",
            package_root
            / "base_models"
            / "three_dimensions"
            / "general_3d"
            / "geocalib"
            / "autolevel.py",
            package_root
            / "base_models"
            / "diffusion_model"
            / "schedulers"
            / "__init__.py",
            package_root
            / "base_models"
            / "diffusion_model"
            / "models"
            / "autoencoders"
            / "wan"
            / "resident.py",
            package_root
            / "base_models"
            / "diffusion_model"
            / "models"
            / "encoders"
            / "wan"
            / "resident.py",
            package_root
            / "base_models"
            / "diffusion_model"
            / "models"
            / "networks"
            / "wan"
            / "variants"
            / "moverse"
            / "__init__.py",
            package_root
            / "base_models"
            / "diffusion_model"
            / "loaders"
            / "wan_variant.py",
            package_root
            / "base_models"
            / "diffusion_model"
            / "models"
            / "autoencoders"
            / "wan"
            / "variants"
            / "taehv.py",
        )
        runtime_errors = [
            f"in-tree MoVerse runtime dependency is missing: {path}"
            for path in required_runtime_files
            if not path.is_file()
        ]
        plan_path = run_dir / "moverse_runtime_plan.json"
        plan = {
            "schema_version": "worldfoundry-moverse-runtime-plan-v1",
            "model_id": self.model_id,
            "source_repo": OFFICIAL_SOURCE_REPO,
            "source_revision": UPSTREAM_REVISION,
            "runtime_root": str(runtime_root()),
            "command": command,
            "image": image,
            "prompt": prompt,
            "back_prompt": back_prompt,
            "trajectory": trajectory,
            "num_frames": num_frames,
            "fps": output_fps,
            "output_path": str(output),
            "stage_dir": str(stage_dir),
            "assets": unresolved_assets,
            "resolved_assets": resolved_assets,
            "missing_requirements": [*input_errors, *asset_errors, *runtime_errors],
        }
        plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
        if plan_only:
            return {
                "status": "prepared",
                "model_id": self.model_id,
                "artifact_kind": "runtime_profile_plan",
                "artifact_path": str(plan_path),
                "plan_path": str(plan_path),
                "command": command,
                "runtime": "worldfoundry.moverse.in_tree.plan",
                "metadata": plan,
            }

        errors = [*input_errors, *asset_errors, *runtime_errors]
        if errors:
            return {
                "status": "failed",
                "model_id": self.model_id,
                "artifact_kind": "generated_world",
                "artifact_path": str(output),
                "plan_path": str(plan_path),
                "command": command,
                "runtime": "worldfoundry.moverse.in_tree",
                "error": "; ".join(errors),
                "metadata": plan,
            }

        env = {
            "WAN_MODEL_PATH": resolved_assets["wan_path"],
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
        if self.device.startswith("cuda:"):
            env["CUDA_VISIBLE_DEVICES"] = self.device.split(":", 1)[1]
        result = execute_in_tree(
            command,
            cwd=runtime_root(),
            output_path=output,
            search_roots=(stage_dir,),
            env=env,
            python_paths=(runtime_root(), Path(__file__).resolve().parents[4]),
            preferred_names=("video.mp4", output.name),
        )
        side_by_side = stage_dir / "video_side_by_side.mp4"
        gaussian = next(stage_dir.glob("*_pano.ply"), None)
        return {
            **result,
            "model_id": self.model_id,
            "artifact_kind": "generated_world",
            "plan_path": str(plan_path),
            "runtime": "worldfoundry.moverse.in_tree",
            "world_artifact_path": str(gaussian) if gaussian else None,
            "comparison_video_path": str(side_by_side) if side_by_side.is_file() else None,
            "metadata": {**plan, **dict(result.get("metadata") or {})},
        }


def _materialize_image(images: Any, run_dir: Path) -> str:
    value = images[0] if isinstance(images, Sequence) and not isinstance(images, (str, bytes, bytearray)) and images else images
    if value is None:
        return ""
    if isinstance(value, (str, Path)):
        return str(Path(value).expanduser().resolve())
    save = getattr(value, "save", None)
    if callable(save):
        path = run_dir / "input.png"
        save(path)
        return str(path)
    return str(value)


def _trajectory_path(options: Mapping[str, Any], interactions: Any) -> str:
    value = options.get("trajectory_path") or options.get("trajectory") or options.get("traj")
    if not value and isinstance(interactions, Mapping):
        value = interactions.get("trajectory_path") or interactions.get("trajectory")
    if not value and isinstance(interactions, Sequence) and not isinstance(interactions, (str, bytes, bytearray)):
        for item in interactions:
            if isinstance(item, Mapping):
                value = item.get("trajectory_path") or item.get("trajectory") or item.get("traj")
            elif isinstance(item, (str, Path)):
                value = item
            if value:
                break
    return str(Path(value).expanduser().resolve()) if value else ""


def _resolve_assets(values: Mapping[str, str], *, required: bool) -> tuple[dict[str, str], list[str]]:
    resolved = dict(values)
    errors: list[str] = []
    for role, value in values.items():
        try:
            resolved[role] = str(
                resolve_local_hf_model_path(value, required_files=REQUIRED_ASSET_FILES.get(role, ()))
            )
        except FileNotFoundError as exc:
            if required:
                errors.append(str(exc).splitlines()[0])
    return resolved, errors


def _input_errors(image: str, prompt: str, trajectory: str, *, required: bool) -> list[str]:
    if not required:
        return []
    errors: list[str] = []
    if not image or not Path(image).is_file():
        errors.append(f"input image not found: {image or '<missing>'}")
    if not prompt.strip():
        errors.append("MoVerse requires a non-empty prompt")
    if trajectory and not Path(trajectory).is_file():
        errors.append(f"camera trajectory not found: {trajectory}")
    return errors


def _output_path(value: str | Path | None, default: Path) -> Path:
    if value is None:
        return default.resolve()
    path = Path(value).expanduser()
    return (path / "moverse.mp4" if not path.suffix else path).resolve()


__all__ = [
    "DEFAULT_DA3_REPO",
    "DEFAULT_FLUX_REPO",
    "DEFAULT_MOVERSE_REPO",
    "DEFAULT_WAN_REPO",
    "MoVerseSynthesis",
    "OFFICIAL_SOURCE_REPO",
    "UPSTREAM_REVISION",
    "runtime_root",
]
