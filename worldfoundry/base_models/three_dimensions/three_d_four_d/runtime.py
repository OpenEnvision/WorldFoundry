"""Subprocess adapter for the in-tree 3D/4D model runtimes.

Each runtime keeps its upstream command-line entrypoint as a module of a vendored
package under ``worldfoundry`` and runs it as ``python -m`` in the model's
environment.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from worldfoundry.core.execution.process import package_module_command
from worldfoundry.core.io.paths import REPO_ROOT, package_data_path
from worldfoundry.synthesis.base_synthesis import BaseSynthesis

_PACKAGE_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class ThreeDFourDRuntimeSpec:
    model_id: str
    display_name: str
    entrypoint: str
    command_kind: str
    artifact_kind: str
    artifact_filename: str
    required_input: str = "prompt_or_image"
    default_config: str | None = None
    source_repo_url: str = ""
    input_schema: Mapping[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()


_CAMERA_INPUT_SCHEMA = {
    "prompt": True,
    "image": True,
    "video": True,
    "actions": ["camera_path", "camera_pose_sequence", "navigation_path"],
}


THREE_D_FOUR_D_RUNTIME_SPECS: Mapping[str, ThreeDFourDRuntimeSpec] = {
    "lagernvs": ThreeDFourDRuntimeSpec(
        model_id="lagernvs",
        display_name="LagrNVS",
        entrypoint="minimal_inference.py",
        command_kind="lagernvs_minimal",
        artifact_kind="generated_video",
        artifact_filename="lagernvs.mp4",
        required_input="image",
        source_repo_url="https://github.com/facebookresearch/lagernvs",
        input_schema={"prompt": False, "image": True, "video": False, "actions": ["camera_path"]},
    ),
    "mvdiffusion": ThreeDFourDRuntimeSpec(
        model_id="mvdiffusion",
        display_name="MVDiffusion",
        entrypoint="demo.py",
        command_kind="mvdiffusion_demo",
        artifact_kind="generated_image",
        artifact_filename="mvdiffusion.png",
        required_input="prompt_or_image",
        source_repo_url="https://github.com/Tangshitao/MVDiffusion",
        input_schema={"prompt": True, "image": True, "video": False, "actions": ["camera_views"]},
    ),
    "wonderjourney": ThreeDFourDRuntimeSpec(
        model_id="wonderjourney",
        display_name="WonderJourney",
        entrypoint="run.py",
        command_kind="wonderjourney_run",
        artifact_kind="generated_world",
        artifact_filename="wonderjourney.mp4",
        required_input="config",
        default_config=str(package_data_path("models", "runtime", "configs", "wonderjourney", "village.yaml")),
        source_repo_url="https://github.com/KovenYu/WonderJourney",
        input_schema=_CAMERA_INPUT_SCHEMA,
    ),
    "worldgen": ThreeDFourDRuntimeSpec(
        model_id="worldgen",
        display_name="WorldGen",
        entrypoint="inference.py",
        command_kind="worldgen_inference",
        artifact_kind="generated_3d_asset",
        artifact_filename="worldgen.ply",
        required_input="prompt_or_image",
        source_repo_url="https://github.com/ZiYang-xie/WorldGen",
        input_schema={"prompt": True, "image": True, "video": False, "actions": ["camera_path"]},
    ),
}


IN_TREE_RUNTIME_DIRS: Mapping[str, tuple[str, ...]] = {
    "lagernvs": ("base_models", "three_dimensions", "general_3d", "lagernvs", "lagernvs_runtime"),
    "mvdiffusion": ("base_models", "three_dimensions", "general_3d", "mvdiffusion", "mvdiffusion_runtime"),
    "wonderjourney": ("synthesis", "visual_generation", "wonderjourney", "wonderjourney_runtime"),
    "worldgen": ("synthesis", "visual_generation", "worldgen", "worldgen_runtime"),
}


def runtime_dir(spec: ThreeDFourDRuntimeSpec) -> Path:
    """Directory of the vendored runtime package that holds ``spec.entrypoint``."""
    return _PACKAGE_ROOT.joinpath(*IN_TREE_RUNTIME_DIRS[spec.model_id])


def runtime_module(spec: ThreeDFourDRuntimeSpec) -> str:
    """Dotted package module of ``spec.entrypoint``."""
    return ".".join(("worldfoundry", *IN_TREE_RUNTIME_DIRS[spec.model_id], Path(spec.entrypoint).stem))


class ThreeDFourDRuntimeSynthesis(BaseSynthesis):
    """Subprocess adapter for 3D/4D repository inference entrypoints."""

    MODEL_ID: str | None = None

    def __init__(
        self,
        *,
        spec: ThreeDFourDRuntimeSpec,
        device: str = "cuda",
        options: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.profile = spec
        self.model_id = spec.model_id
        self.model_name = spec.display_name
        self.runtime_dir = runtime_dir(spec)
        self.module = runtime_module(spec)
        self.device = device
        self.options = dict(options or {})

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Any = None,
        args: Any = None,
        device: str | None = None,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> ThreeDFourDRuntimeSynthesis:
        del args
        options = dict(pretrained_model_path) if isinstance(pretrained_model_path, Mapping) else {}
        options.update(kwargs)
        resolved_model_id = str(options.get("model_id") or options.get("profile_id") or model_id or cls.MODEL_ID or "")
        if not resolved_model_id:
            raise ValueError("ThreeDFourDRuntimeSynthesis requires model_id/profile_id.")
        spec = three_d_four_d_runtime_spec(resolved_model_id)
        return cls(
            spec=spec,
            device=str(device or options.get("device") or "cuda"),
            options=options,
        )

    def predict(
        self,
        prompt: str = "",
        images: Any = None,
        video: Any = None,
        interactions: Any = None,
        output_path: str | Path | None = None,
        fps: int | None = None,
        timeout_seconds: int = 21600,
        plan_only: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        run_dir = Path(kwargs.pop("run_dir", "") or tempfile.mkdtemp(prefix=f"{_safe_name(self.model_id)}_"))
        run_dir = run_dir.expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        output = _resolve_output(output_path, run_dir, self.spec.artifact_filename)
        output = _coerce_model_artifact_output(output, self.spec)
        media = _materialize_media(images=images, video=video, run_dir=run_dir)
        context = {
            "prompt": prompt,
            "images": media["images"],
            "video": media["video"],
            "interactions": _jsonable(interactions or []),
            "run_dir": str(run_dir),
            "output_path": str(output),
            "output_dir": str(output.parent),
            "fps": fps,
            "device": self.device,
            "options": _jsonable({**self.options, **kwargs}),
            "module": self.module,
        }
        command = self._build_command(context, kwargs)
        plan_path = run_dir / "subprocess_runtime_plan.json"
        plan_payload = {
            "schema_version": "worldfoundry-three-d-four-d-runtime-plan",
            "model_id": self.model_id,
            "display_name": self.model_name,
            "source_repo_url": self.spec.source_repo_url,
            "module": self.module,
            "command": command,
            "context": context,
        }
        _write_json(plan_path, plan_payload)
        if plan_only:
            requested_plan_path = _plan_output_path(output, self.model_id)
            if requested_plan_path != plan_path:
                _write_json(requested_plan_path, plan_payload)
            return self._prepared(run_dir, requested_plan_path, command)

        missing = self._missing_runtime_requirements(context)
        if missing:
            return {
                "status": "failed",
                "model_id": self.model_id,
                "artifact_kind": self.spec.artifact_kind,
                "run_dir": str(run_dir),
                "plan_path": str(plan_path),
                "command": command,
                "runtime": "worldfoundry.three_d_four_d.subprocess",
                "backend_quality": "model_entrypoint",
                "error": "; ".join(missing),
                "metadata": plan_payload,
            }

        log_path = run_dir / "subprocess_runtime.log"
        env = self._subprocess_env(kwargs)
        artifacts_before = _artifact_snapshot((run_dir, output.parent))
        started_at = time.time()
        try:
            completed = subprocess.run(
                command,
                cwd=str(run_dir),
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout_seconds,
                check=False,
            )
        except Exception as exc:  # pragma: no cover - runtime/environment dependent.
            return {
                "status": "failed",
                "model_id": self.model_id,
                "artifact_kind": self.spec.artifact_kind,
                "run_dir": str(run_dir),
                "plan_path": str(plan_path),
                "command": command,
                "runtime": "worldfoundry.three_d_four_d.subprocess",
                "backend_quality": "model_entrypoint",
                "error": f"{type(exc).__name__}: {exc}",
                "metadata": plan_payload,
            }
        log_path.write_text(completed.stdout or "", encoding="utf-8")
        if completed.returncode != 0:
            return {
                "status": "failed",
                "model_id": self.model_id,
                "artifact_kind": self.spec.artifact_kind,
                "run_dir": str(run_dir),
                "plan_path": str(plan_path),
                "command": command,
                "runtime": "worldfoundry.three_d_four_d.subprocess",
                "backend_quality": "model_entrypoint",
                "error": f"runtime exited with code {completed.returncode}; see {log_path}",
                "metadata": {**plan_payload, "log_path": str(log_path)},
            }

        artifact_path = _select_artifact(output, run_dir, since=started_at, known_before=artifacts_before)
        if artifact_path is None:
            return {
                "status": "failed",
                "model_id": self.model_id,
                "artifact_kind": self.spec.artifact_kind,
                "run_dir": str(run_dir),
                "plan_path": str(plan_path),
                "command": command,
                "runtime": "worldfoundry.three_d_four_d.subprocess",
                "backend_quality": "model_entrypoint",
                "error": f"runtime completed without a fresh artifact; see {log_path}",
                "metadata": {**plan_payload, "log_path": str(log_path)},
            }
        if artifact_path.exists() and artifact_path.is_file() and artifact_path.resolve() != output.resolve():
            output.parent.mkdir(parents=True, exist_ok=True)
            target = output
            if artifact_path.suffix.lower() and artifact_path.suffix.lower() != output.suffix.lower():
                target = output.with_suffix(artifact_path.suffix)
            if artifact_path.resolve() != target.resolve():
                shutil.copy2(artifact_path, target)
                artifact_path = target
        result = {
            "status": "succeeded",
            "model_id": self.model_id,
            "artifact_kind": self.spec.artifact_kind,
            "artifact_path": str(artifact_path),
            "run_dir": str(run_dir),
            "plan_path": str(plan_path),
            "command": command,
            "runtime": "worldfoundry.three_d_four_d.subprocess",
            "backend_quality": "model_entrypoint",
            "metadata": {**plan_payload, "log_path": str(log_path)},
        }
        return result

    def _prepared(self, run_dir: Path, plan_path: Path, command: Sequence[str]) -> dict[str, Any]:
        return {
            "status": "prepared",
            "model_id": self.model_id,
            "artifact_kind": "runtime_profile_plan",
            "artifact_path": str(plan_path),
            "run_dir": str(run_dir),
            "plan_path": str(plan_path),
            "command": list(command),
            "runtime": "worldfoundry.three_d_four_d.subprocess.plan",
            "backend_quality": "plan",
            "metadata": {"module": self.module},
        }

    def _missing_runtime_requirements(self, context: Mapping[str, Any]) -> list[str]:
        missing: list[str] = []
        if not (self.runtime_dir / self.spec.entrypoint).is_file():
            missing.append(f"runtime entrypoint missing: {self.module}")
        if self.spec.required_input == "prompt" and not str(context.get("prompt") or "").strip():
            missing.append("prompt input required")
        if self.spec.required_input == "image" and not context.get("images"):
            missing.append("image input required")
        if self.spec.required_input == "video" and not context.get("video"):
            missing.append("video input required")
        if self.spec.required_input == "image_or_video" and not (context.get("images") or context.get("video")):
            missing.append("image or video input required")
        if self.spec.required_input == "prompt_or_image" and not (
            str(context.get("prompt") or "").strip() or context.get("images")
        ):
            missing.append("prompt or image input required")
        if self.spec.required_input == "dataset":
            options = context.get("options") if isinstance(context.get("options"), Mapping) else {}
            data_dir = (
                options.get("data_dir")
                or options.get("source_path")
                or options.get("dataset_path")
                or options.get("data_root")
            )
            if not str(data_dir or "").strip():
                missing.append(
                    "dataset source path required; pass data_dir/source_path/dataset_path in runtime options"
                )
        return missing

    def _subprocess_env(self, options: Mapping[str, Any] | None = None) -> dict[str, str]:
        env = os.environ.copy()
        options = options or {}
        env.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        if self.device:
            env.setdefault("WORLDFOUNDRY_DEVICE", self.device)
        gpu = options.get("gpu", self.options.get("gpu"))
        if gpu is not None and str(gpu).strip():
            env["CUDA_VISIBLE_DEVICES"] = str(gpu).strip()
        existing_da2_model_path = env.get("WORLDFOUNDRY_DEPTH_ANYTHING2_MODEL_PATH") or env.get(
            "WORLDFOUNDRY_DA2_MODEL_PATH"
        )
        if existing_da2_model_path and not _has_depth_anything2_checkpoint(existing_da2_model_path):
            env.pop("WORLDFOUNDRY_DEPTH_ANYTHING2_MODEL_PATH", None)
            env.pop("WORLDFOUNDRY_DA2_MODEL_PATH", None)
            existing_da2_model_path = ""
        da2_model_path = existing_da2_model_path or _depth_anything2_model_path()
        if da2_model_path:
            env["WORLDFOUNDRY_DEPTH_ANYTHING2_MODEL_PATH"] = str(da2_model_path)
            env["WORLDFOUNDRY_DA2_MODEL_PATH"] = str(da2_model_path)
        ckpt_root = _ckpt_root()
        env.setdefault("TORCH_HOME", str(ckpt_root / "torch_hub"))
        hf_home_candidates = [
            Path(str(env.get("WORLDFOUNDRY_HFD_ROOT") or "")).expanduser(),
            Path(str(env.get("HF_HOME") or "")).expanduser(),
            ckpt_root / "hfd",
            ckpt_root / "huggingface",
        ]
        hf_home = next(
            (candidate for candidate in hf_home_candidates if str(candidate) != "." and candidate.exists()), None
        )
        if hf_home is not None:
            env["HF_HOME"] = str(hf_home)
            env["HUGGINGFACE_HUB_CACHE"] = str(hf_home / "hub")
            env["TRANSFORMERS_CACHE"] = str(hf_home / "hub")
        env.setdefault("WORLDFOUNDRY_CKPT_DIR", str(ckpt_root))
        return env

    def _build_command(self, context: Mapping[str, Any], options: Mapping[str, Any]) -> list[str]:
        python = str(options.get("python_executable") or sys.executable)
        if os.getenv("WORLDFOUNDRY_STUDIO_CONDA_CHILD", "").strip().lower() in {"1", "true", "yes", "on"}:
            python = sys.executable
        prompt = str(context.get("prompt") or "")
        output_path = str(context["output_path"])
        output_dir = str(context["output_dir"])
        images = [str(item) for item in context.get("images") or []]
        config = str(options.get("config") or self.options.get("config") or self.spec.default_config or "")

        if self.spec.command_kind == "lagernvs_minimal":
            length = str(options.get("video_length", self.options.get("video_length", 100)))
            model_repo = _lagernvs_model_repo(
                str(options.get("model_repo", self.options.get("model_repo", "facebook/lagernvs_general_512")))
            )
            image_args = _expand_image_inputs(images)
            arguments = [
                "--images",
                *image_args,
                "--video_length",
                length,
                "--output",
                output_path,
                "--model_repo",
                model_repo,
            ]
        elif self.spec.command_kind == "worldgen_inference":
            arguments = ["--prompt", prompt or "a 3D scene", "--output_dir", output_dir]
            resolution = options.get("resolution", self.options.get("resolution"))
            if resolution is not None:
                arguments.extend(["--resolution", str(int(resolution))])
            pano_image = options.get("pano_image", self.options.get("pano_image"))
            if pano_image:
                arguments.extend(["--pano_image", str(pano_image)])
            expanded_images = _expand_image_inputs(images)
            if expanded_images and not pano_image:
                arguments.extend(["--image", expanded_images[0]])
            if not _bool_option(options, self.options, "viewer"):
                arguments.append("--no_viewer")
            if _bool_option(options, self.options, "return_mesh"):
                arguments.append("--return_mesh")
            if _bool_option(options, self.options, "save_scene", default=True):
                arguments.append("--save_scene")
            if _bool_option(options, self.options, "low_vram"):
                arguments.append("--low_vram")
        elif self.spec.command_kind == "wonderjourney_run":
            config = _prepare_wonder_config(
                self.spec.command_kind,
                self.runtime_dir,
                config,
                Path(output_dir),
                options,
                prompt=prompt,
                images=images,
            )
            arguments = ["--example_config", config]
        elif self.spec.command_kind == "mvdiffusion_demo":
            arguments = []
            if prompt:
                arguments.extend(["--text", prompt])
            if images:
                arguments.extend(["--image_path", images[0]])
            text_path = options.get("text_path", self.options.get("text_path"))
            if text_path:
                resolved_text_path = Path(str(text_path)).expanduser()
                if not resolved_text_path.is_absolute():
                    resolved_text_path = self.runtime_dir / resolved_text_path
                if resolved_text_path.is_file():
                    arguments.extend(["--text_path", str(resolved_text_path.resolve())])
            steps = options.get("steps", self.options.get("steps"))
            if steps is not None:
                arguments.extend(["--steps", str(max(int(steps), 1))])
            if _bool_option(options, self.options, "gen_video"):
                arguments.append("--gen_video")
        else:
            arguments = []
        return package_module_command(self.module, arguments, python_executable=python)


def three_d_four_d_runtime_spec(model_id: str) -> ThreeDFourDRuntimeSpec:
    key = model_id.lower().replace("_", "-")
    try:
        return THREE_D_FOUR_D_RUNTIME_SPECS[key]
    except KeyError as exc:
        known = ", ".join(sorted(THREE_D_FOUR_D_RUNTIME_SPECS))
        raise KeyError(f"Unknown 3D/4D runtime {model_id!r}. Known ids: {known}") from exc


def _materialize_media(*, images: Any, video: Any, run_dir: Path) -> dict[str, Any]:
    image_paths: list[str] = []
    if images is not None:
        values = images if isinstance(images, (list, tuple)) else [images]
        for index, item in enumerate(values):
            path = _coerce_media_item(item, run_dir, f"input_image_{index:02d}.png")
            if path:
                image_paths.append(path)
    return {
        "images": image_paths,
        "video": _coerce_media_item(video, run_dir, "input_video.mp4") if video is not None else "",
    }


def _coerce_media_item(value: Any, run_dir: Path, filename: str) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, Path)):
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        return str(path)
    save = getattr(value, "save", None)
    if callable(save):
        path = run_dir / filename
        save(path)
        return str(path.resolve())
    return str(value)


def _resolve_output(output_path: str | Path | None, run_dir: Path, filename: str) -> Path:
    if output_path is None:
        return (run_dir / filename).resolve()
    path = Path(output_path).expanduser()
    if path.suffix:
        return path.resolve()
    return (path / filename).resolve()


def _coerce_model_artifact_output(output: Path, spec: ThreeDFourDRuntimeSpec) -> Path:
    artifact_suffix = Path(spec.artifact_filename).suffix.lower()
    if not artifact_suffix or output.suffix.lower() == artifact_suffix:
        return output
    if spec.command_kind == "worldgen_inference":
        return output.with_name(Path(spec.artifact_filename).name)
    return output


def _plan_output_path(output: Path, model_id: str) -> Path:
    if output.suffix.lower() == ".json":
        return output
    return output.with_name(f"{output.stem or _safe_name(model_id)}_subprocess_runtime_plan.json")


def _image_data_dir(images: Sequence[str], run_dir: Path) -> str:
    if not images:
        return str(run_dir)
    if len(images) == 1:
        path = Path(images[0])
        return str(path.parent if path.parent != Path("") else run_dir)
    return str(Path(images[0]).parent)


def _lagernvs_model_repo(value: str) -> str:
    text = str(value or "").strip()
    if not text or text == "facebook/lagernvs_general_512":
        return "facebook/lagernvs_general_512"
    path = Path(text)
    if path.expanduser().exists():
        return str(path.expanduser())
    normalized = text.replace("\\", "/")
    if "lagernvs_general_512" in normalized or "lagernvs-general-512" in normalized:
        return "facebook/lagernvs_general_512"
    return text


def _ckpt_root() -> Path:
    return Path(os.environ.get("WORLDFOUNDRY_CKPT_DIR", REPO_ROOT.parent / "ckpt")).expanduser()


def _has_depth_anything2_checkpoint(path: str | Path, encoder: str = "vitl") -> bool:
    candidate = Path(path).expanduser()
    checkpoint_name = f"depth_anything_v2_{encoder}.pth"
    if candidate.is_file():
        return candidate.name == checkpoint_name or candidate.name.startswith("depth_anything_v2_")
    return (candidate / checkpoint_name).is_file() or (candidate / "checkpoints" / checkpoint_name).is_file()


def _depth_anything2_model_path(encoder: str = "vitl") -> str | None:
    ckpt_root = _ckpt_root()
    for candidate in (
        ckpt_root / "Depth-Anything-V2-Large",
        ckpt_root / "Prior-Depth-Anything",
        ckpt_root / "DA-2",
        ckpt_root / "hfd" / "depth-anything--Depth-Anything-V2-Large",
        ckpt_root / "hfd" / "depth-anything--Prior-Depth-Anything",
    ):
        if _has_depth_anything2_checkpoint(candidate, encoder=encoder):
            return str(candidate)
    return None


def _first_existing_ckpt(*parts: str) -> str | None:
    root = _ckpt_root()
    for part in parts:
        candidate = root / part
        if candidate.exists():
            return str(candidate)
    return None


def _prepare_wonder_config(
    command_kind: str,
    runtime_dir: Path,
    config: str,
    output_dir: Path,
    options: Mapping[str, Any],
    *,
    prompt: str = "",
    images: Sequence[str] | None = None,
) -> str:
    config_path = Path(config)
    if not config_path.is_absolute():
        config_path = runtime_dir / config_path
    if not config_path.is_file():
        return str(config)

    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    data["runs_dir"] = str(output_dir)
    data["debug"] = False
    data["use_gpt"] = False
    data["regenerate_times"] = 0
    data["enable_regenerate"] = False
    data["stable_diffusion_checkpoint"] = (
        _first_existing_ckpt("stable-diffusion-2-inpainting", "sd2-community--stable-diffusion-2-inpainting")
        or data.get("stable_diffusion_checkpoint")
        or "stabilityai/stable-diffusion-2-inpainting"
    )

    def _runtime_image_path(value: object) -> str:
        if not value:
            return ""
        path = Path(str(value)).expanduser()
        if path.is_absolute():
            return str(path)
        for root in (REPO_ROOT, Path.cwd()):
            candidate = (root / path).resolve()
            if candidate.exists():
                return str(candidate)
        return str((REPO_ROOT / path).resolve())

    image_path = (
        options.get("image_filepath")
        or (images[0] if images else "")
        or options.get("image_path")
        or options.get("input_path")
        or data.get("image_filepath")
        or _default_wonderjourney_image()
    )
    if image_path:
        data["image_filepath"] = _runtime_image_path(image_path)
    data["content_prompt"] = str(
        options.get("content_prompt") or prompt or data.get("content_prompt") or "village, cottages, trees"
    )
    data["style_prompt"] = str(options.get("style_prompt") or data.get("style_prompt") or "photorealistic")
    data["negative_prompt"] = str(
        options.get("negative_prompt") or data.get("negative_prompt") or "low quality, blurry, distorted"
    )
    data["num_scenes"] = int(options.get("num_scenes", 1))
    data["num_keyframes"] = int(options.get("num_keyframes", 2))
    data["frames"] = int(options.get("frames", 2))
    data["save_fps"] = int(options.get("save_fps", 10))
    data["skip_interp"] = bool(options.get("skip_interp", False))
    data.setdefault("rotation_path", [0, 0, 0, 1])

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_config = output_dir / f"{command_kind}_worldfoundry_demo.yaml"
    generated_config.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return str(generated_config)


def _default_wonderjourney_image() -> str:
    for candidate in (
        REPO_ROOT / "worldfoundry" / "data" / "test_cases" / "hunyuan_game_craft" / "village.png",
        REPO_ROOT / "worldfoundry" / "data" / "test_cases" / "studio_demo" / "00" / "image.jpg",
        REPO_ROOT / "worldfoundry" / "data" / "test_cases" / "wonderjourney" / "assets" / "logo.png",
    ):
        if candidate.is_file():
            return str(candidate)
    return ""


def _expand_image_inputs(images: Sequence[str]) -> list[str]:
    expanded: list[str] = []
    for item in images:
        path = Path(item)
        if path.is_dir():
            expanded.extend(
                str(candidate)
                for candidate in sorted(path.iterdir())
                if candidate.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
            )
        elif item:
            expanded.append(str(path))
    return expanded


def _bool_option(options: Mapping[str, Any], defaults: Mapping[str, Any], key: str, default: bool = False) -> bool:
    value = options.get(key, defaults.get(key, default))
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _select_artifact(
    output: Path,
    run_dir: Path,
    *,
    since: float | None = None,
    known_before: Mapping[str, tuple[int, int]] | None = None,
) -> Path | None:
    def is_fresh(path: Path) -> bool:
        if not path.is_file():
            return False
        stat = path.stat()
        if since is not None and stat.st_mtime < int(since):
            return False
        if known_before is not None:
            return known_before.get(str(path.resolve())) != (stat.st_mtime_ns, stat.st_size)
        return True

    if is_fresh(output):
        return output
    search_dirs = [run_dir, output.parent]
    suffixes = (".mp4", ".glb", ".splat", ".ply", ".png", ".json")
    suffixes = tuple(dict.fromkeys((output.suffix.lower(), *suffixes)))
    for suffix in suffixes:
        matches = sorted(
            (match for directory in search_dirs for match in directory.rglob(f"*{suffix}") if is_fresh(match)),
            key=lambda item: item.stat().st_mtime,
        )
        if matches:
            return matches[-1]
    return None


def _artifact_snapshot(directories: Sequence[Path]) -> dict[str, tuple[int, int]]:
    suffixes = {".mp4", ".glb", ".splat", ".ply", ".png", ".json"}
    snapshot: dict[str, tuple[int, int]] = {}
    for directory in directories:
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix.lower() in suffixes:
                stat = path.stat()
                snapshot[str(path.resolve())] = (stat.st_mtime_ns, stat.st_size)
    return snapshot


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in value).strip("-") or "item"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


__all__ = [
    "IN_TREE_RUNTIME_DIRS",
    "THREE_D_FOUR_D_RUNTIME_SPECS",
    "ThreeDFourDRuntimeSpec",
    "ThreeDFourDRuntimeSynthesis",
    "runtime_dir",
    "runtime_module",
    "three_d_four_d_runtime_spec",
]
