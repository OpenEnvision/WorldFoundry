"""Offline, in-tree subprocess adapter for Uni3C inference."""

from __future__ import annotations

import json
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from worldfoundry.core.execution.process import package_module_command
from worldfoundry.core.io.paths import resolve_local_hf_model_path
from worldfoundry.runtime.in_tree_cli import execute_in_tree
from worldfoundry.synthesis.base_synthesis import BaseSynthesis

OFFICIAL_SOURCE_REPO = "https://github.com/alibaba-damo-academy/Uni3C"
UPSTREAM_REVISION = "75ed6e2180316b7f07398e4e88ea8bdba3e6970c"
DEFAULT_CONTROLLER_REPO = "ewrfcas/Uni3C"
DEFAULT_CAMERA_BASE_REPO = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
DEFAULT_UNIFIED_MODEL_REPO = "theFoxofSky/RealisDance-DiT"


def runtime_root() -> Path:
    """Return the vendored, infer-only Uni3C source root."""

    return Path(__file__).resolve().parent / "uni3c_runtime"


class Uni3CSynthesis(BaseSynthesis):
    """Execute Uni3C against pre-staged local Hugging Face snapshots.

    ``camera`` mode wraps the official PCDController entrypoint. ``unified``
    mode wraps camera plus human-pose control. Both consume the official
    pre-rendered control bundle instead of silently running preprocessing.
    """

    def __init__(self, *, device: str = "cuda", options: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        self.device = device
        self.options = dict(options or {})
        self.model_id = "uni3c"
        self.model_name = "Uni3C"

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Any = None,
        args: Any = None,
        device: str | None = None,
        **kwargs: Any,
    ) -> Uni3CSynthesis:
        del args
        options = dict(pretrained_model_path) if isinstance(pretrained_model_path, Mapping) else {}
        if pretrained_model_path is not None and not isinstance(pretrained_model_path, Mapping):
            options["controller_path"] = str(pretrained_model_path)
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
        mode = _normalize_mode(options.get("mode", "camera"))
        run_dir = Path(options.get("run_dir") or tempfile.mkdtemp(prefix="uni3c_")).expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        output = _output_path(output_path, run_dir / "uni3c.mp4")
        reference_image = _materialize_image(images, run_dir)
        render_path = _render_path(options, interactions)
        num_frames = int(options.get("num_frames", options.get("nframe", 81)))
        max_area = int(options.get("max_area", 480 * 768 if mode == "camera" else 768 * 768))
        seed = int(options.get("seed", 1024))
        output_fps = int(fps or options.get("fps", 16))
        num_gpus = int(options.get("num_gpus", 1))

        unresolved_assets = {
            "controller_path": str(options.get("controller_path") or DEFAULT_CONTROLLER_REPO),
            "base_model_path": str(options.get("base_model_path") or DEFAULT_CAMERA_BASE_REPO),
            "model_path": str(options.get("model_path") or DEFAULT_UNIFIED_MODEL_REPO),
        }
        resolved_assets, asset_errors = _resolve_assets(unresolved_assets, mode=mode, required=not plan_only)
        command = self._command(
            mode=mode,
            reference_image=reference_image,
            render_path=render_path,
            output=output,
            prompt=prompt,
            num_frames=num_frames,
            max_area=max_area,
            seed=seed,
            fps=output_fps,
            num_gpus=num_gpus,
            assets=resolved_assets,
            options=options,
        )
        input_errors = _input_errors(reference_image, render_path, mode=mode, required=not plan_only)
        plan_path = run_dir / "uni3c_runtime_plan.json"
        plan = {
            "schema_version": "worldfoundry-uni3c-runtime-plan-v1",
            "model_id": self.model_id,
            "mode": mode,
            "source_repo": OFFICIAL_SOURCE_REPO,
            "source_revision": UPSTREAM_REVISION,
            "runtime_root": str(runtime_root()),
            "command": command,
            "reference_image": reference_image,
            "render_path": render_path,
            "output_path": str(output),
            "assets": unresolved_assets,
            "resolved_assets": resolved_assets,
            "missing_requirements": [*input_errors, *asset_errors],
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
                "runtime": "worldfoundry.uni3c.in_tree.plan",
                "metadata": plan,
            }

        errors = [*input_errors, *asset_errors]
        if errors:
            return _failed(plan_path, output, command, errors, plan)

        result = execute_in_tree(
            command,
            cwd=runtime_root(),
            output_path=output,
            search_roots=(output.parent,),
            env={
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
            },
            preferred_names=(output.name, "result.mp4"),
        )
        return {
            **result,
            "model_id": self.model_id,
            "artifact_kind": "generated_video",
            "plan_path": str(plan_path),
            "runtime": "worldfoundry.uni3c.in_tree",
            "metadata": {**plan, **dict(result.get("metadata") or {})},
        }

    def _command(
        self,
        *,
        mode: str,
        reference_image: str,
        render_path: str,
        output: Path,
        prompt: str,
        num_frames: int,
        max_area: int,
        seed: int,
        fps: int,
        num_gpus: int,
        assets: Mapping[str, str],
        options: Mapping[str, Any],
    ) -> list[str]:
        python = str(options.get("python_executable") or sys.executable)
        module = "worldfoundry.synthesis.visual_generation.uni3c.uni3c_runtime." + (
            "cam_control" if mode == "camera" else "uni3c_inference"
        )
        command = package_module_command(
            module,
            python_executable=python,
            nproc_per_node=num_gpus if num_gpus > 1 else None,
            launcher_options=(),
        )
        controller = Path(assets["controller_path"])
        command.extend(
            [
                "--reference_image",
                reference_image,
                "--render_path",
                render_path,
                "--output_path",
                str(output),
                "--prompt",
                prompt,
                "--nframe",
                str(num_frames),
                "--max_area",
                str(max_area),
                "--seed",
                str(seed),
                "--fps",
                str(fps),
                "--config_path",
                str(controller / "config.json"),
                "--controlnet_path",
                str(controller / "controlnet.pth"),
                "--base_model_path",
                assets["base_model_path"],
            ]
        )
        if mode == "unified":
            command.extend(["--model_path", assets["model_path"]])
            if options.get("save_gpu_memory"):
                command.append("--save-gpu-memory")
        if num_gpus > 1:
            command.append("--enable_sp")
            if mode == "camera" and options.get("fsdp", True):
                command.append("--fsdp")
        return command


def _normalize_mode(value: Any) -> str:
    mode = str(value or "camera").strip().lower().replace("-", "_")
    aliases = {"pcd": "camera", "pcd_controller": "camera", "human": "unified", "human_motion": "unified"}
    mode = aliases.get(mode, mode)
    if mode not in {"camera", "unified"}:
        raise ValueError("Uni3C mode must be 'camera' or 'unified'.")
    return mode


def _materialize_image(images: Any, run_dir: Path) -> str:
    value = (
        images[0]
        if isinstance(images, Sequence) and not isinstance(images, (str, bytes, bytearray)) and images
        else images
    )
    if value is None:
        return ""
    if isinstance(value, (str, Path)):
        return str(Path(value).expanduser().resolve())
    save = getattr(value, "save", None)
    if callable(save):
        path = run_dir / "reference.png"
        save(path)
        return str(path)
    return str(value)


def _render_path(options: Mapping[str, Any], interactions: Any) -> str:
    value = options.get("render_path") or options.get("control_path") or options.get("condition_path")
    if not value and isinstance(interactions, Mapping):
        value = interactions.get("render_path") or interactions.get("control_path")
    if not value and isinstance(interactions, Sequence) and not isinstance(interactions, (str, bytes, bytearray)):
        for item in interactions:
            if isinstance(item, Mapping):
                value = item.get("render_path") or item.get("control_path") or item.get("condition_path")
            elif isinstance(item, (str, Path)):
                value = item
            if value:
                break
    return str(Path(value).expanduser().resolve()) if value else ""


def _resolve_assets(values: Mapping[str, str], *, mode: str, required: bool) -> tuple[dict[str, str], list[str]]:
    resolved = dict(values)
    errors: list[str] = []
    roles = ["controller_path", "base_model_path"]
    if mode == "unified":
        roles.append("model_path")
    for role in roles:
        value = values[role]
        try:
            required_files = ("config.json", "controlnet.pth") if role == "controller_path" else ()
            resolved[role] = str(resolve_local_hf_model_path(value, required_files=required_files))
        except FileNotFoundError as exc:
            if required:
                errors.append(str(exc).splitlines()[0])
    return resolved, errors


def _input_errors(reference_image: str, render_path: str, *, mode: str, required: bool) -> list[str]:
    if not required:
        return []
    errors: list[str] = []
    if not reference_image or not Path(reference_image).is_file():
        errors.append(f"reference image not found: {reference_image or '<missing>'}")
    root = Path(render_path) if render_path else None
    required_files = ["render.mp4", "render_mask.mp4", "cam_info.json"]
    if mode == "unified":
        required_files.extend(["smpl_render.mp4", "hand_render.mp4"])
    if root is None or not root.is_dir():
        errors.append(f"Uni3C render bundle not found: {render_path or '<missing>'}")
    else:
        errors.extend(
            f"Uni3C render bundle is missing {name}: {root / name}"
            for name in required_files
            if not (root / name).is_file()
        )
    return errors


def _output_path(value: str | Path | None, default: Path) -> Path:
    if value is None:
        return default.resolve()
    path = Path(value).expanduser()
    return (path / "uni3c.mp4" if not path.suffix else path).resolve()


def _failed(
    plan_path: Path, output: Path, command: list[str], errors: list[str], plan: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "status": "failed",
        "model_id": "uni3c",
        "artifact_kind": "generated_video",
        "artifact_path": str(output),
        "plan_path": str(plan_path),
        "command": command,
        "runtime": "worldfoundry.uni3c.in_tree",
        "error": "; ".join(errors),
        "metadata": dict(plan),
    }


__all__ = [
    "DEFAULT_CAMERA_BASE_REPO",
    "DEFAULT_CONTROLLER_REPO",
    "DEFAULT_UNIFIED_MODEL_REPO",
    "OFFICIAL_SOURCE_REPO",
    "UPSTREAM_REVISION",
    "Uni3CSynthesis",
    "runtime_root",
]
