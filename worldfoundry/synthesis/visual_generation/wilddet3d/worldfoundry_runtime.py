"""Run pinned upstream WildDet3D from a local checkout without vendoring its code."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.core.io.paths import checkpoint_root_path, official_runtime_repo_path
from worldfoundry.runtime.assets import expand_worldfoundry_path
from worldfoundry.synthesis.visual_generation.runtime_manifest import command_settings


RUNTIME_DIR = Path(__file__).resolve().parent
OFFICIAL_ENTRYPOINT = RUNTIME_DIR / "infer.py"
OFFICIAL_REPO_URL = "https://github.com/allenai/WildDet3D"
SOURCE_ENV_VAR = "WORLDFOUNDRY_WILDDET3D_SOURCE"
SOURCE_REVISION = "1b8aa52b6ff3f00d0ebfa07175efc0c0c440964a"
SUBMODULE_REVISIONS = {
    "third_party/sam3": "159490bb13108749dd5ceff68f42cd1792e60641",
    "third_party/lingbot_depth": "ff29de5bf433d1b1519cd851da00bc652668964a",
}
DEFAULT_CHECKPOINT = checkpoint_root_path("allenai--WildDet3D", "wilddet3d_alldata_all_prompt_v1.0.pt")
DEFAULT_LINGBOT_CONFIG = checkpoint_root_path("robbyant--lingbot-depth-postrain-dc-vitl14", "model.pt")
BLOCKED_REASON = ""


def runtime_root() -> Path:
    return official_runtime_repo_path("WildDet3D", specific_env=SOURCE_ENV_VAR)


def _option(options: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        value = options.get(name)
        if value not in (None, ""):
            return value
    return default


def _path(options: Mapping[str, Any], *names: str, default: Path | None = None) -> Path | None:
    value = _option(options, *names, default=default)
    return expand_worldfoundry_path(str(value)).resolve() if value is not None else None


def pythonpath_entries(*, runtime_root, options, profile) -> list[str]:
    del options, profile
    # Upstream LingBot-Depth imports utils3d, which is already provided by this tree.
    from worldfoundry.core.io.paths import project_root

    return [str(project_root(__file__) / "worldfoundry/base_models/three_dimensions/general_3d/eastern_journalist")]


def missing_requirements(*, options, runtime_root, entrypoint, profile) -> list[dict[str, str]]:
    del profile
    options = dict(options or {})
    source = Path(runtime_root)
    missing: list[dict[str, str]] = []

    def need(kind: str, path: Path | None, reason: str) -> None:
        if path is None or not path.is_file() or path.stat().st_size == 0:
            missing.append({"kind": kind, "path": str(path or ""), "reason": reason})

    for relative in (
        "wilddet3d/inference.py",
        "third_party/sam3/sam3/model_builder.py",
        "third_party/lingbot_depth/mdm/model/v2.py",
    ):
        need("source_repo", source / relative, f"pinned upstream WildDet3D asset is missing; clone {OFFICIAL_REPO_URL} with submodules and set {SOURCE_ENV_VAR}")
    if source.is_dir() and (source / ".git").exists():
        for relative, expected in {".": SOURCE_REVISION, **SUBMODULE_REVISIONS}.items():
            path = source / relative
            revision = subprocess.run(
                ["git", "-C", str(path), "rev-parse", "HEAD"],
                capture_output=True, text=True, check=False,
            )
            if revision.returncode != 0 or revision.stdout.strip() != expected:
                missing.append({"kind": "source_repo", "path": str(path), "reason": f"upstream checkout must be pinned to {expected}"})
    else:
        missing.append({"kind": "source_repo", "path": str(source), "reason": "a pinned Git checkout is required"})

    need("entrypoint", Path(entrypoint) if entrypoint else None, "WorldFoundry WildDet3D launcher is missing")
    need("checkpoint", _path(options, "checkpoint_path", "model_path", default=DEFAULT_CHECKPOINT), "WildDet3D detector checkpoint is missing")
    need("checkpoint", _path(options, "lingbot_config_path", default=DEFAULT_LINGBOT_CONFIG), "official LingBot-Depth model.pt is needed to build the geometry model")
    need("asset", _path(options, "image_path", "input_image"), "an RGB input image_path is required")
    intrinsics = _path(options, "intrinsics_path")
    if intrinsics is not None:
        need("asset", intrinsics, "camera intrinsics .npy is missing")
    modules = ("torch", "numpy", "PIL", "cv2", "vis4d", "ml_collections", "terminaltables", "huggingface_hub")
    python = str(_option(options, "python_executable", default=sys.executable))
    if python == sys.executable:
        unavailable = [module for module in modules if importlib.util.find_spec(module) is None]
    else:
        executable = shutil.which(python)
        if executable is None:
            missing.append({"kind": "python_executable", "path": python, "reason": "WildDet3D Python interpreter does not exist"})
            unavailable = []
        else:
            probe = subprocess.run(
                [executable, "-c", "import importlib.util,sys; print('\\n'.join(m for m in sys.argv[1:] if importlib.util.find_spec(m) is None))", *modules],
                capture_output=True, text=True, timeout=30, check=False,
            )
            if probe.returncode:
                missing.append({"kind": "python_executable", "path": executable, "reason": f"WildDet3D dependency probe failed: {probe.stderr[-300:]}"})
                unavailable = []
            else:
                unavailable = [module for module in probe.stdout.splitlines() if module]
    for module in unavailable:
        missing.append({"kind": "python_module", "path": module, "reason": f"WildDet3D dependency {module} is unavailable in {python}"})
    return missing


def build_command(context: Mapping[str, Any]) -> list[str]:
    settings = command_settings(context)
    command = [
        str(_option(settings, "python_executable", default=context["python"])),
        str(context["entrypoint"]),
        "--source-dir", str(context["runtime_root"]),
        "--checkpoint-path", str(_path(settings, "checkpoint_path", "model_path", default=DEFAULT_CHECKPOINT)),
        "--lingbot-config-path", str(_path(settings, "lingbot_config_path", default=DEFAULT_LINGBOT_CONFIG)),
        "--image-path", str(_path(settings, "image_path", "input_image")),
        "--output-path", str(context["output_path"]),
        "--device", str(context.get("device") or "cuda"),
        "--classes", str(_option(settings, "classes", "prompt", default=context.get("prompt") or "person, car")),
        "--score-threshold", str(_option(settings, "score_threshold", default=0.3)),
        "--score-3d-threshold", str(_option(settings, "score_3d_threshold", default=0.1)),
    ]
    intrinsics = _path(settings, "intrinsics_path")
    if intrinsics is not None:
        command += ["--intrinsics-path", str(intrinsics)]
    return command
