from __future__ import annotations

import configparser
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.core.io.paths import project_root
from worldfoundry.studio.inference.catalog import CatalogEntry
from worldfoundry.studio.inference.paths import studio_model_roots
from worldfoundry.studio.ui.vendor_assets import VENDOR_ASSET_INSTALL_COMMAND

REPO_ROOT = project_root(__file__)


LOCAL_REPO_ALIASES: dict[str, tuple[str, ...]] = {
    "gen3c": ("GEN3C",),
    "infinite-world": ("Infinite-World",),
    "lingbot-world": ("lingbot-world",),
    "matrix-game-2": ("Matrix-Game", "Matrix-Game/Matrix-Game-2"),
    "matrix-game-3": ("Matrix-Game", "Matrix-Game/Matrix-Game-3"),
    "neoverse": ("NeoVerse",),
    "vmem": ("vmem",),
    "worldcam": ("WorldCam",),
    "yume": ("YUME",),
    "yume-1p5": ("YUME",),
    "worldfm": ("worldfm",),
    "animatediff": ("AnimateDiff",),
    "being-h05": ("Being-H", "Being-H/Being-H05"),
    "cameractrl": ("CameraCtrl",),
    "dreamdojo": ("DreamDojo",),
    "hunyuan-game-craft": ("Hunyuan-GameCraft-1.0",),
    "hunyuan-worldplay": ("HY-WorldPlay",),
    "lyra1": ("lyra", "lyra/Lyra-1"),
    "lyra2": ("lyra", "lyra/Lyra-2"),
    "motionctrl": ("MotionCtrl",),
    "recammaster": ("ReCamMaster",),
    "solaris": ("solaris",),
    "wan-2p1-i2v": ("Wan2.2",),
    "wan-2p1-t2v": ("Wan2.2",),
    "wan-2p2": ("Wan2.2",),
    "depth-anything-v3": ("Depth-Anything-3",),
    "dreamzero": ("dreamzero",),
    "giga-brain-0": ("giga-brain-0",),
    "lingbot-va": ("lingbot-va",),
    "openvla": ("RoboTwin/policy/openvla-oft", "RoboTwin"),
    "openpi": ("RoboTwin/policy/pi0", "RoboTwin"),
    "diffusion-policy": ("RoboTwin/policy/DP", "RoboTwin"),
    "act": ("RoboTwin/policy/ACT", "RoboTwin"),
    "hunyuanworld-mirror": ("HunyuanWorld-Mirror",),
    "hunyuan-mirror": ("HunyuanWorld-Mirror",),
    "hunyuan-world-voyager": ("HunyuanWorld-Voyager",),
    "hy-world-2p0": ("HY-World-2.0",),
    "inspatio-world": ("inspatio-world",),
    "flash-world": ("FlashWorld",),
}


@dataclass(frozen=True)
class LocalRepoEvidence:
    path: str = ""
    status: str = "missing"
    remote_urls: tuple[str, ...] = ()
    readme_path: str = ""
    entrypoints: tuple[str, ...] = ()
    env_files: tuple[str, ...] = ()
    gui_paths: tuple[str, ...] = ()
    gui_entrypoints: tuple[str, ...] = ()
    gui_assets: tuple[str, ...] = ()


@dataclass(frozen=True)
class StudioInterfaceSpec:
    template_id: str
    template_title: str
    input_groups: tuple[str, ...]
    output_groups: tuple[str, ...]
    interaction_model: str
    task_family: str = ""
    artifact_kind: str = ""
    runtime_status: str = ""
    profile_available: bool = False
    local_repo: LocalRepoEvidence = field(default_factory=LocalRepoEvidence)
    source_urls: tuple[str, ...] = ()
    gui_refs: tuple[str, ...] = ()
    launch_hints: tuple[str, ...] = ()

    @property
    def repo_status(self) -> str:
        return self.local_repo.status


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _read_git_remotes(repo_path: Path) -> tuple[str, ...]:
    config_path = repo_path / ".git" / "config"
    if not config_path.exists():
        return ()
    parser = configparser.ConfigParser()
    try:
        parser.read(config_path)
    except configparser.Error:
        return ()
    urls: list[str] = []
    for section in parser.sections():
        if section.startswith('remote "') and parser.has_option(section, "url"):
            urls.append(parser.get(section, "url"))
    return tuple(dict.fromkeys(urls))


def _repo_entrypoints(repo_path: Path) -> tuple[str, ...]:
    names = (
        "app.py",
        "demo.py",
        "inference.py",
        "infer.py",
        "gradio_app.py",
        "predict.py",
        "scripts/inference.py",
        "examples/inference.py",
    )
    found = [str((repo_path / name).relative_to(repo_path)) for name in names if (repo_path / name).exists()]
    if not found:
        for pattern in ("*_synthesis.py", "pipeline_*.py", "*_runner.py", "runtime.py"):
            found.extend(str(path.relative_to(repo_path)) for path in sorted(repo_path.glob(pattern)))
    return tuple(found[:8])


def _repo_env_files(repo_path: Path) -> tuple[str, ...]:
    names = ("requirements.txt", "environment.yml", "pyproject.toml", "setup.py", "docker/Dockerfile", "Dockerfile")
    return tuple(str((repo_path / name).relative_to(repo_path)) for name in names if (repo_path / name).exists())


def _repo_gui_paths(repo_path: Path) -> tuple[str, ...]:
    names = ("gui", "examples/viewer", "examples", "gradio_app.py", "app.py")
    return tuple(str((repo_path / name).relative_to(repo_path)) for name in names if (repo_path / name).exists())


def _repo_gui_entrypoints(repo_path: Path) -> tuple[str, ...]:
    names = (
        "gui/api/client.py",
        "gui/api/server.py",
        "examples/viewer/index.html",
        "gradio_app.py",
        "app.py",
    )
    return tuple(str((repo_path / name).relative_to(repo_path)) for name in names if (repo_path / name).exists())


def _repo_gui_assets(repo_path: Path) -> tuple[str, ...]:
    names = (
        "gui/assets/gui_preview.webp",
        "examples/viewer/index.html",
        "dist/spark.module.js",
    )
    return tuple(str((repo_path / name).relative_to(repo_path)) for name in names if (repo_path / name).exists())


@lru_cache(maxsize=1)
def _local_repo_index() -> dict[str, Path]:
    index: dict[str, Path] = {}
    for root in studio_model_roots(repo_root=REPO_ROOT):
        if not root.exists():
            continue
        for path in root.iterdir():
            if path.is_dir():
                index.setdefault(_normalize_key(path.name), path)
    return index


def _in_tree_source_candidates(entry: CatalogEntry) -> tuple[Path, ...]:
    keys = tuple(
        dict.fromkeys(
            _normalize_key(value).replace("-", "_")
            for value in (entry.model_id, entry.family, entry.display_name)
            if value
        )
    )
    roots = (
        REPO_ROOT / "worldfoundry" / "pipelines",
        REPO_ROOT / "worldfoundry" / "synthesis" / "visual_generation",
        REPO_ROOT / "worldfoundry" / "synthesis" / "action_generation",
        REPO_ROOT / "worldfoundry" / "base_models" / "three_dimensions" / "depth",
        REPO_ROOT / "worldfoundry" / "base_models" / "three_dimensions" / "point_clouds",
        REPO_ROOT / "worldfoundry" / "base_models" / "three_dimensions" / "general_3d",
    )
    candidates: list[Path] = []
    for root in roots:
        for key in keys:
            candidates.append(root / key)
    return tuple(candidates)


def _resolve_local_repo(entry: CatalogEntry, profile: Any | None) -> LocalRepoEvidence:
    candidates: list[str] = list(LOCAL_REPO_ALIASES.get(entry.model_id, ()))
    candidates.extend([entry.model_id, entry.display_name, entry.family])
    if profile is not None:
        for source in getattr(profile, "source_repos", ()) or ():
            if isinstance(source, Mapping):
                for key in ("local_dir", "repo", "name", "url"):
                    value = str(source.get(key) or "")
                    if value:
                        candidates.append(Path(value).name)

    repo_path: Path | None = None
    for raw_candidate in candidates:
        for root in studio_model_roots(repo_root=REPO_ROOT):
            candidate_path = (root / raw_candidate).expanduser()
            if candidate_path.exists():
                repo_path = candidate_path
                break
        if repo_path is not None:
            break
        indexed = _local_repo_index().get(_normalize_key(raw_candidate))
        if indexed is not None:
            repo_path = indexed
            break

    if repo_path is None:
        for candidate_path in _in_tree_source_candidates(entry):
            if candidate_path.exists():
                repo_path = candidate_path
                break

    if repo_path is None:
        return LocalRepoEvidence()

    readme = repo_path / "README.md"
    return LocalRepoEvidence(
        path=str(repo_path),
        status="present",
        remote_urls=_read_git_remotes(repo_path),
        readme_path=str(readme) if readme.exists() else "",
        entrypoints=_repo_entrypoints(repo_path),
        env_files=_repo_env_files(repo_path),
        gui_paths=_repo_gui_paths(repo_path),
        gui_entrypoints=_repo_gui_entrypoints(repo_path),
        gui_assets=_repo_gui_assets(repo_path),
    )


@lru_cache(maxsize=1)
def _runtime_profiles_index() -> Mapping[str, Any]:
    """Load Studio runtime profile metadata once per process.

    Args:
        None.
    """
    from worldfoundry.evaluation.models.runtime.profiles import load_runtime_profiles

    return load_runtime_profiles(check_conda_env_exists=False)


def _runtime_profile(entry: CatalogEntry) -> Any | None:
    if os.getenv("WORLDFOUNDRY_STUDIO_SKIP_RUNTIME_PROFILES", "").strip().lower() in {"1", "true", "yes", "on"}:
        return None
    try:
        return _runtime_profiles_index().get(entry.model_id)
    except Exception:
        return None


def _schema_flags(profile: Any | None) -> dict[str, Any]:
    schema = dict(getattr(profile, "input_schema", {}) or {}) if profile is not None else {}
    actions = schema.get("actions") or []
    if isinstance(actions, str):
        actions = [actions]
    return {
        "prompt": bool(schema.get("prompt") or schema.get("instruction")),
        "image": bool(schema.get("image") or schema.get("images")),
        "video": bool(schema.get("video") or schema.get("videos")),
        "state": bool(schema.get("state") or schema.get("proprio") or schema.get("world_state")),
        "actions": tuple(str(item) for item in actions),
    }


def _template_for(entry: CatalogEntry, profile: Any | None) -> tuple[str, str, str]:
    task_family = str(getattr(profile, "task_family", "") or "")
    tags = {tag.lower() for tag in entry.tags}
    if entry.category == "Remote API":
        return "hosted-api", "Hosted Provider API", "provider request"
    if entry.category == "Video-to-Video":
        return "video-to-video", "Video-to-Video Editor", "source video + trajectory"
    if entry.category == "Depth / Geometry":
        return "depth-geometry", "Depth / Geometry Extractor", "file batch"
    if entry.category == "3D Scene":
        return "scene-3d", "3D Reconstruction / Viewer", "camera path"
    if entry.category == "Visual Action" or task_family == "visual_action_model":
        return "visual-action", "Visual Action Tokenizer", "latent action stream"
    if entry.category == "Embodied Action" or task_family in {"vla_policy", "world_action_model", "visuomotor_policy", "action_chunking_policy"}:
        return "embodied-policy", "Embodied Policy Console", "robot action stream"
    if "queued-segment-generation" in tags:
        return "interactive-world", "Queued World Segments", "resident segment queue"
    if "prompt-scheduled" in tags:
        return "interactive-world", "Prompt-Scheduled World Rollout", "prompt segment stream"
    if "interactive-world" in tags:
        return "interactive-world", "Interactive World Rollout", "navigation stream"
    if entry.supports_stream and (entry.default_interactions or tags & {"navigation", "camera-control"}):
        return "interactive-world", "Interactive World Rollout", "navigation stream"
    if "i2v" in tags or "image-to-video" in tags or "images" in entry.call_params:
        return "conditioned-video", "Conditioned Video Generator", "one-shot or stream"
    return "text-video", "Text / Media Video Generator", "one-shot or stream"


def _input_groups(entry: CatalogEntry, profile: Any | None, template_id: str) -> tuple[str, ...]:
    flags = _schema_flags(profile)
    groups: list[str] = []
    if template_id == "hosted-api":
        return ("Prompt", "Condition media", "Endpoint / API key", "Provider JSON")
    if template_id == "video-to-video":
        return ("Source video", "Camera trajectory", "Prompt", "Runtime JSON")
    if template_id == "depth-geometry":
        return ("Image / video / folder path", "Batch list", "Depth runtime JSON")
    if template_id == "scene-3d":
        return ("Source image / video", "Scene path", "Camera pose", "Intrinsics", "3DGS asset")
    if flags["prompt"] or "prompt" in entry.call_params or entry.default_prompt:
        groups.append("Instruction / prompt" if "action" in template_id or template_id == "embodied-policy" else "Prompt")
    if flags["image"] or {"images", "image", "image_path", "data_path"} & set(entry.call_params):
        groups.append("Observation image" if "action" in template_id or template_id == "embodied-policy" else "Image")
    if flags["video"] or {"video", "videos", "video_path"} & set(entry.call_params):
        groups.append("Observation video" if "action" in template_id or template_id == "embodied-policy" else "Video")
    if flags["state"]:
        groups.append("Robot state / proprio")
    if flags["actions"] or entry.default_interactions:
        groups.append("Action tokens: " + ", ".join(flags["actions"] or entry.default_interactions))
    if not groups:
        groups.append("Prompt or file inputs from pipeline signature")
    return tuple(dict.fromkeys(groups))


def _output_groups(entry: CatalogEntry, profile: Any | None, template_id: str) -> tuple[str, ...]:
    artifact_kind = str(getattr(profile, "artifact_kind", "") or "")
    if artifact_kind:
        return (artifact_kind.replace("_", " ").title(), "Manifest JSON", "Artifacts")
    if template_id == "hosted-api":
        return ("Provider artifact", "Manifest JSON", "Artifacts")
    if template_id == "video-to-video":
        return ("Edited video", "Trajectory preview", "Manifest JSON")
    if template_id == "depth-geometry":
        return ("Depth map", "Geometry files", "Manifest JSON")
    if template_id == "scene-3d":
        return ("3D asset", "Rendered trajectory", "Manifest JSON")
    if template_id == "embodied-policy":
        return ("Action trace", "Simulator replay", "Manifest JSON")
    if template_id == "visual-action":
        return ("Action trace", "Action JSON", "Manifest JSON")
    return ("Generated video", "Preview image", "Manifest JSON")


def _gui_refs(entry: CatalogEntry, template_id: str, local_repo: LocalRepoEvidence) -> tuple[str, ...]:
    refs: list[str] = []
    if local_repo.gui_paths:
        refs.append(f"Model GUI: {local_repo.path}/{local_repo.gui_paths[0]}")
    if entry.model_id == "gen3c" and local_repo.path and (Path(local_repo.path) / "gui").exists():
        refs.append(f"GEN3C authoring GUI: {local_repo.path}/gui")
    if template_id == "scene-3d":
        refs.append(f"In-tree Spark 3DGS viewer: `worldfoundry-studio {entry.model_id} --frontend spark`")
    return tuple(dict.fromkeys(refs))


def _launch_hints(entry: CatalogEntry, template_id: str, local_repo: LocalRepoEvidence) -> tuple[str, ...]:
    hints: list[str] = []
    if entry.model_id == "gen3c" and local_repo.path and (Path(local_repo.path) / "gui").exists():
        hints.append("GEN3C server: CUDA_HOME=$CONDA_PREFIX fastapi dev --no-reload ./gui/api/server.py --host 0.0.0.0")
        hints.append("GEN3C client: python gui/api/client.py")
    if template_id == "scene-3d":
        hints.append("Spark native frontend: worldfoundry-studio <model> --frontend spark --asset /path/to/scene.splat")
        hints.append(f"Install pinned browser modules: {VENDOR_ASSET_INSTALL_COMMAND}")
    if template_id == "depth-geometry" or entry.runtime_kind in {"pointcloud_nav", "worldfm"}:
        hints.append("Viser native frontend: worldfoundry-studio <model> --frontend points --asset /path/to/scene.ply")
    if template_id == "embodied-policy" or entry.category == "Embodied Action":
        hints.append("Embodied native frontend: start the simulator, then pass --frontend embodied --simulator-url http://127.0.0.1:<port>")
    return tuple(hints)


INTERFACE_SPEC_CACHE: dict[str, StudioInterfaceSpec] = {}


def _build_interface_spec_for_model(model_id: str, entry: CatalogEntry) -> StudioInterfaceSpec:
    profile = _runtime_profile(entry)
    template_id, template_title, interaction_model = _template_for(entry, profile)
    local_repo = _resolve_local_repo(entry, profile)
    source_urls = []
    if profile is not None:
        for source in getattr(profile, "source_repos", ()) or ():
            if isinstance(source, Mapping) and source.get("url"):
                source_urls.append(str(source["url"]))
    return StudioInterfaceSpec(
        template_id=template_id,
        template_title=template_title,
        input_groups=_input_groups(entry, profile, template_id),
        output_groups=_output_groups(entry, profile, template_id),
        interaction_model=interaction_model,
        task_family=str(getattr(profile, "task_family", "") or ""),
        artifact_kind=str(getattr(profile, "artifact_kind", "") or ""),
        runtime_status=str(getattr(profile, "runtime_status", "") or ""),
        profile_available=profile is not None,
        local_repo=local_repo,
        source_urls=tuple(dict.fromkeys(source_urls)),
        gui_refs=_gui_refs(entry, template_id, local_repo),
        launch_hints=_launch_hints(entry, template_id, local_repo),
    )


def interface_spec_for_model(model_id: str, entry: CatalogEntry) -> StudioInterfaceSpec:
    cached = INTERFACE_SPEC_CACHE.get(model_id)
    if cached is not None:
        return cached
    spec = _build_interface_spec_for_model(model_id, entry)
    INTERFACE_SPEC_CACHE[model_id] = spec
    return spec


def interface_spec_for_entry(entry: CatalogEntry) -> StudioInterfaceSpec:
    return interface_spec_for_model(entry.model_id, entry)


def compact_interface_summary(entry: CatalogEntry) -> dict[str, Any]:
    spec = interface_spec_for_entry(entry)
    return {
        "model_id": entry.model_id,
        "display_name": entry.display_name,
        "category": entry.category,
        "template_id": spec.template_id,
        "template_title": spec.template_title,
        "inputs": list(spec.input_groups),
        "outputs": list(spec.output_groups),
        "interaction_model": spec.interaction_model,
        "task_family": spec.task_family,
        "artifact_kind": spec.artifact_kind,
        "runtime_status": spec.runtime_status,
        "profile_available": spec.profile_available,
        "local_repo": {
            "status": spec.local_repo.status,
            "path": spec.local_repo.path,
            "readme_path": spec.local_repo.readme_path,
            "entrypoints": list(spec.local_repo.entrypoints),
            "env_files": list(spec.local_repo.env_files),
            "gui_paths": list(spec.local_repo.gui_paths),
            "gui_entrypoints": list(spec.local_repo.gui_entrypoints),
            "gui_assets": list(spec.local_repo.gui_assets),
            "remote_urls": list(spec.local_repo.remote_urls),
        },
        "source_urls": list(spec.source_urls),
        "gui_refs": list(spec.gui_refs),
        "launch_hints": list(spec.launch_hints),
    }
