"""HY-World 2.0 world-generation integration status.

The official HY-World 2.0 repository ships world-generation inference code, but
WorldFoundry does not yet vendor or wrap that full multi-stage runtime. Keep the
status and migration plan in code so CLI/runtime callers get a precise failure
instead of a misleading generic unsupported-task error.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

from PIL import Image

from worldfoundry.runtime.in_tree_cli import ensure_in_tree_runtime, execute_in_tree, require_path

_PROJECT_ROOT = Path(__file__).resolve().parents[4]


SUPPORTED_IN_TREE_TASKS = ("worldrecon", "panorama")
UNSUPPORTED_WORLDGEN_TASKS = (
    "worldgen",
    "world-gen",
    "world-generation",
    "generation",
    "text-to-3d-world",
    "image-to-3d-world",
)

OFFICIAL_WORLDGEN_REQUIRED_FILES = (
    "hyworld2/worldgen/traj_generate.py",
    "hyworld2/worldgen/traj_render.py",
    "hyworld2/worldgen/video_gen.py",
    "hyworld2/worldgen/gen_gs_data.py",
    "hyworld2/worldgen/world_gs_trainer.py",
    "hyworld2/worldgen/show_gs.py",
    "hyworld2/worldgen/models/worldstereo_wrapper.py",
    "hyworld2/worldgen/models/worldstereo.py",
    "hyworld2/worldgen/models/pipelines/pipeline_dmd_keyframe.py",
    "hyworld2/worldgen/src/navi_utils.py",
    "hyworld2/worldgen/src/panorama_utils.py",
    "hyworld2/worldgen/src/pointcloud.py",
    "hyworld2/worldgen/third_party/gsplat_maskgaussian/setup.py",
    "hyworld2/worldgen/third_party/navmesh/setup.py",
)

OFFICIAL_WORLDGEN_STAGES = (
    {
        "stage": "trajectory_planning",
        "script": "hyworld2/worldgen/traj_generate.py",
        "role": "WorldNav VLM-guided trajectory planning from panorama.png.",
        "blocked_on": (
            "navmesh extension built from hyworld2/worldgen/third_party/navmesh",
            "vLLM endpoint serving a VLM such as Qwen/Qwen3-VL-8B-Instruct",
            "SAM3, MoGe, ZIM, and GroundingDINO checkpoint availability",
        ),
    },
    {
        "stage": "trajectory_rendering",
        "script": "hyworld2/worldgen/traj_render.py",
        "role": "Distributed point-cloud rendering and VLM trajectory captioning.",
        "blocked_on": (
            "torchrun/distributed launch wiring",
            "shared LLM endpoint configuration",
            "global_pcd.ply and camera.json files from trajectory planning",
        ),
    },
    {
        "stage": "world_expansion",
        "script": "hyworld2/worldgen/video_gen.py",
        "role": "WorldStereo 2.0 keyframe/video generation with memory bank alignment.",
        "blocked_on": (
            "hanshanxue/WorldStereo worldstereo-memory-dmd weights",
            "FSDP/sequence-parallel launch policy and GPU allocation",
            "SAM3Video and MoGe model availability",
        ),
    },
    {
        "stage": "gs_data_preparation",
        "script": "hyworld2/worldgen/gen_gs_data.py",
        "role": "Build images, aligned depth, normals, cameras, and point clouds for 3DGS.",
        "blocked_on": (
            "generation_bank_worldstereo-memory-dmd outputs",
            "MoGe depth/normal inference",
            "distributed output collation",
        ),
    },
    {
        "stage": "world_composition",
        "script": "hyworld2/worldgen/world_gs_trainer.py",
        "role": "3D Gaussian Splatting training, mesh/SPZ export, and optional viewer.",
        "blocked_on": (
            "custom gsplat_maskgaussian extension built and importable as gsplat",
            "fused_ssim, nerfview, viser, torchmetrics, pymeshlab/open3d export stack",
            "bounded training presets for 1/2/4/8 GPU runs",
        ),
    },
)


class HYWorld2WorldgenNotIntegratedError(NotImplementedError):
    """Raised when callers request unsupported HY-World 2.0 world generation."""


def _default_official_repo_path() -> Path | None:
    configured = os.environ.get("WORLDFOUNDRY_HY_WORLD_2P0_REPO")
    if configured:
        return Path(configured).expanduser()
    return None


def inspect_official_worldgen_checkout(repo_path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Return lightweight file-level readiness for an explicitly configured HY-World-2.0 checkout."""
    root = Path(repo_path).expanduser() if repo_path is not None else _default_official_repo_path()
    if root is None:
        return {
            "repo_path": None,
            "exists": False,
            "missing_files": list(OFFICIAL_WORLDGEN_REQUIRED_FILES),
        }

    missing = [relative for relative in OFFICIAL_WORLDGEN_REQUIRED_FILES if not (root / relative).is_file()]
    return {
        "repo_path": str(root),
        "exists": root.exists(),
        "missing_files": missing,
    }


def get_hy_world_2p0_worldgen_plan(repo_path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Describe what is needed before WorldFoundry can expose full worldgen."""
    checkout = inspect_official_worldgen_checkout(repo_path)
    return {
        "status": "not_integrated",
        "supported_tasks": list(SUPPORTED_IN_TREE_TASKS),
        "supported_in_tree_tasks": list(SUPPORTED_IN_TREE_TASKS),
        "unsupported_task": "worldgen",
        "official_checkout": checkout,
        "reason": (
            "The official world-generation path is a five-stage distributed runtime "
            "with native extensions and an external VLM service. WorldFoundry has not "
            "yet vendored those files or wrapped the stage orchestration in-tree."
        ),
        "required_stages": list(OFFICIAL_WORLDGEN_STAGES),
        "implementation_plan": [
            "Vendor the official hyworld2/worldgen Python modules under a WorldFoundry runtime package, preserving upstream provenance.",
            "Vendor or package the custom gsplat_maskgaussian and navmesh native extensions with build/install checks.",
            "Add a bounded stage orchestrator for traj_generate, traj_render, video_gen, gen_gs_data, and world_gs_trainer.",
            "Define explicit runtime config for VLM endpoint, GPU counts, FSDP/torchrun launch, checkpoints, and training-step presets.",
            "Add preflight checks for required checkpoints, compiled extensions, CUDA/distributed availability, and input scene layout.",
            "Add small syntax/import tests and command-construction checks; reserve GPU validation for an explicit bounded job.",
        ],
    }


def raise_hy_world_2p0_worldgen_not_integrated(task: str = "worldgen") -> None:
    """Raise an actionable error for unsupported HY-World 2.0 worldgen requests."""
    plan = get_hy_world_2p0_worldgen_plan()
    checkout = plan["official_checkout"]
    missing = checkout.get("missing_files") or []
    missing_text = "none" if not missing else ", ".join(missing[:4]) + (" ..." if len(missing) > 4 else "")
    raise HYWorld2WorldgenNotIntegratedError(
        "HY-World 2.0 task={!r} is not integrated in WorldFoundry. "
        "Current in-tree support is limited to task='worldrecon' and task='panorama'. "
        "Official worldgen checkout: {} (missing required files: {}). "
        "Required migration stages: trajectory_planning, trajectory_rendering, "
        "world_expansion, gs_data_preparation, world_composition.".format(
            task,
            checkout.get("repo_path") or "not configured",
            missing_text,
        )
    )


class HYWorld2WorldgenRuntime:
    """Render a WRBench trajectory through a pre-materialized HY-World scene.

    This wraps the official ``traj_render.py`` stage only. It does not claim
    to run the separate WorldNav, WorldStereo, or 3DGS training stages.
    """

    SOURCE_REVISION = "7f668e67c74338d50684e57be46a438459b6bbe1"

    def __init__(
        self,
        *,
        scene_path: Any = None,
        python_executable: Any = None,
        device: str = "cuda",
        env: Mapping[str, Any] | None = None,
    ) -> None:
        self.repo_root = ensure_in_tree_runtime(self.bundled_repo_root(), package_file=__file__)
        require_path(
            self.repo_root / "hyworld2" / "worldgen" / "traj_render.py",
            "HY-World 2.0 trajectory-render source (install the official checkout in the in-tree runtime directory)",
            kind="file",
        )
        configured_scene = scene_path or os.environ.get("HYWORLD_SCENE_PATH")
        self.scene_path = Path(configured_scene).expanduser() if configured_scene else None
        self.python_executable = str(python_executable or sys.executable)
        self.device = device
        self.env = dict(env or {})

    @staticmethod
    def bundled_repo_root() -> Path:
        return Path(__file__).resolve().parent / "hy_world_2p0_worldgen_runtime"

    @classmethod
    def from_pretrained(cls, pretrained_model_path: Any = None, **kwargs: Any) -> "HYWorld2WorldgenRuntime":
        if pretrained_model_path and "scene_path" not in kwargs:
            kwargs["scene_path"] = pretrained_model_path
        return cls(**kwargs)

    def plan(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "model_id": "hyworld-worldgen",
            "stage": "trajectory_rendering",
            "repo_root": str(self.repo_root),
            "source_revision": self.SOURCE_REVISION,
            "scene_path": str(self.scene_path) if self.scene_path else None,
            "inputs": {key: str(value) for key, value in kwargs.items()},
        }

    def predict(
        self,
        *,
        image_path: Any,
        camera_json: Any,
        output_path: Any,
        width: int = 832,
        height: int = 480,
        seed: int = 1024,
        return_dict: bool = True,
        **_: Any,
    ) -> Any:
        source_scene = require_path(self.scene_path, "HY-World materialized scene", kind="dir")
        source_pcd = require_path(
            source_scene / "render_results" / "global_pcd.ply",
            "HY-World global point cloud",
            kind="file",
        )
        image = require_path(image_path, "HY-World trajectory start frame", kind="file")
        camera = require_path(camera_json, "HY-World camera JSON", kind="file")
        output = Path(output_path).expanduser().resolve()
        input_dir = output.parent / f".{output.stem}_hyworld_inputs"
        scene = input_dir / "scene"
        render_results = scene / "render_results"
        view = render_results / "view0"
        trajectory = view / "traj_wrbench"
        trajectory.mkdir(parents=True, exist_ok=True)

        panorama = scene / "panorama.png"
        with Image.open(image) as first_frame:
            first_frame.convert("RGB").resize((int(width), int(height)), Image.Resampling.LANCZOS).save(panorama)
        shutil.copy2(panorama, view / "start_frame.png")
        shutil.copy2(camera, trajectory / "camera.json")
        staged_pcd = render_results / "global_pcd.ply"
        if not staged_pcd.exists():
            staged_pcd.symlink_to(source_pcd.resolve())

        env = {str(key): str(value) for key, value in self.env.items()}
        if self.device.startswith("cuda:") and "CUDA_VISIBLE_DEVICES" not in env:
            env["CUDA_VISIBLE_DEVICES"] = self.device.split(":", 1)[1]
        worldgen_root = self.repo_root / "hyworld2" / "worldgen"
        command = [
            self.python_executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=1",
            "traj_render.py",
            "--target_path",
            scene,
            "--seed",
            str(seed),
        ]
        result = execute_in_tree(
            command,
            cwd=worldgen_root,
            output_path=output,
            search_roots=(trajectory,),
            preferred_names=("render.mp4",),
            env=env,
            python_paths=(_PROJECT_ROOT, worldgen_root, self.repo_root),
        )
        return result if return_dict else result.get("video")


__all__ = [
    "HYWorld2WorldgenNotIntegratedError",
    "OFFICIAL_WORLDGEN_REQUIRED_FILES",
    "OFFICIAL_WORLDGEN_STAGES",
    "SUPPORTED_IN_TREE_TASKS",
    "UNSUPPORTED_WORLDGEN_TASKS",
    "HYWorld2WorldgenRuntime",
    "get_hy_world_2p0_worldgen_plan",
    "inspect_official_worldgen_checkout",
    "raise_hy_world_2p0_worldgen_not_integrated",
]
