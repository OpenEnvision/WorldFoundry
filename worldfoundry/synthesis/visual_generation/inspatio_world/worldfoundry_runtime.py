from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch

from worldfoundry.core.io import (
    VIDEO_EXTENSIONS,
    load_video_frames,
    materialize_video_input,
)
from worldfoundry.core.io.paths import package_module_root as package_root
from worldfoundry.core.io.paths import (
    checkpoint_root_path,
    hfd_root_path,
    resolve_local_hf_model_path,
)
from worldfoundry.evaluation.utils import worldfoundry_data_path
from .inspatio_world_runtime.utils.reproducibility import SEED_ENV, reproducibility_env


_BUNDLED_REPO_ROOT = (
    package_root("worldfoundry.synthesis.visual_generation.inspatio_world")
    / "inspatio_world_runtime"
)
_CONFIG_ROOT = worldfoundry_data_path("models", "runtime", "configs", "inspatio_world")
_TRAJECTORY_ROOT = _CONFIG_ROOT / "traj"

DEFAULT_CHECKPOINT_REPO = "inspatio/world"
DEFAULT_WAN_MODEL_REPO = "Wan-AI/Wan2.1-T2V-1.3B"
DEFAULT_DA3_MODEL_REPO = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
DEFAULT_FLORENCE_MODEL_REPO = "microsoft/Florence-2-large"
DEFAULT_TRAJECTORY_NAME = "x_y_circle_cycle.txt"
DEFAULT_TRAJECTORY_PATH = _TRAJECTORY_ROOT / DEFAULT_TRAJECTORY_NAME
DEFAULT_CONFIG_PATH = _CONFIG_ROOT / "inference_1.3b.yaml"


def _load_omegaconf():
    try:
        from omegaconf import OmegaConf
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "InSpatio-World runtime config materialization requires the optional `omegaconf` dependency."
        ) from exc
    return OmegaConf


class InspatioWorldRuntime:
    """Runtime owner for the vendored official InSpatio-World pipeline."""

    def __init__(
        self,
        repo_root: Optional[str] = None,
        checkpoint_source: str = DEFAULT_CHECKPOINT_REPO,
        wan_model_source: str = DEFAULT_WAN_MODEL_REPO,
        da3_model_source: str = DEFAULT_DA3_MODEL_REPO,
        florence_model_source: str = DEFAULT_FLORENCE_MODEL_REPO,
        device: str = "cuda",
        defaults: Optional[Dict[str, Any]] = None,
    ) -> None:
        bundled_root = Path(self.bundled_repo_root()).resolve()
        if repo_root is not None and Path(repo_root).expanduser().resolve() != bundled_root:
            raise ValueError("InSpatio-World runtime is vendored in-tree; external repo_root is not supported.")
        self.repo_root = str(bundled_root)
        self.checkpoint_source = checkpoint_source
        self.wan_model_source = wan_model_source
        self.da3_model_source = da3_model_source
        self.florence_model_source = florence_model_source
        self.device = device
        self.defaults = defaults or {}

    @classmethod
    def bundled_repo_root(cls) -> str:
        return str(_BUNDLED_REPO_ROOT)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str = DEFAULT_CHECKPOINT_REPO,
        args=None,
        device: Optional[str] = None,
        wan_model_path: str = DEFAULT_WAN_MODEL_REPO,
        da3_model_path: str = DEFAULT_DA3_MODEL_REPO,
        florence_model_path: str = DEFAULT_FLORENCE_MODEL_REPO,
        config_path: Optional[str] = None,
        tae_checkpoint_path: Optional[str] = None,
        default_traj_txt_path: Optional[str] = None,
        **kwargs,
    ) -> "InspatioWorldRuntime":
        defaults = {
            "config_path": config_path,
            "tae_checkpoint_path": tae_checkpoint_path,
            "default_traj_txt_path": default_traj_txt_path,
            **kwargs,
        }
        return cls(
            repo_root=cls.bundled_repo_root(),
            checkpoint_source=pretrained_model_path or DEFAULT_CHECKPOINT_REPO,
            wan_model_source=wan_model_path or DEFAULT_WAN_MODEL_REPO,
            da3_model_source=da3_model_path or DEFAULT_DA3_MODEL_REPO,
            florence_model_source=florence_model_path or DEFAULT_FLORENCE_MODEL_REPO,
            device=device or "cuda",
            defaults=defaults,
        )

    @staticmethod
    def _resolve_snapshot_dir(path_or_repo: str) -> str:
        """Resolve a local model directory.

        Args:
            path_or_repo: Local directory or file path for an already materialized model component.
        """
        candidate = Path(str(path_or_repo)).expanduser()
        if candidate.exists():
            if candidate.is_dir():
                return str(candidate.resolve())
            return str(candidate.resolve().parent)
        try:
            return str(resolve_local_hf_model_path(str(path_or_repo)))
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Model component not found locally: {path_or_repo}. "
                "The in-tree InSpatio-World wrapper does not download remote repositories."
            ) from exc

    @classmethod
    def _resolve_checkpoint_path(cls, checkpoint_source: str) -> str:
        candidate = Path(str(checkpoint_source)).expanduser()
        if candidate.is_file():
            return str(candidate.resolve())

        root = Path(cls._resolve_snapshot_dir(checkpoint_source))
        candidates = [
            root / "InSpatio-World-1.3B.safetensors",
            root / "InSpatio-World-1.3B" / "InSpatio-World-1.3B.safetensors",
        ]
        for resolved in candidates:
            if resolved.is_file():
                return str(resolved.resolve())

        matches = sorted(root.glob("**/*.safetensors"))
        if matches:
            return str(matches[0].resolve())
        raise FileNotFoundError(f"Unable to locate an InSpatio-World checkpoint under {root}.")

    def _resolve_config_path(self, config_path: Optional[str]) -> str:
        resolved = config_path or self.defaults.get("config_path") or DEFAULT_CONFIG_PATH
        candidate = Path(str(resolved)).expanduser()
        if candidate.is_absolute() and candidate.exists():
            return str(candidate.resolve())
        parts = candidate.parts
        data_relative = Path(*parts[1:]) if parts and parts[0] == "configs" else candidate
        data_candidate = _CONFIG_ROOT / data_relative
        if data_candidate.exists():
            return str(data_candidate.resolve())
        repo_candidate = Path(self.repo_root) / str(resolved)
        if repo_candidate.exists():
            return str(repo_candidate.resolve())
        raise FileNotFoundError(f"InSpatio-World config not found: {resolved}")

    def _resolve_trajectory_path(self, traj_txt_path: Optional[str]) -> str:
        raw_value = traj_txt_path or self.defaults.get("default_traj_txt_path") or DEFAULT_TRAJECTORY_NAME
        candidate = Path(str(raw_value)).expanduser()
        if candidate.exists():
            return str(candidate.resolve())

        name = Path(str(raw_value)).name
        search_candidates = []
        if not candidate.is_absolute():
            search_candidates.append(_CONFIG_ROOT / candidate)
            search_names = [name]
            if not name.endswith(".txt"):
                search_names.append(f"{name}.txt")
            search_candidates.extend(_TRAJECTORY_ROOT / search_name for search_name in search_names)

        for data_candidate in search_candidates:
            if data_candidate.is_file():
                return str(data_candidate.resolve())

        raise FileNotFoundError(
            "Trajectory file not found: "
            f"{raw_value}. Checked local path and data trajectory root {_TRAJECTORY_ROOT}."
        )

    def _resolve_optional_tae_checkpoint(self, tae_checkpoint_path: Optional[str], use_tae: bool) -> Optional[str]:
        if not use_tae:
            return None

        resolved = tae_checkpoint_path or self.defaults.get("tae_checkpoint_path")
        if resolved is None:
            fallbacks = (
                Path(self.repo_root) / "checkpoints" / "taehv" / "taew2_1.pth",
                checkpoint_root_path("taehv", "taew2_1.pth"),
                hfd_root_path("custom--taehv", "taew2_1.pth"),
            )
            for fallback in fallbacks:
                if fallback.is_file():
                    return str(fallback.resolve())
            raise FileNotFoundError(
                "TAE checkpoint is required when use_tae=True. Pass tae_checkpoint_path explicitly; "
                f"checked: {', '.join(str(path) for path in fallbacks)}."
            )

        candidate = Path(str(resolved)).expanduser()
        if candidate.exists():
            return str(candidate.resolve())
        raise FileNotFoundError(f"TAE checkpoint not found: {resolved}")

    @staticmethod
    def _link_or_copy(src: Path, dst: Path) -> None:
        """Stage a local input by symlinking it into the runtime workspace.

        Args:
            src: Existing source file path.
            dst: Destination symlink path under the generated input directory.
        """
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)

    def _stage_input_dir(self, visual_input, output_root: Path, fps: int) -> tuple[Path, list[Path], Optional[Path]]:
        input_dir = output_root / "input"
        input_dir.mkdir(parents=True, exist_ok=True)

        staged_videos: list[Path] = []
        source_dir: Optional[Path] = None
        if isinstance(visual_input, (str, os.PathLike)):
            candidate = Path(visual_input).expanduser()
            if not candidate.exists():
                raise FileNotFoundError(f"Input video path not found: {visual_input}")

            if candidate.is_dir():
                source_dir = candidate.resolve()
                for source in sorted(candidate.iterdir()):
                    if source.is_file() and source.suffix.lower() in VIDEO_EXTENSIONS:
                        target = input_dir / source.name
                        self._link_or_copy(source.resolve(), target)
                        staged_videos.append(target)
                if not staged_videos:
                    raise ValueError(f"No video files found under directory: {visual_input}")
                return input_dir, staged_videos, source_dir

            if candidate.suffix.lower() not in VIDEO_EXTENSIONS:
                raise ValueError(f"Unsupported video extension for {visual_input}")
            target = input_dir / candidate.name
            self._link_or_copy(candidate.resolve(), target)
            return input_dir, [target], None

        if isinstance(visual_input, (list, tuple)) and visual_input and all(
            isinstance(item, (str, os.PathLike)) for item in visual_input
        ):
            if len(visual_input) == 1 and Path(visual_input[0]).expanduser().is_dir():
                # Workspace normalizes ``input_path`` into ``videos=[path]``.
                # Preserve directory inputs used to resume from precomputed
                # ``new_vggt`` assets instead of misreporting the directory as
                # a missing video file.
                return self._stage_input_dir(visual_input[0], output_root, fps)
            for item in visual_input:
                source = Path(item).expanduser()
                if not source.exists() or not source.is_file():
                    raise FileNotFoundError(f"Input video path not found: {item}")
                if source.suffix.lower() not in VIDEO_EXTENSIONS:
                    raise ValueError(f"Unsupported video extension for {item}")
                target = input_dir / source.name
                self._link_or_copy(source.resolve(), target)
                staged_videos.append(target)
            if not staged_videos:
                raise ValueError("No input videos provided.")
            return input_dir, staged_videos, None

        output_path = Path(
            materialize_video_input(
                visual_input,
                output_dir=str(input_dir),
                filename="input.mp4",
                fps=fps,
            )
        )
        return input_dir, [output_path], None

    @staticmethod
    def _write_prompt_override_json(input_dir: Path, staged_videos: Sequence[Path], prompt: str) -> str:
        payload = []
        for video_path in staged_videos:
            vggt_root = input_dir / "new_vggt" / video_path.stem
            payload.append(
                {
                    "video_path": str(video_path.resolve()),
                    "vggt_depth_path": str(vggt_root.resolve()),
                    "vggt_extrinsics_path": str((vggt_root / "extrinsics.txt").resolve()),
                    "radius_ratio": 1,
                    "text": prompt,
                }
            )
        json_path = input_dir / "new.json"
        json_path.write_text(json.dumps(payload, indent=4, ensure_ascii=False), encoding="utf-8")
        return str(json_path)

    @staticmethod
    def _rewrite_existing_json(
        input_dir: Path,
        staged_videos: Sequence[Path],
        source_json_path: Path,
    ) -> str:
        data = json.loads(source_json_path.read_text(encoding="utf-8"))
        staged_by_name = {video.name: video for video in staged_videos}
        payload = []
        for entry in data:
            source_name = Path(entry["video_path"]).name
            if source_name not in staged_by_name:
                continue
            staged_video = staged_by_name[source_name]
            vggt_root = input_dir / "new_vggt" / staged_video.stem
            payload.append(
                {
                    **entry,
                    "video_path": str(staged_video.resolve()),
                    "vggt_depth_path": str(vggt_root.resolve()),
                    "vggt_extrinsics_path": str((vggt_root / "extrinsics.txt").resolve()),
                }
            )
        if not payload:
            raise ValueError(f"No matching entries found in {source_json_path} for staged videos.")
        json_path = input_dir / "new.json"
        json_path.write_text(json.dumps(payload, indent=4, ensure_ascii=False), encoding="utf-8")
        return str(json_path)

    def _stage_existing_vggt(self, source_dir: Optional[Path], input_dir: Path, enable: bool) -> None:
        if not enable or source_dir is None:
            return
        source_vggt_dir = source_dir / "new_vggt"
        if not source_vggt_dir.exists():
            return
        target_vggt_dir = input_dir / "new_vggt"
        if target_vggt_dir.exists():
            return
        target_vggt_dir.symlink_to(source_vggt_dir)

    def _materialize_runtime_config(
        self,
        base_config_path: str,
        wan_model_dir: str,
        output_root: Path,
    ) -> str:
        OmegaConf = _load_omegaconf()
        config = OmegaConf.load(base_config_path)
        config.wan_model_folder = wan_model_dir
        if getattr(config, "generator", None) is not None and getattr(config.generator, "weight_list", None):
            for weight_info in config.generator.weight_list:
                weight_info.path = wan_model_dir
        runtime_config_path = output_root / "runtime_inference.yaml"
        OmegaConf.save(config=config, f=str(runtime_config_path))
        return str(runtime_config_path)

    def _build_runtime_env(self) -> dict[str, str]:
        """Build an isolated subprocess environment for the bundled runtime.

        Args:
            None: Environment values are derived from the current process and scoped to repo_root.
        """
        env = os.environ.copy()
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "DIFFUSERS_OFFLINE"):
            value = env.get(name)
            if value is not None and value.strip().lower() not in {"1", "true", "yes", "on"}:
                raise ValueError(
                    f"InSpatio-World requires offline model loading; {name}={value!r} is not allowed."
                )
        source_root = str(package_root("worldfoundry").parent)
        pythonpath = [source_root, self.repo_root]
        if env.get("PYTHONPATH"):
            pythonpath.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(pythonpath)
        env_bin = str(Path(sys.executable).expanduser().parent)
        env["PATH"] = env_bin + os.pathsep + env.get("PATH", "")
        env["PYTHON"] = sys.executable
        env["WORLDFOUNDRY_INSPATIO_WORLD_CONFIG_ROOT"] = str(_CONFIG_ROOT)
        env["WORLDFOUNDRY_INSPATIO_WORLD_TRAJECTORY_ROOT"] = str(_TRAJECTORY_ROOT)
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["DIFFUSERS_OFFLINE"] = "1"
        return env

    def plan(
        self,
        config_path: Optional[str] = None,
        traj_txt_path: Optional[str] = None,
        skip_step1: bool = False,
        skip_step2: bool = False,
        skip_step3: bool = False,
        use_tae: bool = False,
        tae_checkpoint_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Describe the in-tree runtime plan without launching inference.

        Args:
            config_path: Optional runtime YAML path, resolved relative to the bundled repo.
            traj_txt_path: Optional trajectory path or bundled trajectory name.
            skip_step1: Whether Florence caption generation will be skipped.
            skip_step2: Whether DA3 depth generation will be skipped.
            skip_step3: Whether v2v model inference will be skipped.
            use_tae: Whether the optional TAE checkpoint is required.
            tae_checkpoint_path: Optional local TAE checkpoint path.
        """
        runtime_root = Path(self.repo_root)
        script_path = runtime_root / "run_inference_pipeline.py"
        raw_config = config_path or self.defaults.get("config_path") or DEFAULT_CONFIG_PATH
        config_candidate = Path(str(raw_config)).expanduser()
        if not config_candidate.is_absolute():
            parts = config_candidate.parts
            data_relative = Path(*parts[1:]) if parts and parts[0] == "configs" else config_candidate
            data_candidate = _CONFIG_ROOT / data_relative
            config_candidate = data_candidate if data_candidate.exists() else runtime_root / config_candidate

        raw_traj = traj_txt_path or self.defaults.get("default_traj_txt_path") or DEFAULT_TRAJECTORY_NAME
        traj_candidate = Path(str(raw_traj)).expanduser()
        if not traj_candidate.exists():
            data_candidate = _CONFIG_ROOT / traj_candidate
            if data_candidate.is_file():
                traj_candidate = data_candidate
            else:
                traj_candidate = _TRAJECTORY_ROOT / Path(str(raw_traj)).name
                if traj_candidate.suffix != ".txt":
                    traj_candidate = traj_candidate.with_suffix(".txt")

        blockers = []

        def resolve_for_plan(label: str, source: str, resolver) -> Path:
            try:
                return Path(resolver(source))
            except (FileNotFoundError, OSError) as exc:
                blockers.append(f"{label} is missing: {exc}")
                return Path(str(source)).expanduser()

        checkpoint_candidate = (
            resolve_for_plan("InSpatio-World checkpoint", self.checkpoint_source, self._resolve_checkpoint_path)
            if not skip_step3
            else Path(str(self.checkpoint_source)).expanduser()
        )
        wan_candidate = (
            resolve_for_plan("Wan model", self.wan_model_source, self._resolve_snapshot_dir)
            if not skip_step3
            else Path(str(self.wan_model_source)).expanduser()
        )
        da3_candidate = (
            resolve_for_plan("DA3 model", self.da3_model_source, self._resolve_snapshot_dir)
            if not skip_step2
            else Path(str(self.da3_model_source)).expanduser()
        )
        florence_candidate = (
            resolve_for_plan("Florence model", self.florence_model_source, self._resolve_snapshot_dir)
            if not skip_step1
            else Path(str(self.florence_model_source)).expanduser()
        )
        tae_candidate: Path | None = None
        if use_tae:
            try:
                resolved_tae = self._resolve_optional_tae_checkpoint(tae_checkpoint_path, True)
                tae_candidate = Path(str(resolved_tae))
            except (FileNotFoundError, OSError) as exc:
                blockers.append(f"TAE checkpoint is missing: {exc}")
                tae_candidate = Path(str(tae_checkpoint_path or "taew2_1.pth")).expanduser()

        if not runtime_root.exists():
            blockers.append(f"Bundled runtime directory is missing: {runtime_root}")
        if not script_path.is_file():
            blockers.append(f"Pipeline script is missing: {script_path}")
        if not config_candidate.is_file():
            blockers.append(f"Runtime config is missing: {config_candidate}")
        if not traj_candidate.is_file():
            blockers.append(f"Trajectory file is missing: {traj_candidate}")

        return {
            "status": "ready" if not blockers else "blocked",
            "runtime_root": str(runtime_root),
            "entrypoint": str(script_path),
            "cwd": str(runtime_root),
            "config_path": str(config_candidate),
            "traj_txt_path": str(traj_candidate),
            "checkpoint_path": str(checkpoint_candidate),
            "wan_model_path": str(wan_candidate),
            "da3_model_path": str(da3_candidate),
            "florence_model_path": str(florence_candidate),
            "tae_checkpoint_path": str(tae_candidate) if tae_candidate is not None else None,
            "skip_steps": {
                "step1": bool(skip_step1),
                "step2": bool(skip_step2),
                "step3": bool(skip_step3),
            },
            "blockers": blockers,
        }

    @torch.no_grad()
    def predict(
        self,
        visual_input,
        traj_txt_path: Optional[str] = None,
        prompt: str = "",
        output_root: Optional[str] = None,
        fps: int = 24,
        step1_gpus: str = "0",
        step2_gpus: str = "0",
        step3_gpus: Optional[str] = None,
        step3_nproc: int = 1,
        master_port: int = 29513,
        config_path: Optional[str] = None,
        json_path: Optional[str] = None,
        skip_step1: bool = False,
        skip_step2: bool = False,
        skip_step3: bool = False,
        relative_to_source: bool = False,
        rotation_only: bool = False,
        adaptive_frame: bool = True,
        freeze_repeat: int = 0,
        freeze_frame: Optional[int] = None,
        use_tae: bool = False,
        tae_checkpoint_path: Optional[str] = None,
        compile_dit: bool = False,
        seed: Optional[int] = None,
        deterministic: Optional[bool] = None,
        show_progress: bool = True,
        return_dict: bool = False,
        **kwargs,
    ):
        """Generate video, with optional reproducible RNG in all child processes.

        An omitted seed retains the stage-three default of zero. Explicit seed
        or deterministic mode also initializes captioning and geometry workers.
        Deterministic mode without an explicit seed uses zero in every stage.
        """
        seed = self.defaults.get("seed") if seed is None else seed
        deterministic = self.defaults.get("deterministic", False) if deterministic is None else deterministic
        env = reproducibility_env(self._build_runtime_env(), seed, deterministic)
        resolved_output_root = Path(output_root or tempfile.mkdtemp(prefix="inspatio_world_"))
        resolved_output_root = resolved_output_root.expanduser().resolve()
        resolved_output_root.mkdir(parents=True, exist_ok=True)

        input_dir, staged_videos, source_dir = self._stage_input_dir(
            visual_input,
            resolved_output_root,
            fps=int(fps),
        )
        self._stage_existing_vggt(source_dir, input_dir, enable=bool(skip_step2))

        resolved_traj = self._resolve_trajectory_path(traj_txt_path)
        base_config_path = self._resolve_config_path(config_path)
        resolved_tae = self._resolve_optional_tae_checkpoint(tae_checkpoint_path, use_tae)

        effective_skip_step1 = bool(skip_step1)
        source_json = None
        if json_path is not None:
            source_json = Path(json_path).expanduser().resolve()
        elif source_dir is not None and (source_dir / "new.json").exists():
            source_json = (source_dir / "new.json").resolve()

        if prompt:
            self._write_prompt_override_json(input_dir, staged_videos, prompt)
            effective_skip_step1 = True
        elif effective_skip_step1 and source_json is not None:
            self._rewrite_existing_json(input_dir, staged_videos, source_json)
        elif effective_skip_step1:
            raise ValueError(
                "skip_step1=True requires either prompt text, json_path, or an input directory containing new.json."
            )

        if skip_step2 and not (input_dir / "new_vggt").exists():
            raise ValueError(
                "skip_step2=True requires precomputed new_vggt outputs in the input directory."
            )

        resolved_checkpoint = (
            self._resolve_checkpoint_path(self.checkpoint_source)
            if not skip_step3
            else str(self.checkpoint_source)
        )
        resolved_wan_dir = (
            self._resolve_snapshot_dir(self.wan_model_source)
            if not skip_step3
            else str(self.wan_model_source)
        )
        resolved_da3_dir = (
            self._resolve_snapshot_dir(self.da3_model_source)
            if not skip_step2
            else str(self.da3_model_source)
        )
        resolved_florence_dir = (
            self._resolve_snapshot_dir(self.florence_model_source)
            if not effective_skip_step1
            else str(self.florence_model_source)
        )
        resolved_config = (
            self._materialize_runtime_config(base_config_path, resolved_wan_dir, resolved_output_root)
            if not skip_step3
            else base_config_path
        )
        inherited_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if inherited_devices:
            if step1_gpus == "0":
                step1_gpus = inherited_devices
            if step2_gpus == "0":
                step2_gpus = inherited_devices
            if step3_gpus is None or step3_gpus == "0":
                step3_gpus = inherited_devices

        command = [
            sys.executable,
            str((Path(self.repo_root) / "run_inference_pipeline.py").resolve()),
            "--input_dir",
            str(input_dir),
            "--traj_txt_path",
            resolved_traj,
            "--checkpoint_path",
            resolved_checkpoint,
            "--config_path",
            resolved_config,
            "--da3_model_path",
            resolved_da3_dir,
            "--wan_model_path",
            resolved_wan_dir,
            "--florence_model_path",
            resolved_florence_dir,
            "--output_folder",
            str(resolved_output_root),
            "--step1_gpus",
            str(step1_gpus),
            "--step2_gpus",
            str(step2_gpus),
            "--step3_gpus",
            str(step3_gpus or self.defaults.get("step3_gpus", "0")),
            "--step3_nproc",
            str(int(step3_nproc)),
            "--master_port",
            str(int(master_port)),
        ]

        if effective_skip_step1:
            command.append("--skip_step1")
        if skip_step2:
            command.append("--skip_step2")
        if skip_step3:
            command.append("--skip_step3")
        if relative_to_source:
            command.append("--relative_to_source")
        if rotation_only:
            command.append("--rotation_only")
        if not adaptive_frame:
            command.append("--disable_adaptive_frame")
        if int(freeze_repeat) > 0:
            command.extend(["--freeze_repeat", str(int(freeze_repeat))])
        if freeze_frame is not None:
            command.extend(["--freeze_frame", str(int(freeze_frame))])
        if use_tae:
            command.append("--use_tae")
            if resolved_tae is not None:
                command.extend(["--tae_checkpoint_path", resolved_tae])
        if compile_dit:
            command.append("--compile_dit")
        if seed is not None:
            command.extend(["--seed", str(seed)])
        if deterministic:
            command.append("--deterministic")
        stdout = None if show_progress else subprocess.DEVNULL
        stderr = None if show_progress else subprocess.STDOUT

        subprocess.run(
            command,
            check=True,
            cwd=self.repo_root,
            env=env,
            stdout=stdout,
            stderr=stderr,
        )

        generated_video_paths = sorted(
            path.resolve()
            for path in resolved_output_root.rglob("*-pred_video_rank*.mp4")
        )

        videos = [load_video_frames(str(path)) for path in generated_video_paths]
        result = {
            "video": videos[0] if len(videos) == 1 else None,
            "videos": videos,
            "generated_video_path": str(generated_video_paths[0]) if len(generated_video_paths) == 1 else None,
            "generated_video_paths": [str(path) for path in generated_video_paths],
            "output_root": str(resolved_output_root),
            "input_dir": str(input_dir),
            "traj_txt_path": resolved_traj,
            "checkpoint_path": resolved_checkpoint,
            "wan_model_path": resolved_wan_dir,
            "da3_model_path": resolved_da3_dir,
            "florence_model_path": resolved_florence_dir,
            "config_path": resolved_config,
            "prompt": prompt,
            "seed": int(env.get(SEED_ENV, "0")),
            "deterministic": env.get("WORLDFOUNDRY_DETERMINISTIC", "").strip().lower() in {"1", "true", "yes", "on"},
        }
        if skip_step3 and not generated_video_paths:
            if return_dict:
                return result
            return None

        if not generated_video_paths:
            raise RuntimeError(
                f"InSpatio-World finished but no generated video was found under {resolved_output_root}."
            )

        if return_dict:
            return result
        if len(videos) == 1:
            return videos[0]
        return videos
