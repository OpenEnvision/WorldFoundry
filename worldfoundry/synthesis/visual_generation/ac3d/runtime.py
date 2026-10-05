from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import imageio.v3 as iio
import numpy as np

from worldfoundry.core.io import file_sha256
from worldfoundry.core.io.paths import checkpoint_root_candidates
from worldfoundry.core.media.codecs.video import materialize_video_input

DEFAULT_AC3D_REPO_ID = "snap-research/ac3d"
DEFAULT_COGVIDEOX_2B_REPO = "THUDM/CogVideoX-2b"
DEFAULT_COGVIDEOX_5B_REPO = "THUDM/CogVideoX-5b"
DEFAULT_PROMPT = (
    "Three fluffy sheep sit side by side at a rustic wooden table, each eagerly digging into "
    "their bowls of spaghetti. The pasta is tangled playfully around their woolly faces, and "
    "the bright red sauce splatters across their fur. The scene takes place in a lush, green "
    "meadow surrounded by rolling hills, with a few grazing cows in the background."
)


def _local_repo_candidates() -> tuple[Path, ...]:
    candidates: list[Path] = []
    configured = os.environ.get("WORLDFOUNDRY_AC3D_REPO_ROOT")
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.append(Path(__file__).resolve().parent / "ac3d_runtime")
    return tuple(candidates)


def _ckpt_root_candidates() -> tuple[Path, ...]:
    return checkpoint_root_candidates("ac3d", specific_env="WORLDFOUNDRY_AC3D_CKPT_ROOT")


def _base_model_candidates(variant: str) -> tuple[Path, ...]:
    name = "CogVideoX-5b" if variant == "5b" else "CogVideoX-2b"
    values = [
        os.environ.get("WORLDFOUNDRY_AC3D_BASE_MODEL_PATH"),
        os.environ.get("WORLDFOUNDRY_COGVIDEOX_ROOT") and str(Path(os.environ["WORLDFOUNDRY_COGVIDEOX_ROOT"]) / name),
        *(str(path) for path in checkpoint_root_candidates(name)),
        *(str(path) for path in checkpoint_root_candidates(f"THUDM--{name}")),
    ]
    return tuple(Path(value).expanduser() for value in values if value)


def _first_existing(paths: Sequence[Path]) -> str | None:
    for path in paths:
        if path.exists():
            return str(path)
    return None


def _variant(value: Any) -> str:
    text = str(value or "2b").strip().lower()
    if text in {"2", "2b", "cogvideox-2b"}:
        return "2b"
    if text in {"5", "5b", "cogvideox-5b"}:
        return "5b"
    raise ValueError(f"Unsupported AC3D variant: {value!r}. Expected '2b' or '5b'.")


def _default_controlnet_path(variant: str) -> str:
    shard_dir = "5B" if variant == "5b" else "2B"
    for root in _ckpt_root_candidates():
        candidate = root / shard_dir / "checkpoint-10000.pt"
        if candidate.is_file():
            return str(candidate)
    return str(_ckpt_root_candidates()[0] / shard_dir / "checkpoint-10000.pt") if _ckpt_root_candidates() else ""


def _default_base_model_path(variant: str) -> str:
    local = _first_existing(_base_model_candidates(variant))
    if local:
        return local
    return DEFAULT_COGVIDEOX_5B_REPO if variant == "5b" else DEFAULT_COGVIDEOX_2B_REPO


def _default_repo_root() -> str:
    configured = os.environ.get("WORLDFOUNDRY_AC3D_REPO_ROOT")
    if configured:
        return str(Path(configured).expanduser())
    local = _first_existing(_local_repo_candidates())
    candidates = _local_repo_candidates()
    return local or str(candidates[0])


def _variant_controlnet_kwargs(variant: str) -> dict[str, Any]:
    return {
        "controlnet_transformer_num_attn_heads": 4,
        "controlnet_transformer_attention_head_dim": 64 if variant == "5b" else 32,
        "controlnet_transformer_out_proj_dim_factor": 64,
        "controlnet_transformer_out_proj_dim_zero_init": True,
    }


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _load_video_frames(video_path: str | Path) -> np.ndarray:
    frames = [np.asarray(frame).astype(np.uint8) for frame in iio.imiter(str(video_path))]
    if not frames:
        raise ValueError(f"No frames found in AC3D output video: {video_path}")
    return np.stack(frames, axis=0)


def _normalize_realestate10k_root(video_root_dir: str | Path, annotation_json: str | Path) -> Path:
    root = Path(video_root_dir).expanduser()
    annotation = Path(annotation_json).expanduser()
    if not annotation.is_absolute():
        for candidate_root in (root, root.parent if root.name == "video_clips" else root):
            candidate = candidate_root / annotation
            if candidate.is_file():
                annotation = candidate
                break
    if root.name != "video_clips" or not annotation.is_file():
        return root
    try:
        rows = json.loads(annotation.read_text(encoding="utf-8"))
    except Exception:
        return root
    if isinstance(rows, list) and rows:
        clip_path = str(rows[0].get("clip_path", ""))
        if clip_path.startswith("video_clips/"):
            return root.parent
    return root


def _camera_pose_rows(value: Any, *, num_frames: int) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float64)
    expected_shape = (int(num_frames), 19)
    if rows.shape != expected_shape:
        raise ValueError(
            "AC3D camera_pose_rows must contain one 19-value RealEstate10K row "
            f"per output frame; expected {expected_shape}, got {rows.shape}."
        )
    if not np.isfinite(rows).all():
        raise ValueError("AC3D camera_pose_rows contains non-finite values.")
    for frame_index, flat_w2c in enumerate(rows[:, 7:19]):
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :4] = flat_w2c.reshape(3, 4)
        if abs(float(np.linalg.det(w2c[:3, :3]))) < 1e-8:
            raise ValueError(f"AC3D camera_pose_rows[{frame_index}] has a singular W2C rotation.")
    return rows


def _decodable_video_frame_count(video_path: Path, *, stop_at: int) -> int:
    imageio_error: Exception | None = None
    decoded_frames = 0
    try:
        for _frame in iio.imiter(str(video_path)):
            decoded_frames += 1
            if decoded_frames >= int(stop_at):
                return decoded_frames
        return decoded_frames
    except Exception as error:
        imageio_error = error

    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        probe = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_read_frames,nb_frames",
                "-of",
                "json",
                str(video_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if probe.returncode == 0:
            try:
                stream = json.loads(probe.stdout)["streams"][0]
                for field in ("nb_read_frames", "nb_frames"):
                    value = str(stream.get(field, ""))
                    if value.isdigit():
                        return int(value)
            except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                pass
    raise ValueError(f"could not decode or probe video frames: {imageio_error}")


def stage_direct_camera_dataset(
    *,
    video: Any,
    camera_pose_rows: Any,
    root: str | Path,
    num_frames: int,
    fps: int = 8,
    caption: str = "",
) -> dict[str, str]:
    """Materialize the minimal RealEstate10K layout used by official AC3D inference."""

    root_path = Path(root).expanduser().resolve()
    video_dir = root_path / "video_clips"
    pose_dir = root_path / "poses"
    annotation_dir = root_path / "annotations"
    for directory in (video_dir, pose_dir, annotation_dir):
        directory.mkdir(parents=True, exist_ok=True)

    source_path = Path(
        materialize_video_input(
            video,
            output_dir=str(video_dir),
            filename="input.mp4",
            fps=int(fps),
        )
    ).expanduser().resolve()
    suffix = source_path.suffix.lower() or ".mp4"
    staged_video = video_dir / f"input{suffix}"
    if source_path != staged_video.resolve(strict=False):
        try:
            staged_video.symlink_to(source_path)
        except OSError:
            shutil.copyfile(source_path, staged_video)

    try:
        decoded_frames = _decodable_video_frame_count(staged_video, stop_at=int(num_frames))
    except Exception as error:
        raise ValueError(f"AC3D could not decode source video {source_path}: {error}") from error
    if decoded_frames < int(num_frames):
        raise ValueError(
            f"AC3D source video has only {decoded_frames} decodable frames; "
            f"{int(num_frames)} are required for stride-1 camera control."
        )

    rows = _camera_pose_rows(camera_pose_rows, num_frames=int(num_frames))
    pose_path = pose_dir / "input.txt"
    pose_lines = ["worldfoundry-ac3d-direct-input"]
    pose_lines.extend(" ".join(f"{float(item):.17g}" for item in row) for row in rows)
    pose_path.write_text("\n".join(pose_lines) + "\n", encoding="utf-8")

    annotation_path = annotation_dir / "test.json"
    annotation_path.write_text(
        json.dumps(
            [
                {
                    "clip_path": staged_video.relative_to(root_path).as_posix(),
                    "clip_name": "input",
                    "caption": str(caption),
                    "pose_file": pose_path.relative_to(root_path).as_posix(),
                }
            ],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "video_root_dir": str(root_path),
        "annotation_json": annotation_path.relative_to(root_path).as_posix(),
        "video_path": str(staged_video),
        "pose_path": str(pose_path),
        "annotation_path": str(annotation_path),
    }


class AC3DRuntime:
    """WorldFoundry adapter for the official AC3D CogVideoX-ControlNet runtime."""

    MODEL_ID = "ac3d"
    DISPLAY_NAME = "AC3D"

    def __init__(
        self,
        *,
        model_id: str = MODEL_ID,
        runtime_root: str | Path | None = None,
        base_model_path: str | Path | None = None,
        controlnet_model_path: str | Path | None = None,
        variant: str = "2b",
        device: str = "cuda",
        python_executable: str | None = None,
        defaults: Mapping[str, Any] | None = None,
    ) -> None:
        resolved_variant = _variant(variant)
        self.model_id = model_id
        self.model_name = self.DISPLAY_NAME
        self.generation_type = "camera_control_video"
        self.variant = resolved_variant
        self.runtime_root = str(Path(runtime_root or _default_repo_root()).expanduser())
        self.base_model_path = str(base_model_path or _default_base_model_path(resolved_variant))
        self.controlnet_model_path = str(controlnet_model_path or _default_controlnet_path(resolved_variant))
        self.device = device
        self.python_executable = python_executable or sys.executable
        self.defaults = {
            "annotation_json": "annotations/test.json",
            "controlnet_weights": 1.0,
            "controlnet_guidance_start": 0.0,
            "controlnet_guidance_end": 0.4,
            "use_dynamic_cfg": True,
            "num_inference_steps": 50,
            "guidance_scale": 6.0,
            "num_videos_per_prompt": 1,
            "dtype": "bfloat16",
            "seed": 42,
            "num_frames": 49,
            "height": 480,
            "width": 720,
            "stride_min": 2,
            "stride_max": 2,
            "start_camera_idx": 0,
            "end_camera_idx": 1,
            "downscale_coef": 8,
            "controlnet_input_channels": 6,
            **_variant_controlnet_kwargs(resolved_variant),
        }
        if defaults:
            self.defaults.update(dict(defaults))

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Any = None,
        *,
        device: str | None = None,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> "AC3DRuntime":
        options = dict(pretrained_model_path) if isinstance(pretrained_model_path, Mapping) else {}
        if pretrained_model_path is not None and not isinstance(pretrained_model_path, Mapping):
            options["controlnet_model_path"] = str(pretrained_model_path)
        options.update(kwargs)
        variant = _variant(options.get("variant") or options.get("model_variant") or "2b")
        defaults = {
            key: options.pop(key)
            for key in tuple(options)
            if key
            in {
                "annotation_json",
                "controlnet_weights",
                "controlnet_guidance_start",
                "controlnet_guidance_end",
                "use_dynamic_cfg",
                "num_inference_steps",
                "guidance_scale",
                "num_videos_per_prompt",
                "dtype",
                "seed",
                "num_frames",
                "height",
                "width",
                "stride_min",
                "stride_max",
                "start_camera_idx",
                "end_camera_idx",
                "downscale_coef",
                "controlnet_input_channels",
                "controlnet_transformer_num_attn_heads",
                "controlnet_transformer_attention_head_dim",
                "controlnet_transformer_out_proj_dim_factor",
                "controlnet_transformer_out_proj_dim_zero_init",
            }
        }
        return cls(
            model_id=str(options.get("model_id") or options.get("profile_id") or model_id or cls.MODEL_ID),
            runtime_root=options.get("runtime_root") or options.get("repo_root") or _default_repo_root(),
            base_model_path=options.get("base_model_path") or options.get("cogvideox_model_path"),
            controlnet_model_path=(
                options.get("controlnet_model_path")
                or options.get("checkpoint_path")
                or options.get("ckpt_path")
                or _default_controlnet_path(variant)
            ),
            variant=variant,
            device=str(device or options.get("device") or "cuda"),
            python_executable=options.get("python_executable"),
            defaults=defaults,
        )

    def runtime_plan(self, **overrides: Any) -> dict[str, Any]:
        options = {**self.defaults, **overrides}
        runtime_root_exists = Path(self.runtime_root).is_dir()
        controlnet_model_exists = Path(self.controlnet_model_path).is_file()
        return {
            "status": "prepared" if runtime_root_exists and controlnet_model_exists else "blocked",
            "backend": "official_ac3d_in_tree_runtime",
            "backend_quality": "official_in_tree_runtime",
            "model_id": self.model_id,
            "variant": self.variant,
            "runtime_root": self.runtime_root,
            "runtime_root_exists": runtime_root_exists,
            "base_model_path": self.base_model_path,
            "base_model_exists": Path(self.base_model_path).exists(),
            "controlnet_model_path": self.controlnet_model_path,
            "controlnet_model_exists": controlnet_model_exists,
            "video_root_dir": str(options.get("video_root_dir") or ""),
            "annotation_json": str(options.get("annotation_json") or self.defaults["annotation_json"]),
            "python_executable": self.python_executable,
        }

    def _command(self, output_dir: Path, prompt: str, options: Mapping[str, Any]) -> list[str]:
        command = [
            self.python_executable,
            "-m",
            "worldfoundry.synthesis.visual_generation.ac3d.runtime",
            "--run-official",
            "--runtime_root",
            self.runtime_root,
            "--prompt",
            prompt or DEFAULT_PROMPT,
            "--video_root_dir",
            str(options["video_root_dir"]),
            "--annotation_json",
            str(options["annotation_json"]),
            "--base_model_path",
            self.base_model_path,
            "--controlnet_model_path",
            self.controlnet_model_path,
            "--output_path",
            str(output_dir),
            "--controlnet_weights",
            str(float(options["controlnet_weights"])),
            "--controlnet_guidance_start",
            str(float(options["controlnet_guidance_start"])),
            "--controlnet_guidance_end",
            str(float(options["controlnet_guidance_end"])),
            "--num_inference_steps",
            str(int(options["num_inference_steps"])),
            "--guidance_scale",
            str(float(options["guidance_scale"])),
            "--num_videos_per_prompt",
            str(int(options["num_videos_per_prompt"])),
            "--dtype",
            str(options["dtype"]),
            "--seed",
            str(int(options["seed"])),
            "--stride_min",
            str(int(options["stride_min"])),
            "--stride_max",
            str(int(options["stride_max"])),
            "--height",
            str(int(options["height"])),
            "--width",
            str(int(options["width"])),
            "--num_frames",
            str(int(options["num_frames"])),
            "--start_camera_idx",
            str(int(options["start_camera_idx"])),
            "--end_camera_idx",
            str(int(options["end_camera_idx"])),
            "--downscale_coef",
            str(int(options["downscale_coef"])),
            "--controlnet_input_channels",
            str(int(options["controlnet_input_channels"])),
        ]
        if _as_bool(options.get("use_dynamic_cfg")):
            command.append("--use_dynamic_cfg")
        for key in (
            "lora_path",
            "lora_rank",
            "controlnet_transformer_num_attn_heads",
            "controlnet_transformer_attention_head_dim",
            "controlnet_transformer_out_proj_dim_factor",
        ):
            value = options.get(key)
            if value is not None:
                command.extend([f"--{key}", str(value)])
        if _as_bool(options.get("controlnet_transformer_out_proj_dim_zero_init")):
            command.append("--controlnet_transformer_out_proj_dim_zero_init")
        return command

    def _subprocess_env(self) -> dict[str, str]:
        env = os.environ.copy()
        src_root = Path(__file__).resolve().parents[4]
        pythonpath = [str(src_root), self.runtime_root, env.get("PYTHONPATH", "")]
        env["PYTHONPATH"] = os.pathsep.join(item for item in pythonpath if item)
        if self.device.startswith("cuda:"):
            env.setdefault("CUDA_VISIBLE_DEVICES", self.device.split(":", 1)[1])
        return env

    def predict(
        self,
        prompt: str = "",
        images: Any = None,
        video: Any = None,
        interactions: Sequence[Any] = (),
        output_path: str | Path | None = None,
        fps: int | None = None,
        return_dict: bool = True,
        show_progress: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any] | np.ndarray:
        del images
        options = {**self.defaults, **kwargs}
        if "video_root_dir" not in options or not options["video_root_dir"]:
            camera_rows = options.pop("camera_pose_rows", None)
            if video is None or camera_rows is None:
                raise ValueError(
                    "AC3D requires either video_root_dir for an existing RealEstate10K-style dataset, "
                    "or direct video plus camera_pose_rows."
                )
            with tempfile.TemporaryDirectory(prefix="worldfoundry_ac3d_input_") as temporary_root:
                staged = stage_direct_camera_dataset(
                    video=video,
                    camera_pose_rows=camera_rows,
                    root=temporary_root,
                    num_frames=int(options["num_frames"]),
                    fps=int(fps or 8),
                    caption=prompt,
                )
                options.update(
                    {
                        "video_root_dir": staged["video_root_dir"],
                        "annotation_json": staged["annotation_json"],
                        "stride_min": 1,
                        "stride_max": 1,
                        "start_camera_idx": 0,
                        "end_camera_idx": 1,
                    }
                )
                return self.predict(
                    prompt=prompt,
                    video=None,
                    interactions=(),
                    output_path=output_path,
                    fps=fps,
                    return_dict=return_dict,
                    show_progress=show_progress,
                    **options,
                )
        if interactions:
            indices = [int(item) for item in interactions]
            options["start_camera_idx"] = min(indices)
            options["end_camera_idx"] = max(indices) + 1
        options["video_root_dir"] = str(
            _normalize_realestate10k_root(options["video_root_dir"], options["annotation_json"])
        )

        output_target = Path(output_path).expanduser() if output_path is not None else None
        is_video_target = output_target is not None and output_target.suffix.lower() == ".mp4"
        if is_video_target:
            assert output_target is not None
            output_target.parent.mkdir(parents=True, exist_ok=True)
            # The official CLI skips generation when 00000_out.mp4 already exists.
            # Isolate each file-targeted run so a prior run cannot be returned as fresh output.
            output_dir = Path(
                tempfile.mkdtemp(prefix=f".{output_target.stem}_ac3d_", dir=output_target.parent)
            ).resolve()
        else:
            output_dir = Path(output_target or tempfile.mkdtemp(prefix="ac3d_")).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        plan = self.runtime_plan(**options)
        if not plan["runtime_root_exists"]:
            raise FileNotFoundError(f"AC3D runtime_root does not exist: {self.runtime_root}")
        if not plan["controlnet_model_exists"]:
            raise FileNotFoundError(f"AC3D controlnet_model_path does not exist: {self.controlnet_model_path}")
        if not Path(str(options["video_root_dir"])).expanduser().is_dir():
            raise FileNotFoundError(f"AC3D video_root_dir does not exist: {options['video_root_dir']}")

        command = self._command(output_dir, prompt, options)
        stdout = None if show_progress else subprocess.DEVNULL
        stderr = None if show_progress else subprocess.STDOUT
        try:
            subprocess.run(
                command,
                check=True,
                cwd=self.runtime_root,
                env=self._subprocess_env(),
                stdout=stdout,
                stderr=stderr,
            )
        except subprocess.CalledProcessError as error:
            raise RuntimeError("AC3D generation failed. Check runtime dependencies, dataset paths, and GPU memory.") from error

        start_idx = int(options["start_camera_idx"])
        generated_path = output_dir / f"{start_idx:05d}_out.mp4"
        if not generated_path.is_file():
            candidates = sorted(output_dir.glob("*_out.mp4"))
            if not candidates:
                raise FileNotFoundError(f"AC3D output video not found under {output_dir}")
            generated_path = candidates[0]

        artifact_path = generated_path
        if is_video_target:
            assert output_target is not None
            output_target = output_target.resolve()
            if generated_path != output_target:
                shutil.copyfile(generated_path, output_target)
            artifact_path = output_target
            shutil.rmtree(output_dir)

        frames = _load_video_frames(artifact_path)
        result = {
            "status": "success",
            "model_id": self.model_id,
            "artifact_kind": "generated_video",
            "artifact_path": str(artifact_path),
            "generated_video_path": str(artifact_path),
            "video": frames,
            "frames": frames,
            "video_sha256": file_sha256(artifact_path),
            "runtime": "worldfoundry.ac3d.official_in_tree_runtime",
            "backend_quality": "official_in_tree_runtime",
            "runtime_plan": plan,
        }
        if return_dict:
            return result
        return frames


def _prepend_sys_path(path_value: str | Path) -> None:
    resolved = str(Path(path_value).expanduser().resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def _run_official(args: argparse.Namespace) -> None:
    _prepend_sys_path(args.runtime_root)
    import inference.cli_demo_camera as cli_demo_camera
    import torch

    # Upstream's generate_video reads the parsed CLI namespace as a module global
    # for a few ControlNet transformer options. Mirror the official CLI execution
    # path when calling it from the WorldFoundry bridge.
    cli_demo_camera.args = args

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    cli_demo_camera.generate_video(
        prompt=args.prompt,
        video_root_dir=args.video_root_dir,
        annotation_json=args.annotation_json,
        base_model_path=args.base_model_path,
        controlnet_model_path=args.controlnet_model_path,
        controlnet_weights=args.controlnet_weights,
        controlnet_guidance_start=args.controlnet_guidance_start,
        controlnet_guidance_end=args.controlnet_guidance_end,
        use_dynamic_cfg=args.use_dynamic_cfg,
        lora_path=args.lora_path,
        lora_rank=args.lora_rank,
        output_path=args.output_path,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_videos_per_prompt=args.num_videos_per_prompt,
        dtype=dtype,
        seed=args.seed,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        stride_min=args.stride_min,
        stride_max=args.stride_max,
        start_camera_idx=args.start_camera_idx,
        end_camera_idx=args.end_camera_idx,
        controlnet_transformer_num_attn_heads=args.controlnet_transformer_num_attn_heads,
        controlnet_transformer_attention_head_dim=args.controlnet_transformer_attention_head_dim,
        controlnet_transformer_out_proj_dim_factor=args.controlnet_transformer_out_proj_dim_factor,
        controlnet_transformer_out_proj_dim_zero_init=args.controlnet_transformer_out_proj_dim_zero_init,
        downscale_coef=args.downscale_coef,
        controlnet_input_channels=args.controlnet_input_channels,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="WorldFoundry AC3D official runtime bridge")
    parser.add_argument("--run-official", action="store_true")
    parser.add_argument("--runtime_root", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--video_root_dir", required=True)
    parser.add_argument("--annotation_json", default="annotations/test.json")
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--controlnet_model_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--controlnet_weights", type=float, default=1.0)
    parser.add_argument("--controlnet_guidance_start", type=float, default=0.0)
    parser.add_argument("--controlnet_guidance_end", type=float, default=0.4)
    parser.add_argument("--use_dynamic_cfg", action="store_true")
    parser.add_argument("--lora_path", default=None)
    parser.add_argument("--lora_rank", type=int, default=128)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--num_videos_per_prompt", type=int, default=1)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stride_min", type=int, default=2)
    parser.add_argument("--stride_max", type=int, default=2)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--start_camera_idx", type=int, default=0)
    parser.add_argument("--end_camera_idx", type=int, default=1)
    parser.add_argument("--controlnet_transformer_num_attn_heads", type=int, default=None)
    parser.add_argument("--controlnet_transformer_attention_head_dim", type=int, default=None)
    parser.add_argument("--controlnet_transformer_out_proj_dim_factor", type=int, default=None)
    parser.add_argument("--controlnet_transformer_out_proj_dim_zero_init", action="store_true")
    parser.add_argument("--downscale_coef", type=int, default=8)
    parser.add_argument("--controlnet_input_channels", type=int, default=6)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.run_official:
        _run_official(args)
    else:
        print(json.dumps({"status": "ready", "runtime_root": args.runtime_root}))


if __name__ == "__main__":
    main()


__all__ = ["AC3DRuntime", "stage_direct_camera_dataset"]
