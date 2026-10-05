"""Process adapter for the in-tree LingBot-Video inference runtime."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.core import cuda_visible_devices_from_device
from worldfoundry.runtime.assets import expand_worldfoundry_path


SOURCE_ROOT = Path(__file__).resolve().parents[4]
INFERENCE_ROOT = Path(__file__).resolve().parent
RUNNER_MODULE = "worldfoundry.synthesis.visual_generation.lingbot_video.runner"
SUPPORTED_MODES = frozenset({"t2i", "t2v", "ti2v"})
REQUIRED_CHECKPOINT_COMPONENTS = (
    "model_index.json",
    "processor",
    "scheduler",
    "text_encoder",
    "transformer",
    "vae",
)
REQUIRED_RUNTIME_FILES = (
    "runner.py",
    "pipeline_lingbot_video.py",
    "pipeline_lingbot_video_i2v.py",
    "transformer_lingbot_video.py",
    "scheduling_flow_unipc.py",
    "utils.py",
)


def _expand_path(value: str | Path | None) -> Path | None:
    if value is None or not str(value).strip():
        return None
    expanded = os.path.expandvars(str(value))
    return Path(expand_worldfoundry_path(expanded)).expanduser().resolve()


def _first_path(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        return str(value)
    if isinstance(value, (list, tuple)) and value:
        return str(value[0])
    return None


def _append_value(command: list[str], flag: str, value: Any) -> None:
    if value is not None and value != "":
        command.extend((flag, str(value)))


def _validate_frame_count_request(
    request: Mapping[str, Any], prompt_json: Path | None, mode: str
) -> None:
    """Catch frame requests that the official CLI would silently override."""
    if mode == "t2i" or request.get("num_frames") is None:
        return
    duration = request.get("duration")
    if duration is None and prompt_json is not None:
        sample = json.loads(prompt_json.read_text(encoding="utf-8"))
        if isinstance(sample, list):
            sample = sample[0] if sample else {}
        if isinstance(sample, dict):
            duration = sample.get("duration")
    if duration is None:
        return
    fps = int(request.get("fps") or 24)
    # Match runner's num_frames_from_duration: frames are rounded to 4n+1.
    frame_count = int(float(duration) * fps)
    effective_frames = ((frame_count - 1) // 4 + 1) * 4 + 1
    requested_frames = int(request["num_frames"])
    if requested_frames != effective_frames:
        raise ValueError(
            "LingBot-Video num_frames conflicts with duration: "
            f"requested {requested_frames}, but duration={duration} at fps={fps} "
            f"produces {effective_frames} frames. Set num_frames={effective_frames} "
            "or omit num_frames to use the prompt duration."
        )


@dataclass(frozen=True)
class LingBotVideoRuntimePlan:
    """Serializable launch plan for one LingBot-Video inference request."""

    command: tuple[str, ...]
    env: Mapping[str, str]
    workdir: str
    checkpoint_dir: str
    output_path: str
    refiner_output_path: str | None
    mode: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": list(self.command),
            "env": dict(self.env),
            "workdir": self.workdir,
            "checkpoint_dir": self.checkpoint_dir,
            "output_path": self.output_path,
            "refiner_output_path": self.refiner_output_path,
            "mode": self.mode,
        }


class LingBotVideoRuntime:
    """Checkpoint-aware adapter around the vendored LingBot-Video CLI runner."""

    def __init__(
        self,
        checkpoint_dir: str | Path | None,
        *,
        device: str = "cuda",
        python_executable: str | Path | None = None,
        inference_root: str | Path | None = None,
    ) -> None:
        self.checkpoint_dir = _expand_path(checkpoint_dir)
        self.device = str(device)
        self.python_executable = (
            Path(python_executable or sys.executable).expanduser().resolve()
        )
        self.inference_root = _expand_path(inference_root) or INFERENCE_ROOT

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: str | Path | None,
        *,
        device: str = "cuda",
        **kwargs: Any,
    ) -> "LingBotVideoRuntime":
        return cls(checkpoint_dir, device=device, **kwargs)

    def missing_checkpoint_files(
        self,
        *,
        run_refiner: bool = False,
        refiner_model_dir: str | Path | None = None,
    ) -> tuple[str, ...]:
        if self.checkpoint_dir is None:
            return REQUIRED_CHECKPOINT_COMPONENTS
        required = [
            self.checkpoint_dir / component
            for component in REQUIRED_CHECKPOINT_COMPONENTS
        ]
        if run_refiner:
            refiner_root = _expand_path(refiner_model_dir) or self.checkpoint_dir
            required.append(refiner_root / "refiner")
        return tuple(str(path) for path in required if not path.exists())

    def missing_runtime_files(self) -> tuple[str, ...]:
        return tuple(
            str(self.inference_root / name)
            for name in REQUIRED_RUNTIME_FILES
            if not (self.inference_root / name).is_file()
        )

    def preflight(
        self,
        *,
        run_refiner: bool = False,
        refiner_model_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        missing_checkpoint = self.missing_checkpoint_files(
            run_refiner=run_refiner,
            refiner_model_dir=refiner_model_dir,
        )
        missing_runtime = self.missing_runtime_files()
        checkpoint_ready = (
            self.checkpoint_dir is not None
            and self.checkpoint_dir.is_dir()
            and not missing_checkpoint
        )
        runtime_ready = self.inference_root.is_dir() and not missing_runtime
        return {
            "status": "ready" if checkpoint_ready and runtime_ready else "blocked",
            "checkpoint_ready": checkpoint_ready,
            "runtime_ready": runtime_ready,
            "checkpoint_dir": str(self.checkpoint_dir or ""),
            "inference_root": str(self.inference_root),
            "missing_checkpoint_files": list(missing_checkpoint),
            "missing_runtime_files": list(missing_runtime),
            "runner_module": RUNNER_MODULE,
            "requires": [
                "torch and torchvision builds compatible with the selected CUDA runtime",
                "diffusers>=0.38.0",
                "transformers>=5.0,<6",
                "accelerate, peft, safetensors, imageio, and imageio-ffmpeg",
            ],
        }

    def build_plan(
        self,
        *,
        request: Mapping[str, Any],
        output_path: str | Path,
    ) -> LingBotVideoRuntimePlan:
        if self.checkpoint_dir is None:
            raise ValueError(
                "LingBot-Video inference requires a local checkpoint directory."
            )

        mode = str(request.get("mode") or "t2v").strip().lower()
        if mode not in SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported LingBot-Video mode {mode!r}; expected one of {sorted(SUPPORTED_MODES)}."
            )

        output = Path(output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        prompt_json = _expand_path(request.get("prompt_json"))
        prompt = request.get("prompt")
        if prompt_json is None and (prompt is None or not str(prompt).strip()):
            raise ValueError(
                "LingBot-Video requires prompt_json or a structured prompt string."
            )
        _validate_frame_count_request(request, prompt_json, mode)
        if mode == "ti2v" and not _first_path(
            request.get("image") or request.get("images")
        ):
            raise ValueError("LingBot-Video ti2v inference requires an image.")

        base_args = [
            "-m",
            RUNNER_MODULE,
            "--backend",
            str(request.get("backend") or "diffusers"),
            "--model_dir",
            str(self.checkpoint_dir),
            "--mode",
            mode,
            "--output",
            str(output),
        ]
        if prompt_json is not None:
            _append_value(base_args, "--prompt_json", prompt_json)
        else:
            _append_value(base_args, "--prompt", prompt)

        image = _first_path(request.get("image") or request.get("images"))
        _append_value(base_args, "--image", image)
        _append_value(base_args, "--negative_prompt", request.get("negative_prompt"))
        _append_value(
            base_args, "--negative_prompt_json", request.get("negative_prompt_json")
        )

        value_flags = {
            "resolution": "--resolution",
            "ratio": "--ratio",
            "duration": "--duration",
            "height": "--height",
            "width": "--width",
            "num_frames": "--num_frames",
            "steps": "--steps",
            "guidance_scale": "--guidance_scale",
            "shift": "--shift",
            "seed": "--seed",
            "fps": "--fps",
            "default_dtype": "--default_dtype",
            "transformer_dtype": "--transformer_dtype",
            "transformer_subfolder": "--transformer_subfolder",
            "text_encoder_dtype": "--text_encoder_dtype",
            "vae_dtype": "--vae_dtype",
            "diffusers_attn_backend": "--diffusers_attn_backend",
            "cfg_parallel_degree": "--cfg_parallel_degree",
            "context_parallel_degree": "--context_parallel_degree",
            "refiner_transformer_subfolder": "--refiner_transformer_subfolder",
            "refiner_height": "--refiner_height",
            "refiner_width": "--refiner_width",
            "refiner_steps": "--refiner_steps",
            "refiner_guidance_scale": "--refiner_guidance_scale",
            "refiner_shift": "--refiner_shift",
            "refiner_t_thresh": "--refiner_t_thresh",
            "refiner_sigma_tail_steps": "--refiner_sigma_tail_steps",
            "refiner_fps": "--refiner_fps",
            "refiner_sample_fps": "--refiner_sample_fps",
            "refiner_max_video_frames": "--refiner_max_video_frames",
            "refiner_vae_dtype": "--refiner_vae_dtype",
        }
        normalized_request = dict(request)
        if normalized_request.get("steps") is None:
            normalized_request["steps"] = normalized_request.get("num_inference_steps")
        for key, flag in value_flags.items():
            _append_value(base_args, flag, normalized_request.get(key))
        _append_value(
            base_args,
            "--refiner_model_dir",
            _expand_path(request.get("refiner_model_dir")),
        )

        boolean_flags = {
            "quiet_progress": "--quiet_progress",
            "context_parallel_ulysses_anything": "--context_parallel_ulysses_anything",
            "enable_fsdp_inference": "--enable_fsdp_inference",
            "batch_cfg": "--batch_cfg",
            "null_cond_clone_zero": "--null_cond_clone_zero",
            "reuse_condition_features": "--reuse_condition_features",
            "run_refiner": "--run_refiner",
            "refiner_batch_cfg": "--refiner_batch_cfg",
            "refiner_no_null_cond_clone_zero": "--refiner_no_null_cond_clone_zero",
            "refiner_offload_vae_during_denoise": "--refiner_offload_vae_during_denoise",
        }
        for key, flag in boolean_flags.items():
            if bool(request.get(key, False)):
                base_args.append(flag)
        if request.get("allow_tf32") is False:
            base_args.append("--no-allow_tf32")

        refiner_output: Path | None = None
        if bool(request.get("run_refiner") or request.get("refiner_model_dir")):
            refiner_output = _expand_path(request.get("refiner_output"))
            if refiner_output is None:
                refiner_output = output.with_name(
                    f"{output.stem}_refined{output.suffix}"
                )
            _append_value(base_args, "--refiner_output", refiner_output)

        cfg_degree = int(request.get("cfg_parallel_degree") or 1)
        context_degree = int(request.get("context_parallel_degree") or 1)
        topology_size = cfg_degree * context_degree
        requested_nproc = int(request.get("nproc_per_node") or 1)
        nproc = (
            topology_size
            if topology_size > 1 and requested_nproc == 1
            else requested_nproc
        )
        if topology_size > 1 and nproc != topology_size:
            raise ValueError(
                "LingBot-Video nproc_per_node must equal cfg_parallel_degree * context_parallel_degree "
                f"({topology_size}), got {nproc}."
            )
        command = [str(self.python_executable)]
        if nproc > 1:
            command.extend(
                (
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nproc_per_node",
                    str(nproc),
                )
            )
        command.extend(base_args)

        env = {
            "PYTHONPATH": f"{SOURCE_ROOT}{os.pathsep + os.environ['PYTHONPATH'] if os.environ.get('PYTHONPATH') else ''}",
        }
        visible_devices = cuda_visible_devices_from_device(self.device)
        if visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = visible_devices

        return LingBotVideoRuntimePlan(
            command=tuple(command),
            env=env,
            workdir=str(SOURCE_ROOT),
            checkpoint_dir=str(self.checkpoint_dir),
            output_path=str(output),
            refiner_output_path=(
                str(refiner_output) if refiner_output is not None else None
            ),
            mode=mode,
        )

    def run_plan(
        self,
        plan: LingBotVideoRuntimePlan,
        *,
        timeout_seconds: int = 7200,
        log_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        target_log_dir = (
            Path(log_dir or Path(plan.output_path).parent).expanduser().resolve()
        )
        target_log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = target_log_dir / "lingbot_video_stdout.log"
        stderr_path = target_log_dir / "lingbot_video_stderr.log"
        started = time.monotonic()
        returncode = -1
        error: str | None = None
        env = os.environ.copy()
        env.update(plan.env)
        try:
            with (
                stdout_path.open("w", encoding="utf-8") as stdout,
                stderr_path.open("w", encoding="utf-8") as stderr,
            ):
                completed = subprocess.run(
                    list(plan.command),
                    cwd=plan.workdir,
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                    timeout=timeout_seconds,
                    check=False,
                )
            returncode = completed.returncode
        except Exception as exc:  # pragma: no cover - depends on the deployment runtime
            error = str(exc)

        candidates = [Path(plan.output_path)]
        if plan.refiner_output_path:
            candidates.append(Path(plan.refiner_output_path))
        generated_files = [
            str(path)
            for path in candidates
            if path.is_file() and path.stat().st_size > 0
        ]
        ok = returncode == 0 and bool(generated_files)
        if returncode != 0 and error is None:
            error = f"LingBot-Video runner exited with code {returncode}."
        elif returncode == 0 and not generated_files and error is None:
            error = "LingBot-Video runner completed without producing an artifact."
        return {
            "ok": ok,
            "status": "success" if ok else "failed",
            "returncode": returncode,
            "duration_seconds": round(time.monotonic() - started, 3),
            "generated_files": generated_files,
            "error": error,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "runtime_plan": plan.to_dict(),
        }


__all__ = ["LingBotVideoRuntime", "LingBotVideoRuntimePlan"]
