from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from worldfoundry.core import as_list, cuda_visible_devices_from_device, jsonable, resolve_hf_snapshot_path


@dataclass(frozen=True)
class PusaVidGenRuntimePlan:
    """Executable Pusa VidGen command and artifact paths."""

    command: tuple[str, ...]
    env: Mapping[str, str]
    workdir: str
    checkpoint_root: str
    output_dir: str
    output_path: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize the runtime plan."""
        return {
            "command": list(self.command),
            "env": dict(self.env),
            "workdir": self.workdir,
            "checkpoint_root": self.checkpoint_root,
            "output_dir": self.output_dir,
            "output_path": self.output_path,
        }


class PusaVidGenRuntime:
    """Lightweight in-tree runtime shim for Pusa VidGen request preparation."""

    def __init__(
        self,
        model_id: str,
        checkpoint_root: str | Path,
        base_model_root: str | Path,
        high_lora_path: str | Path | None = None,
        low_lora_path: str | Path | None = None,
        lightx2v_root: str | Path | None = None,
        device: str = "cuda",
        python_executable: str | Path | None = None,
        runner_version: str | None = None,
    ) -> None:
        """
        Store Pusa V1 asset paths without importing model components.

        Args:
            model_id: WorldFoundry model identifier.
            checkpoint_root: Local Pusa checkpoint directory.
            base_model_root: Local Wan base model checkpoint directory.
            high_lora_path: Optional high-noise LoRA checkpoint.
            low_lora_path: Optional low-noise LoRA checkpoint.
            lightx2v_root: Optional LightX2V checkpoint directory.
            device: Runtime device label.
            python_executable: Optional Python executable with the Pusa V1 dependencies.
        """
        self.model_id = model_id
        self.checkpoint_root = resolve_hf_snapshot_path(
            checkpoint_root,
            required_files=("high_noise_pusa.safetensors", "low_noise_pusa.safetensors"),
        )
        self.base_model_root = resolve_hf_snapshot_path(
            base_model_root,
            required_files=("models_t5_umt5-xxl-enc-bf16.pth", "Wan2.1_VAE.pth"),
        )
        self.high_lora_path = (
            Path(high_lora_path).expanduser()
            if high_lora_path
            else self.checkpoint_root / "high_noise_pusa.safetensors"
        )
        self.low_lora_path = (
            Path(low_lora_path).expanduser() if low_lora_path else self.checkpoint_root / "low_noise_pusa.safetensors"
        )
        self.lightx2v_root = resolve_hf_snapshot_path(lightx2v_root) if lightx2v_root else None
        self.device = device
        self.python_executable = Path(python_executable).expanduser() if python_executable else Path(sys.executable)
        self.runner_version = (runner_version or "v1").strip().lower() or "v1"
        if self.runner_version != "v1":
            raise ValueError("Pusa v0 / Mochi route is not available in this in-tree build; use v1")

    def prepare(
        self,
        prompt: str | None,
        images: Any,
        video: Any,
        interactions: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Build a Pusa generation request without running heavyweight inference.

        Args:
            prompt: Text prompt for generation.
            images: Optional conditioning image path or paths.
            video: Optional conditioning video path.
            interactions: Optional conditioning frame positions.
            **kwargs: Model-specific generation settings.
        """
        mode = kwargs.pop("mode", None) or ("v2v" if video is not None else "i2v" if images is not None else "t2v")
        conditioning = {
            "images": as_list(images),
            "video": str(video) if video is not None else "",
            "cond_position": as_list(kwargs.pop("cond_position", interactions)),
            "noise_multipliers": as_list(kwargs.pop("noise_multipliers", kwargs.pop("noise_multiplier", None))),
        }
        lightx2v = bool(kwargs.pop("lightx2v", False))
        default_cfg_scale = 1.0 if lightx2v else 3.0
        if "cfg_scale" in kwargs:
            cfg_scale = float(kwargs.pop("cfg_scale"))
        elif "guidance_scale" in kwargs:
            cfg_scale = float(kwargs.pop("guidance_scale"))
        else:
            cfg_scale = default_cfg_scale
        request = {
            "model_id": self.model_id,
            "mode": mode,
            "task": f"pusa_{mode}_video_generation",
            "prompt": "" if prompt is None else str(prompt),
            "negative_prompt": kwargs.pop("negative_prompt", ""),
            "conditioning": conditioning,
            "sampling": {
                "num_inference_steps": int(kwargs.pop("num_inference_steps", kwargs.pop("num_steps", 30))),
                "cfg_scale": cfg_scale,
                "switch_dit_boundary": float(
                    kwargs.pop("switch_dit_boundary", kwargs.pop("switch_DiT_boundary", 0.875))
                ),
                "num_frames": int(kwargs.pop("num_frames", 81)),
                "height": int(kwargs.pop("height", 720)),
                "width": int(kwargs.pop("width", 1280)),
                "seed": int(kwargs.pop("seed", 0)),
                "lightx2v": lightx2v,
                "fps": int(kwargs.pop("fps", 24)),
                "quality": int(kwargs.pop("quality", 5)),
                "high_lora_alpha": float(kwargs.pop("high_lora_alpha", 1.5)),
                "low_lora_alpha": float(kwargs.pop("low_lora_alpha", 1.4)),
            },
            "runtime": {
                "implementation": "worldfoundry.synthesis.visual_generation.pusa_vidgen.adapter.PusaVidGenRuntime",
                "checkpoint_root": str(self.checkpoint_root),
                "base_model_root": str(self.base_model_root),
                "high_lora_path": str(self.high_lora_path),
                "low_lora_path": str(self.low_lora_path),
                "lightx2v_root": str(self.lightx2v_root) if self.lightx2v_root else "",
                "runner_version": self.runner_version,
                "device": self.device,
            },
            "extra_inputs": dict(kwargs),
        }
        return jsonable(request)

    def build_plan(
        self,
        *,
        request: Mapping[str, Any],
        output_dir: str | Path,
        output_path: str | Path | None = None,
    ) -> PusaVidGenRuntimePlan:
        """Build the in-tree Pusa V1 runner command.

        Args:
            request: Request payload returned by :meth:`prepare`.
            output_dir: Directory for generated videos and sidecar metadata.
            output_path: Optional generated-video path. When omitted, the
                runner writes ``pusa_vidgen_public_runner.mp4`` in output_dir.
        """
        sampling = dict(request.get("sampling") or {})
        target_dir = Path(output_dir).expanduser().resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        target_output_path = (
            Path(output_path).expanduser().resolve()
            if output_path is not None
            else target_dir / "pusa_vidgen_public_runner.mp4"
        )
        target_output_path.parent.mkdir(parents=True, exist_ok=True)
        runner_version = str((request.get("runtime") or {}).get("runner_version") or self.runner_version).lower()
        if runner_version != "v1":
            raise ValueError("Pusa v0 / Mochi route is not available in this in-tree build; use v1")
        missing_loras = [path for path in (self.high_lora_path, self.low_lora_path) if not path.is_file()]
        if missing_loras:
            raise FileNotFoundError(
                "Pusa v1 requires high- and low-noise LoRA checkpoints: "
                + ", ".join(str(path) for path in missing_loras)
            )
        mode = str(request.get("mode") or "t2v")
        conditioning = dict(request.get("conditioning") or {})
        images = [str(item) for item in conditioning.get("images") or [] if str(item)]
        command_list = [
            str(self.python_executable),
            "-m",
            "worldfoundry.synthesis.visual_generation.pusa_vidgen.pusa_v1_runner",
            "--mode",
            "multi" if images else mode,
            "--checkpoint-root",
            str(self.checkpoint_root),
            "--base-model-root",
            str(self.base_model_root),
            "--high-lora-path",
            str(self.high_lora_path),
            "--low-lora-path",
            str(self.low_lora_path),
            "--output-path",
            str(target_output_path),
            "--prompt",
            str(request.get("prompt") or ""),
            "--negative-prompt",
            str(request.get("negative_prompt") or ""),
            "--width",
            str(int(sampling.get("width", 1280))),
            "--height",
            str(int(sampling.get("height", 720))),
            "--num-frames",
            str(int(sampling.get("num_frames", 81))),
            "--num-inference-steps",
            str(int(sampling.get("num_inference_steps", 4))),
            "--cfg-scale",
            str(float(sampling.get("cfg_scale", 1.0 if bool(sampling.get("lightx2v", False)) else 3.0))),
            "--seed",
            str(int(sampling.get("seed", 0))),
            "--fps",
            str(int(sampling.get("fps", 24))),
            "--quality",
            str(int(sampling.get("quality", 5))),
            "--high-lora-alpha",
            str(float(sampling.get("high_lora_alpha", 1.5))),
            "--low-lora-alpha",
            str(float(sampling.get("low_lora_alpha", 1.4))),
            "--switch-dit-boundary",
            str(float(sampling.get("switch_dit_boundary", 0.875))),
        ]
        if self.lightx2v_root:
            command_list.extend(["--lightx2v-root", str(self.lightx2v_root)])
        if bool(sampling.get("lightx2v", False)):
            command_list.append("--lightx2v")
        for image in images:
            command_list.extend(["--image-path", image])
        cond_position = ",".join(str(item) for item in conditioning.get("cond_position") or [])
        noise_multipliers = ",".join(str(item) for item in conditioning.get("noise_multipliers") or [])
        if cond_position:
            command_list.extend(["--cond-position", cond_position])
        if noise_multipliers:
            command_list.extend(["--noise-multipliers", noise_multipliers])
        command = tuple(command_list)
        env = {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
        visible_devices = cuda_visible_devices_from_device(self.device)
        if visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = visible_devices
        return PusaVidGenRuntimePlan(
            command=command,
            env=env,
            workdir=str(target_dir),
            checkpoint_root=str(self.checkpoint_root),
            output_dir=str(target_dir),
            output_path=str(target_output_path),
        )

    def run_plan(
        self,
        plan: PusaVidGenRuntimePlan,
        *,
        timeout_seconds: int = 1800,
        log_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Execute a Pusa VidGen runtime plan and collect generated videos."""
        started = time.monotonic()
        target_log_dir = Path(log_dir or plan.output_dir).expanduser().resolve()
        target_log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = target_log_dir / "pusa_vidgen_stdout.log"
        stderr_path = target_log_dir / "pusa_vidgen_stderr.log"
        env = os.environ.copy()
        env.update(plan.env)
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
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
        generated_files = sorted(str(path) for path in Path(plan.output_dir).expanduser().resolve().rglob("*.mp4"))
        metadata_path = str(Path(plan.output_path).with_suffix(".json"))
        ok = completed.returncode == 0 and bool(generated_files)
        return {
            "ok": ok,
            "status": "success" if ok else "failed",
            "returncode": completed.returncode,
            "duration_seconds": round(time.monotonic() - started, 3),
            "generated_count": len(generated_files),
            "generated_files": generated_files,
            "metadata_path": metadata_path if Path(metadata_path).is_file() else "",
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "runtime_plan": plan.to_dict(),
        }
