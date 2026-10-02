"""Inference-only Astronex camera/action/event adapter; no duplicated Wan model."""

from __future__ import annotations

import math
import re
from pathlib import Path

from worldfoundry.core.io import resolve_data_path
from worldfoundry.runtime.official_inference import (
    InferencePlan,
    OfficialInferenceRuntime,
    local_path,
    positive_int,
)


class AstronexWorldRuntime(OfficialInferenceRuntime):
    MODEL_ID = "astronex-world"
    REVISION = "27584bf1a89da01ef35a03b48d83bc818d2adf2b"
    REQUIRED_SOURCE_FILES = ("inference/generate.py", "inference/sample.py", "astronex_env.py")

    def __init__(
        self,
        *,
        source_root="${WORLDFOUNDRY_MODEL_SOURCE_DIR}/Astronex-Robotics--Astronex-World",
        checkpoint_dir="${WORLDFOUNDRY_HFD_ROOT}/Astronex-Lab--Astronex-World",
        mode="causal",
        consumer=False,
        python_executable=None,
        device="cuda",
    ):
        super().__init__(source_root=source_root, python_executable=python_executable, device=device)
        if mode not in {"causal", "bidirectional"}:
            raise ValueError("Astronex mode must be causal or bidirectional")
        if consumer and mode != "causal":
            raise ValueError("The consumer profile is only released for causal inference")
        if self.device.startswith("cuda:") and "," in self.device:
            raise ValueError("This Astronex entrypoint uses one CUDA device")
        self.checkpoint_dir = local_path(checkpoint_dir)
        self.mode = mode
        self.config_path = Path(
            resolve_data_path(f"models/runtime/configs/astronex-world/{'causal_consumer' if consumer else mode}.yaml")
        )

    def required_assets(self):
        return (
            self.config_path,
            *(self.checkpoint_dir / name for name in ("transformer/config.json", "vae", "text_encoder", "tokenizer")),
        )

    def preflight(self):
        result = super().preflight()
        if not any(self.checkpoint_dir.glob("model*.safetensors")):
            result["missing_paths"].append(str(self.checkpoint_dir / "model*.safetensors"))
            result["status"] = "blocked"
        return result

    def build_plan(self, *, request, output_dir):
        request = dict(request)
        allowed = {
            "prompt",
            "images",
            "trajectory",
            "latent_frames",
            "steps",
            "seed",
            "fps",
            "event_prompt",
            "event_start_frame",
            "action_tokens",
            "action_keys",
            "action_scale",
            "embodiment_id",
            "height",
            "width",
            "vram_limit_gb",
        }
        if request.keys() - allowed:
            raise ValueError(
                f"Unknown Astronex inference options: {sorted(request.keys() - allowed)}; lengths use latent_frames."
            )
        prompt = request.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Astronex requires a non-empty prompt")
        image = request.get("images")
        reference = image is not None
        frames = positive_int(
            request.get("latent_frames", (23 if reference else 24) if self.mode == "causal" else 17), "latent_frames"
        )
        if self.mode == "causal" and (frames + int(reference)) % 8:
            raise ValueError("Causal Astronex requires (latent_frames + reference_count) divisible by 8")
        trajectory = str(request.get("trajectory") or f"h*{frames}")
        # Upstream expands comma-separated camera segments on the latent timeline.
        segments = trajectory.split(",")
        if not all(
            re.fullmatch(r"(?:dn|[wasdujlikh])(?:\+(?:dn|[wasdujlikh]))*\*[1-9][0-9]*", s.strip()) for s in segments
        ):
            raise ValueError("trajectory must contain camera-key segments such as w*12,h*11")
        axes = {
            "w": "z",
            "s": "z",
            "a": "x",
            "d": "x",
            "u": "y",
            "dn": "y",
            "j": "yaw",
            "l": "yaw",
            "i": "pitch",
            "k": "pitch",
            "h": "hold",
        }
        for segment in segments:
            selected = [axes[key] for key in segment.strip().split("*")[0].split("+") if key != "h"]
            if len(selected) != len(set(selected)):
                raise ValueError("A compound trajectory segment cannot drive the same axis twice")
        if sum(int(s.split("*")[1]) for s in segments) != frames:
            raise ValueError("trajectory segment lengths must sum to latent_frames")
        event = request.get("event_start_frame")
        if event is not None:
            if self.mode != "causal" or int(event) != float(event) or not 0 <= int(event) < frames:
                raise ValueError("event_start_frame must be a causal latent-frame index within the rollout")
            if not request.get("event_prompt"):
                raise ValueError("event_start_frame requires event_prompt")
        if "embodiment_id" in request and not 0 <= int(request["embodiment_id"]) < 32:
            raise ValueError("embodiment_id must be in [0, 31]")
        for name in ("action_scale", "vram_limit_gb"):
            if name in request and (not math.isfinite(float(request[name])) or float(request[name]) <= 0):
                raise ValueError(f"{name} must be finite and positive")
        output = local_path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        prompt_path = output / "prompt.txt"
        prompt_path.write_text(" ".join(prompt.split()) + "\n", encoding="utf-8")
        command = [
            self.python_executable,
            str(self.source_root / "inference/generate.py"),
            "--mode",
            self.mode,
            "--prompt",
            str(prompt_path),
            "--weights",
            str(self.checkpoint_dir),
            "--config",
            str(self.config_path),
            "--out",
            str(output / "artifacts"),
            "--frames",
            str(frames),
            "--trajectory",
            trajectory,
            "--steps",
            str(positive_int(request.get("steps", 8 if self.mode == "causal" else 50), "steps")),
            "--fps",
            str(positive_int(request.get("fps", 24), "fps")),
            "--seed",
            str(int(request.get("seed", 0))),
        ]
        # generate.py overwrites CUDA_VISIBLE_DEVICES, so pass the selected visible index explicitly.
        import os

        visible = (
            self.device[5:]
            if self.device.startswith("cuda:")
            else os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        )
        command.extend(("--gpu", visible))
        if reference:
            from worldfoundry.core.io.image import (
                load_pil_image,
                materialize_image_input,
            )

            image_path = materialize_image_input(load_pil_image(image), str(output), filename="input.png")
            command.extend(("--image", str(image_path)))
        flags = {
            "event_prompt": "--event-prompt",
            "event_start_frame": "--event-start-frame",
            "vram_limit_gb": "--vram-limit-gb",
            "action_tokens": "--action_tokens",
            "action_scale": "--action_scale",
            "embodiment_id": "--embodiment_id",
        }
        for name, flag in flags.items():
            if request.get(name) is not None:
                command.extend((flag, str(request[name])))
        if request.get("action_keys"):
            command.append("--action_keys")
        for name in ("height", "width"):
            if name in request:
                value = positive_int(request[name], name)
                if value % 32:
                    raise ValueError(f"{name} must be a multiple of 32")
                command.extend((f"--latent_{name}", str(value // 16)))
        return InferencePlan(
            self.MODEL_ID,
            tuple(command),
            str(self.source_root),
            str(output),
            self.REVISION,
            self.environment(self.source_root),
            artifact_globs=("artifacts/**/*.mp4",),
        )
