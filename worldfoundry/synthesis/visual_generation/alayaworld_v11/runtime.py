"""Inference-only launch contract for the AlayaWorld v1.1 AR and DMD releases."""

from __future__ import annotations

import json
from pathlib import Path

from worldfoundry.core.io import resolve_data_path
from worldfoundry.runtime.official_inference import (
    InferencePlan,
    OfficialInferenceRuntime,
    local_path,
    positive_int,
)


class AlayaWorldV11Runtime(OfficialInferenceRuntime):
    MODEL_ID = "alayaworld-v1.1"
    REVISION = "ea03cfbb2e4c4e9102ed8ea8562e0b5370ca9b79"
    REQUIRED_SOURCE_FILES = (
        "scripts/infer/prepare_i2v_inputs.py",
        "alaya/memory/vigeo_geometry.py",
        "ltx2/modules/model_ltx_2_3.py",
    )

    def __init__(
        self,
        *,
        source_root="${WORLDFOUNDRY_MODEL_SOURCE_DIR}/AlayaLab--AlayaWorld",
        checkpoint_dir="${WORLDFOUNDRY_HFD_ROOT}/AlayaLab--AlayaWorld-v1.1-stage2b",
        student_dir="${WORLDFOUNDRY_HFD_ROOT}/AlayaLab--AlayaWorld-v1.1-stage3",
        base_checkpoint=None,
        gemma_path=None,
        vigeo_checkpoint=None,
        vigeo_source_root=None,
        variant="dmd",
        python_executable=None,
        device="cuda",
    ):
        super().__init__(source_root=source_root, python_executable=python_executable, device=device)
        if variant not in {"ar", "dmd"}:
            raise ValueError("AlayaWorld v1.1 variant must be ar or dmd")
        if self.device.startswith("cuda:") and "," in self.device:
            raise ValueError("AlayaWorld v1.1 inference currently supports one CUDA device")
        self.variant = variant
        self.checkpoint_dir = local_path(checkpoint_dir)
        self.student_dir = local_path(student_dir)
        self.base_checkpoint = local_path(base_checkpoint) if base_checkpoint else None
        self.gemma_path = local_path(gemma_path) if gemma_path else None
        self.vigeo_checkpoint = local_path(vigeo_checkpoint) if vigeo_checkpoint else None
        self.vigeo_source_root = (
            local_path(vigeo_source_root) if vigeo_source_root else self.source_root / "third_party/ViGeo"
        )

    def required_assets(self):
        paths = [
            self.checkpoint_dir / "transformer.pt",
            self.checkpoint_dir / "history_encoder.pt",
            self.vigeo_source_root / "vigeo/__init__.py",
        ]
        if self.variant == "dmd":
            paths.append(self.student_dir / "lora.safetensors")
        paths.extend(p for p in (self.base_checkpoint, self.gemma_path, self.vigeo_checkpoint) if p)
        return tuple(paths)

    def preflight(self):
        result = super().preflight()
        result["missing_options"] = [
            key for key in ("base_checkpoint", "gemma_path", "vigeo_checkpoint") if getattr(self, key) is None
        ]
        if result["missing_options"]:
            result["status"] = "blocked"
        return result

    def build_plan(self, *, request, output_dir):
        allowed = {"prompt", "images", "camera_path", "rounds", "seed", "intrinsic", "forward", "yaw", "pitch"}
        if request.keys() - allowed:
            raise ValueError(f"Unknown AlayaWorld v1.1 inference options: {sorted(request.keys() - allowed)}")
        if not request.get("prompt") or request.get("images") is None:
            raise ValueError("AlayaWorld v1.1 requires prompt and images")
        if any(p is None for p in (self.base_checkpoint, self.gemma_path, self.vigeo_checkpoint)):
            raise ValueError("base_checkpoint, gemma_path and vigeo_checkpoint are required")
        rounds = positive_int(request.get("rounds", 5), "rounds")
        output = local_path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        from worldfoundry.core.io.image import load_pil_image, materialize_image_input

        image = materialize_image_input(load_pil_image(request["images"]), str(output), filename="input.png")
        payload = {
            "variant": self.variant,
            "source_root": str(self.source_root),
            "image": str(image),
            "prompt": str(request["prompt"]),
            "rounds": rounds,
            "seed": int(request.get("seed", 42)),
            "output_dir": str(output),
            "checkpoint_dir": str(self.checkpoint_dir),
            "student_dir": str(self.student_dir),
            "base_checkpoint": str(self.base_checkpoint),
            "gemma_path": str(self.gemma_path),
            "vigeo_checkpoint": str(self.vigeo_checkpoint),
            "vigeo_source_root": str(self.vigeo_source_root),
            "config": str(resolve_data_path(f"models/runtime/configs/alayaworld-v1.1/{self.variant}.yaml")),
        }
        if request.get("camera_path") is not None:
            camera = local_path(request["camera_path"])
            if not camera.is_file():
                raise FileNotFoundError(camera)
            if camera.suffix not in {".npy", ".npz", ".txt"}:
                raise ValueError("v1.1 camera_path must be a C2W .npy, .npz or .txt file")
            payload["camera_path"] = str(camera)
        for key in ("intrinsic", "forward", "yaw", "pitch"):
            if key in request:
                payload[key] = request[key]
        request_path = output / "request.json"
        request_path.write_text(json.dumps(payload, indent=2) + "\n")
        entrypoint = Path(__file__).with_name("inference_entry.py")
        env = self.environment(self.source_root, Path(__file__).resolve().parents[4], self.vigeo_source_root)
        env["ALAYA_USE_FA3"] = "0"
        return InferencePlan(
            self.MODEL_ID,
            (self.python_executable, str(entrypoint), str(request_path)),
            str(self.source_root),
            str(output),
            self.REVISION,
            env,
            artifact_globs=("artifacts/*.mp4",),
            completion_globs=("INFERENCE_COMPLETE.json",),
        )
