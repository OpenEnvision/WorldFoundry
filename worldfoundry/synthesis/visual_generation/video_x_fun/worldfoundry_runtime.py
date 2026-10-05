"""Independent in-tree runtimes for VideoX-Fun camera-control checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.core.io.paths import resolve_data_path, resolve_local_hf_model_path
from worldfoundry.runtime.in_tree_cli import ensure_in_tree_runtime, execute_in_tree, require_path


WAN21_CAMERA_COMPONENT_FILES = (
    "config.json",
    "diffusion_pytorch_model.safetensors",
    "Wan2.1_VAE.pth",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
    "google/umt5-xxl/spiece.model",
    "google/umt5-xxl/tokenizer.json",
    "google/umt5-xxl/tokenizer_config.json",
)
WAN22_5B_CAMERA_COMPONENT_FILES = (
    "config.json",
    "diffusion_pytorch_model.safetensors",
    "Wan2.2_VAE.pth",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "google/umt5-xxl/spiece.model",
    "google/umt5-xxl/tokenizer.json",
    "google/umt5-xxl/tokenizer_config.json",
)
WAN22_A14B_CAMERA_COMPONENT_FILES = (
    "configuration.json",
    "low_noise_model/config.json",
    "low_noise_model/diffusion_pytorch_model.safetensors",
    "high_noise_model/config.json",
    "high_noise_model/diffusion_pytorch_model.safetensors",
    "Wan2.1_VAE.pth",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "google/umt5-xxl/spiece.model",
    "google/umt5-xxl/tokenizer.json",
    "google/umt5-xxl/tokenizer_config.json",
)


def _replace_assignment(source: str, name: str, value: Any) -> str:
    pattern = re.compile(rf"(?m)^{re.escape(name)}\s*=.*$")
    replacement = f"{name:<24}= {value!r}"
    rendered, count = pattern.subn(lambda _match: replacement, source, count=1)
    if count != 1:
        raise RuntimeError(f"VideoX-Fun entrypoint does not expose top-level assignment {name!r}")
    return rendered


def _replace_assignment_expression(source: str, name: str, expression: str) -> str:
    """Replace a generated-script assignment without quoting Python code."""

    pattern = re.compile(rf"(?m)^{re.escape(name)}\s*=.*$")
    replacement = f"{name:<24}= {expression}"
    rendered, count = pattern.subn(lambda _match: replacement, source, count=1)
    if count != 1:
        raise RuntimeError(f"VideoX-Fun entrypoint does not expose top-level assignment {name!r}")
    return rendered


class _WanFunCameraRuntime:
    SOURCE_REVISION = "403f1f7b78dcafccc4f6606dda4ec28e16e667dc"
    MODEL_ID = ""
    CHECKPOINT_REPO = ""
    ENTRYPOINT = ""
    CONFIG = ""
    BASE_COMPONENT_REPO = ""
    IMAGE_ENCODER_COMPONENT_REPO = ""
    TRANSFORMER_CONFIG_OVERRIDES: Mapping[str, Any] = {}
    TRANSFORMER_SCHEMA_SHA256 = ""
    TRANSFORMER_TENSOR_COUNT = 0
    REQUIRED_COMPONENT_FILES = WAN21_CAMERA_COMPONENT_FILES

    def __init__(
        self,
        *,
        checkpoint_path: Any = None,
        python_executable: Any = None,
        device: str = "cuda",
        env: Mapping[str, Any] | None = None,
        gpu_memory_mode: str = "sequential_cpu_offload",
    ) -> None:
        self.repo_root = ensure_in_tree_runtime(self.bundled_repo_root(), package_file=__file__)
        hfd = Path(os.environ.get("WORLDFOUNDRY_HFD_ROOT", "cache/hfd"))
        self.checkpoint_path = Path(checkpoint_path or hfd / self.CHECKPOINT_REPO).expanduser()
        self.python_executable = str(python_executable or sys.executable)
        self.device = device
        self.env = dict(env or {})
        self.gpu_memory_mode = str(gpu_memory_mode)

    @staticmethod
    def bundled_repo_root() -> Path:
        return Path(__file__).resolve().parent / "video_x_fun_runtime"

    @classmethod
    def from_pretrained(cls, pretrained_model_path: Any = None, **kwargs: Any):
        return cls(checkpoint_path=pretrained_model_path, **kwargs)

    def plan(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "model_id": self.MODEL_ID,
            "repo_root": str(self.repo_root),
            "source_revision": self.SOURCE_REVISION,
            "checkpoint_path": str(self.checkpoint_path),
            "entrypoint": self.ENTRYPOINT,
            "inputs": {key: str(value) for key, value in kwargs.items()},
        }

    def _configured_entrypoint(
        self,
        destination: Path,
        *,
        checkpoint: Path,
        image: Path,
        pose: Path,
        prompt: str,
        native_output: Path,
        width: int,
        height: int,
        num_frames: int,
        fps: int,
        num_inference_steps: int,
        seed: int,
        enable_teacache: bool,
    ) -> Path:
        upstream = require_path(self.repo_root / self.ENTRYPOINT, f"{self.MODEL_ID} entrypoint", kind="file")
        source = upstream.read_text(encoding="utf-8")
        values = {
            "GPU_memory_mode": self.gpu_memory_mode,
            "config_path": str(
                resolve_data_path("models", "runtime", "configs", "video_x_fun", self.CONFIG)
            ),
            "model_name": str(checkpoint),
            "sample_size": [int(height), int(width)],
            "video_length": int(num_frames),
            "fps": int(fps),
            "control_video": None,
            "control_camera_txt": str(pose),
            # Camera-only checkpoints condition on the first frame through the
            # normal I2V latent path.  ``ref_image`` is reserved for models
            # trained with a separate ref-conv branch and is not present in
            # the released camera checkpoint.
            "start_image": str(image),
            "ref_image": None,
            "prompt": prompt.strip(),
            "seed": int(seed),
            "num_inference_steps": int(num_inference_steps),
            "enable_teacache": bool(enable_teacache),
            "save_path": str(native_output),
        }
        for name, value in values.items():
            source = _replace_assignment(source, name, value)
        # BF16 is the preferred inference format on Ampere, Hopper and newer
        # architectures.  Turing/Volta GPUs do not implement native BF16, so
        # generating the selection in the child process keeps those common
        # cards on the supported FP16 path without weakening H100 throughput.
        source = _replace_assignment_expression(
            source,
            "weight_dtype",
            "torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16",
        )
        destination.write_text(source, encoding="utf-8")
        return destination

    @staticmethod
    def _link_checkpoint_file(destination: Path, source: Path) -> None:
        if not source.is_file():
            raise FileNotFoundError(f"VideoX-Fun component is missing: {source}")
        if Path(f"{source}.aria2").exists():
            raise FileNotFoundError(f"VideoX-Fun component transfer is incomplete: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(source.resolve())

    def _validate_transformer_schema(self, checkpoint_file: Path) -> None:
        if not self.TRANSFORMER_SCHEMA_SHA256:
            return
        from safetensors import safe_open

        with safe_open(str(checkpoint_file), framework="pt", device="cpu") as reader:
            rows = [
                f"{key}|{tuple(reader.get_slice(key).get_shape())}|"
                f"{reader.get_slice(key).get_dtype()}"
                for key in reader.keys()
            ]
        if len(rows) != self.TRANSFORMER_TENSOR_COUNT:
            raise RuntimeError(
                f"{self.MODEL_ID} transformer schema has {len(rows)} tensors; "
                f"expected {self.TRANSFORMER_TENSOR_COUNT}"
            )
        digest = hashlib.sha256("\n".join(sorted(rows)).encode("utf-8")).hexdigest()
        if digest != self.TRANSFORMER_SCHEMA_SHA256:
            raise RuntimeError(
                f"{self.MODEL_ID} transformer schema mismatch: "
                f"expected {self.TRANSFORMER_SCHEMA_SHA256}, got {digest}"
            )

    def _materialize_checkpoint_view(self, destination: Path, checkpoint: Path) -> Path:
        """Compose split official assets without copying or mutating checkpoints."""

        required = self.REQUIRED_COMPONENT_FILES
        if all((checkpoint / relative).is_file() for relative in required):
            return checkpoint
        if not self.BASE_COMPONENT_REPO or not self.IMAGE_ENCODER_COMPONENT_REPO:
            missing = [relative for relative in required if not (checkpoint / relative).is_file()]
            raise FileNotFoundError(
                f"{self.MODEL_ID} checkpoint is incomplete and has no split-component layout: {missing}"
            )

        base_files = (
            "config.json",
            "Wan2.1_VAE.pth",
            "models_t5_umt5-xxl-enc-bf16.pth",
            "google/umt5-xxl/spiece.model",
            "google/umt5-xxl/tokenizer.json",
            "google/umt5-xxl/tokenizer_config.json",
        )
        base_root = resolve_local_hf_model_path(
            self.BASE_COMPONENT_REPO,
            required_files=base_files,
        )
        image_root = resolve_local_hf_model_path(
            self.IMAGE_ENCODER_COMPONENT_REPO,
            required_files=("models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",),
        )
        destination.mkdir(parents=True, exist_ok=True)
        transformer_checkpoint = checkpoint / "diffusion_pytorch_model.safetensors"
        self._validate_transformer_schema(transformer_checkpoint)
        self._link_checkpoint_file(
            destination / "diffusion_pytorch_model.safetensors",
            transformer_checkpoint,
        )
        for relative in base_files:
            if relative == "config.json" and self.TRANSFORMER_CONFIG_OVERRIDES:
                config = json.loads((base_root / relative).read_text(encoding="utf-8"))
                config.update(dict(self.TRANSFORMER_CONFIG_OVERRIDES))
                (destination / relative).write_text(
                    json.dumps(config, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            else:
                self._link_checkpoint_file(destination / relative, base_root / relative)
        self._link_checkpoint_file(
            destination / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
            image_root / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        )
        return destination

    def predict(
        self,
        *,
        prompt: str,
        image_path: Any,
        pose_txt: Any,
        output_path: Any,
        width: int = 832,
        height: int = 480,
        num_frames: int = 81,
        fps: int = 16,
        num_inference_steps: int = 50,
        seed: int = 43,
        enable_teacache: bool = True,
        return_dict: bool = True,
        **_: Any,
    ) -> Any:
        image = require_path(image_path, f"{self.MODEL_ID} reference image", kind="file")
        pose = require_path(pose_txt, f"{self.MODEL_ID} CameraCtrl pose file", kind="file")
        checkpoint = require_path(self.checkpoint_path, f"{self.MODEL_ID} checkpoint", kind="dir")
        output = Path(output_path).expanduser().resolve()
        input_dir = output.parent / f".{output.stem}_{self.MODEL_ID}_inputs"
        native_output = input_dir / "outputs"
        input_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = self._materialize_checkpoint_view(input_dir / "checkpoint", checkpoint)
        configured = self._configured_entrypoint(
            input_dir / "predict.py",
            checkpoint=checkpoint,
            image=image,
            pose=pose,
            prompt=prompt,
            native_output=native_output,
            width=width,
            height=height,
            num_frames=num_frames,
            fps=fps,
            num_inference_steps=num_inference_steps,
            seed=seed,
            enable_teacache=enable_teacache,
        )
        env = dict(self.env)
        if self.device.startswith("cuda:") and "CUDA_VISIBLE_DEVICES" not in env:
            env["CUDA_VISIBLE_DEVICES"] = self.device.split(":", 1)[1]
        result = execute_in_tree(
            [self.python_executable, configured],
            cwd=self.repo_root,
            output_path=output,
            search_roots=(native_output,),
            env=env,
            python_paths=(self.repo_root,),
        )
        return result if return_dict else result.get("video")


class Wan21Fun1P3BCameraRuntime(_WanFunCameraRuntime):
    MODEL_ID = "wan21-fun-1p3b-cam"
    CHECKPOINT_REPO = "alibaba-pai--Wan2.1-Fun-V1.1-1.3B-Control-Camera"
    ENTRYPOINT = "examples/wan2.1_fun/predict_v2v_control_ref.py"
    CONFIG = "wan2.1/wan_civitai.yaml"
    BASE_COMPONENT_REPO = "Wan-AI/Wan2.1-T2V-1.3B"
    IMAGE_ENCODER_COMPONENT_REPO = "Wan-AI/Wan2.1-I2V-14B-480P"
    TRANSFORMER_CONFIG_OVERRIDES = {
        "model_type": "i2v",
        "in_dim": 32,
        "add_control_adapter": True,
        "in_dim_control_adapter": 24,
        "downscale_factor_control_adapter": 8,
        "add_ref_conv": False,
    }
    TRANSFORMER_TENSOR_COUNT = 989
    TRANSFORMER_SCHEMA_SHA256 = "5a036572c54433d0831b6987b84ce0d4e95dcabf69e939c4e95b5cc0d7964e49"


class Wan21Fun14BCameraRuntime(_WanFunCameraRuntime):
    MODEL_ID = "wan21-fun-14b-cam"
    CHECKPOINT_REPO = "alibaba-pai--Wan2.1-Fun-V1.1-14B-Control-Camera"
    ENTRYPOINT = "examples/wan2.1_fun/predict_v2v_control_ref.py"
    CONFIG = "wan2.1/wan_civitai.yaml"


class Wan22Fun5BCameraRuntime(_WanFunCameraRuntime):
    MODEL_ID = "wan22-fun-5b-cam"
    CHECKPOINT_REPO = "alibaba-pai--Wan2.2-Fun-5B-Control-Camera"
    ENTRYPOINT = "examples/wan2.2_fun/predict_v2v_control_ref_5b.py"
    CONFIG = "wan2.2/wan_civitai_5b.yaml"
    REQUIRED_COMPONENT_FILES = WAN22_5B_CAMERA_COMPONENT_FILES


class Wan22FunA14BCameraRuntime(_WanFunCameraRuntime):
    MODEL_ID = "wan22-fun-a14b-cam"
    CHECKPOINT_REPO = "alibaba-pai--Wan2.2-Fun-A14B-Control-Camera"
    ENTRYPOINT = "examples/wan2.2_fun/predict_v2v_control_ref.py"
    CONFIG = "wan2.2/wan_civitai_i2v.yaml"
    REQUIRED_COMPONENT_FILES = WAN22_A14B_CAMERA_COMPONENT_FILES


__all__ = [
    "Wan21Fun1P3BCameraRuntime",
    "Wan21Fun14BCameraRuntime",
    "Wan22Fun5BCameraRuntime",
    "Wan22FunA14BCameraRuntime",
]
