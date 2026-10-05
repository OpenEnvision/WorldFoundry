"""Independent in-tree runtimes for minWM HY and Wan Action2V."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.core.io.paths import checkpoint_root_path, hfd_root_path
from worldfoundry.core.distributed.runtime.multiprocess_launch import find_free_port
from worldfoundry.runtime.in_tree_cli import ensure_in_tree_runtime, execute_in_tree, require_path


_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_MINWM_CONFIG_ROOT = _PROJECT_ROOT / "worldfoundry" / "data" / "models" / "runtime" / "configs" / "minwm"


def _rotation_step(token_mapping_details: Any, default: float = 3.0) -> float:
    if isinstance(token_mapping_details, Mapping):
        value = token_mapping_details.get("runtime_yaw_deg_per_token")
        if value is not None:
            return float(value)
    return default


class _MinWMRuntime:
    SOURCE_REVISION = "df522a26cd4409d3e3e8f269cc98eac069b5df47"

    def __init__(
        self,
        *,
        checkpoint_path: Any = None,
        base_model_path: Any = None,
        python_executable: Any = None,
        device: str = "cuda",
        env: Mapping[str, Any] | None = None,
    ) -> None:
        self.repo_root = ensure_in_tree_runtime(self.bundled_repo_root(), package_file=__file__)
        hfd = hfd_root_path()
        checkpoints = checkpoint_root_path()
        self.checkpoint_path = Path(checkpoint_path or hfd / "MIN-Lab--minWM").expanduser()
        self.base_model_path = Path(base_model_path or self.default_base_path(checkpoints)).expanduser()
        self.python_executable = str(python_executable or sys.executable)
        self.device = device
        self.env = dict(env or {})

    @staticmethod
    def bundled_repo_root() -> Path:
        return Path(__file__).resolve().parent / "minwm_runtime"

    @classmethod
    def from_pretrained(cls, pretrained_model_path: Any = None, **kwargs: Any):
        return cls(checkpoint_path=pretrained_model_path, **kwargs)

    def plan(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "model_id": self.MODEL_ID,
            "repo_root": str(self.repo_root),
            "source_revision": self.SOURCE_REVISION,
            "checkpoint_path": str(self.checkpoint_path),
            "base_model_path": str(self.base_model_path),
            "inputs": {key: str(value) for key, value in kwargs.items()},
        }


class MinWMHYAction2VRuntime(_MinWMRuntime):
    MODEL_ID = "minwm-hy-action2v"

    @staticmethod
    def bundled_repo_root() -> Path:
        return Path(__file__).resolve().parent

    @staticmethod
    def default_base_path(checkpoints: Path) -> Path:
        return checkpoints / "HunyuanVideo-1.5"

    def predict(
        self,
        *,
        prompt: str,
        image_path: Any,
        trajectory: str,
        output_path: Any,
        token_mapping_details: Any = None,
        fps: int = 16,
        width: int = 832,
        height: int = 480,
        num_frames: int = 77,
        num_inference_steps: int = 4,
        seed: int = 0,
        return_dict: bool = True,
        **_: Any,
    ) -> Any:
        image = require_path(image_path, "minWM HY first-frame image", kind="file")
        checkpoints = require_path(self.checkpoint_path, "minWM checkpoint root", kind="dir")
        transformer = require_path(checkpoints / "HY15" / "Action2V" / "dmd", "minWM HY DMD checkpoint", kind="dir")
        require_path(transformer / "config.json", "minWM HY DMD config", kind="file")
        require_path(
            transformer / "diffusion_pytorch_model.safetensors",
            "minWM HY DMD weights",
            kind="file",
        )
        base = require_path(self.base_model_path, "HunyuanVideo-1.5 base model", kind="dir")
        for relative in (
            "config.json",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
        ):
            require_path(base / relative, f"HunyuanVideo-1.5 asset {relative}", kind="file")
        output = Path(output_path).expanduser().resolve()
        input_dir = output.parent / f".{output.stem}_minwm_hy_inputs"
        native_output = input_dir / "outputs"
        input_dir.mkdir(parents=True, exist_ok=True)
        example = input_dir / "example.json"
        example.write_text(
            json.dumps([{"id": 0, "image": str(image), "caption": prompt, "trajectory": trajectory}], indent=2),
            encoding="utf-8",
        )
        env = {
            **self.env,
            # The module is launched from the WorldFoundry root; do not let an
            # ambient checkout shadow the in-tree implementation.
            "PYTHONPATH": "",
            "WORLDFOUNDRY_MINWM_ROT_STEP_DEG": str(_rotation_step(token_mapping_details)),
        }
        command = [
            self.python_executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=1",
            "--module",
            "worldfoundry.synthesis.visual_generation.minwm.hy15_inference",
            "--mode",
            "ar_rollout",
            "--transformer_dir",
            transformer,
            "--model_path",
            base,
            "--example_json",
            example,
            "--output_dir",
            native_output,
            "--trajectory",
            trajectory,
            "--use_camera",
            "--num_inference_steps",
            str(num_inference_steps),
            "--shift",
            "5.0",
            "--guidance_scale",
            "1.0",
            "--fps",
            str(fps),
            "--height",
            str(height),
            "--width",
            str(width),
            "--video_length",
            str(num_frames),
            "--chunk_latent_frames",
            "4",
            "--stabilization_level",
            "1",
            "--seed",
            str(seed),
        ]
        result = execute_in_tree(
            command,
            cwd=_PROJECT_ROOT,
            output_path=output,
            search_roots=(native_output,),
            env=env,
            python_paths=(),
        )
        return result if return_dict else result.get("video")


class MinWMWanAction2VRuntime(_MinWMRuntime):
    MODEL_ID = "minwm-wan-action2v"

    @staticmethod
    def default_base_path(checkpoints: Path) -> Path:
        return checkpoints / "Wan2.1-T2V-1.3B"

    def predict(
        self,
        *,
        prompt: str,
        trajectory: str,
        output_path: Any,
        token_mapping_details: Any = None,
        num_output_frames: int = 20,
        seed: int = 0,
        sp_size: int = 1,
        master_port: int | None = None,
        return_dict: bool = True,
        **_: Any,
    ) -> Any:
        sp_size = int(sp_size)
        if sp_size < 1:
            raise ValueError(f"minWM Wan sp_size must be >= 1, got {sp_size}")
        checkpoints = require_path(self.checkpoint_path, "minWM checkpoint root", kind="dir")
        checkpoint = require_path(
            checkpoints / "Wan21" / "Action2V" / "dmd" / "model.pt",
            "minWM Wan DMD checkpoint",
            kind="file",
        )
        base = require_path(self.base_model_path, "Wan2.1-T2V-1.3B base model", kind="dir")
        for relative in (
            "config.json",
            "diffusion_pytorch_model.safetensors",
            "Wan2.1_VAE.pth",
            "models_t5_umt5-xxl-enc-bf16.pth",
        ):
            require_path(base / relative, f"Wan2.1-T2V-1.3B asset {relative}", kind="file")
        output = Path(output_path).expanduser().resolve()
        input_dir = output.parent / f".{output.stem}_minwm_wan_inputs"
        native_output = input_dir / "outputs"
        input_dir.mkdir(parents=True, exist_ok=True)
        prompt_txt = input_dir / "prompt.txt"
        trajectory_txt = input_dir / "trajectory.txt"
        prompt_txt.write_text(prompt.strip() + "\n", encoding="utf-8")
        trajectory_txt.write_text(trajectory.strip() + "\n", encoding="utf-8")
        env = {
            **self.env,
            # The module is launched from the WorldFoundry root; do not let an
            # ambient checkout shadow the in-tree implementation.
            "PYTHONPATH": "",
            "SP_SIZE": str(sp_size),
            "WORLDFOUNDRY_MINWM_ROT_STEP_DEG": str(_rotation_step(token_mapping_details)),
            "WORLDFOUNDRY_MINWM_WAN_BASE_MODEL": str(base),
        }
        command = [
            self.python_executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node",
            str(sp_size),
            "--master_port",
            str(int(master_port) if master_port is not None else find_free_port()),
            "--module",
            "worldfoundry.synthesis.visual_generation.minwm.minwm_runtime.Wan21.wan_inference",
            "--config_path",
            _MINWM_CONFIG_ROOT / "wan_action2v.yaml",
            "--checkpoint_path",
            checkpoint,
            "--data_path",
            prompt_txt,
            "--output_folder",
            native_output,
            "--sp_size",
            str(sp_size),
            "--trajectory_path",
            trajectory_txt,
            "--num_output_frames",
            str(num_output_frames),
            "--seed",
            str(seed),
        ]
        result = execute_in_tree(
            command,
            cwd=_PROJECT_ROOT,
            output_path=output,
            search_roots=(native_output,),
            env=env,
            python_paths=(),
        )
        return result if return_dict else result.get("video")


__all__ = ["MinWMHYAction2VRuntime", "MinWMWanAction2VRuntime"]
