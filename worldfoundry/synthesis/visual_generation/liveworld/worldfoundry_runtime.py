"""Independent in-tree runtime for LiveWorld."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

import yaml

from worldfoundry.core.io.paths import checkpoint_root_candidates, hfd_root_path
from worldfoundry.evaluation.utils import worldfoundry_data_path
from worldfoundry.runtime.in_tree_cli import ensure_in_tree_runtime, execute_in_tree, require_path

_CONFIG_ROOT = worldfoundry_data_path("models", "runtime", "configs", "liveworld")


def _first_local_path(*candidates: Path) -> Path:
    """Prefer an existing local export while preserving the configured default."""
    return next((path for path in candidates if path.exists()), candidates[0])


def _first_frame(video_path: Path, output: Path) -> Path:
    import imageio.v3 as iio

    output.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(output, iio.imread(video_path, index=0))
    return output


class LiveWorldRuntime:
    """Run the bundled LiveWorld monitor-centric inference entrypoint."""

    SOURCE_REVISION = "a31145ffffb61be92f93c1e0d2ac5d826f29a256"

    def __init__(
        self,
        *,
        checkpoint_path: Any = None,
        base_model_path: Any = None,
        qwen_model_path: Any = None,
        sam3_model_path: Any = None,
        stream3r_model_path: Any = None,
        dinov3_model_path: Any = None,
        python_executable: Any = None,
        device: str = "cuda",
        env: Mapping[str, Any] | None = None,
    ) -> None:
        self.repo_root = ensure_in_tree_runtime(self.bundled_repo_root(), package_file=__file__)
        hfd = hfd_root_path()
        self.checkpoint_path = Path(
            checkpoint_path
            or _first_local_path(
                hfd / "ZichengD--LiveWorld",
                *checkpoint_root_candidates("ZichengD--LiveWorld"),
            )
        ).expanduser()
        self.base_model_path = Path(
            base_model_path
            or _first_local_path(
                *checkpoint_root_candidates("Wan2.1-T2V-14B"),
                *checkpoint_root_candidates("Wan-AI--Wan2.1-T2V-14B"),
            )
        ).expanduser()
        self.qwen_model_path = Path(
            qwen_model_path
            or _first_local_path(
                *checkpoint_root_candidates("modelscope/Qwen--Qwen3-VL-8B-Instruct"),
                *checkpoint_root_candidates("Qwen--Qwen3-VL-8B-Instruct"),
            )
        ).expanduser()
        self.sam3_model_path = Path(
            sam3_model_path
            or _first_local_path(
                *checkpoint_root_candidates("sam3"),
                *checkpoint_root_candidates("facebook--sam3"),
            )
        ).expanduser()
        self.stream3r_model_path = Path(
            stream3r_model_path
            or _first_local_path(
                hfd / "yslan--STream3R",
                *checkpoint_root_candidates("yslan--STream3R"),
            )
        ).expanduser()
        self.dinov3_model_path = Path(
            dinov3_model_path
            or _first_local_path(
                *checkpoint_root_candidates("modelscope/facebook--dinov3-vith16plus-pretrain-lvd1689m"),
                *checkpoint_root_candidates("facebook--dinov3-vith16plus-pretrain-lvd1689m"),
            )
        ).expanduser()
        self.python_executable = str(python_executable or sys.executable)
        self.device = device
        self.env = dict(env or {})

    @staticmethod
    def bundled_repo_root() -> Path:
        return Path(__file__).resolve().parent / "liveworld_runtime"

    @classmethod
    def from_pretrained(cls, pretrained_model_path: Any = None, **kwargs: Any) -> "LiveWorldRuntime":
        return cls(checkpoint_path=pretrained_model_path, **kwargs)

    def plan(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "model_id": "liveworld",
            "repo_root": str(self.repo_root),
            "source_revision": self.SOURCE_REVISION,
            "checkpoint_path": str(self.checkpoint_path),
            "base_model_path": str(self.base_model_path),
            "inputs": {key: str(value) for key, value in kwargs.items()},
        }

    def _write_system_config(
        self,
        destination: Path,
        *,
        checkpoint_root: Path,
        base_model: Path,
        qwen: Path,
        sam3: Path,
        stream3r: Path,
        dinov3: Path,
        width: int,
        height: int,
        num_frames: int,
        fps: int,
        seed: int,
    ) -> Path:
        system = yaml.safe_load((_CONFIG_ROOT / "infer_system_config_14b.yaml").read_text())
        observer = yaml.safe_load((_CONFIG_ROOT / "observer_liveworld_14b.yaml").read_text())
        observer["wan_model_name"] = str(base_model)
        observer_path = destination.parent / "observer_liveworld_14b.yaml"
        observer_path.write_text(yaml.safe_dump(observer, sort_keys=False), encoding="utf-8")

        system["runtime"]["seed"] = int(seed)
        system["observer"]["config"] = str(observer_path)
        system["observer"]["lora_path"] = str(checkpoint_root / "lora" / "model.pt")
        system["observer"]["sp_path"] = str(checkpoint_root / "state_adapter" / "model.pt")
        system["observer"]["frames_per_iter"] = min(int(num_frames), 65)
        system["observer"]["fps"] = int(fps)
        system["observer"]["width"] = int(width)
        system["observer"]["height"] = int(height)
        system["event"]["deduplication"]["model_path"] = str(dinov3)
        system["auxiliary"]["qwen_model_path"] = str(qwen)
        system["auxiliary"]["sam3_model_path"] = str(sam3 / "sam3.pt" if sam3.is_dir() else sam3)
        system["auxiliary"]["stream3r_model_path"] = str(stream3r)
        destination.write_text(yaml.safe_dump(system, sort_keys=False), encoding="utf-8")
        return destination

    def predict(
        self,
        *,
        prompt: str,
        geometry_npz: Any,
        output_path: Any,
        video_path: Any = None,
        image_path: Any = None,
        width: int = 832,
        height: int = 480,
        num_frames: int = 65,
        fps: int = 16,
        seed: int = 71,
        return_dict: bool = True,
        **_: Any,
    ) -> Any:
        geometry = require_path(geometry_npz, "LiveWorld geometry NPZ", kind="file")
        output = Path(output_path).expanduser().resolve()
        input_dir = output.parent / f".{output.stem}_liveworld_inputs"
        input_dir.mkdir(parents=True, exist_ok=True)
        if image_path:
            image = require_path(image_path, "LiveWorld first frame", kind="file")
        else:
            video = require_path(video_path, "LiveWorld source video", kind="file")
            image = _first_frame(video, input_dir / "first_frame.png")

        checkpoint_snapshot = require_path(self.checkpoint_path, "LiveWorld checkpoint root", kind="dir")
        # The official Hub snapshot stores the two runtime weights under ``ckpts/``.
        # Also accept an explicit path to that directory for backwards compatibility.
        nested_checkpoint_root = checkpoint_snapshot / "ckpts"
        checkpoint_root = (
            nested_checkpoint_root if nested_checkpoint_root.is_dir() else checkpoint_snapshot
        )
        require_path(checkpoint_root / "lora" / "model.pt", "LiveWorld LoRA", kind="file")
        require_path(checkpoint_root / "state_adapter" / "model.pt", "LiveWorld state adapter", kind="file")
        base = require_path(self.base_model_path, "Wan2.1-T2V-14B base model", kind="dir")
        for relative in (
            "config.json",
            "diffusion_pytorch_model.safetensors.index.json",
            "Wan2.1_VAE.pth",
            "models_t5_umt5-xxl-enc-bf16.pth",
        ):
            require_path(base / relative, f"Wan2.1-T2V-14B asset {relative}", kind="file")
        qwen = require_path(self.qwen_model_path, "Qwen3-VL-8B-Instruct", kind="dir")
        for relative in ("config.json", "model.safetensors.index.json", "tokenizer.json"):
            require_path(qwen / relative, f"Qwen3-VL-8B-Instruct asset {relative}", kind="file")
        sam3 = require_path(self.sam3_model_path, "SAM3 checkpoint")
        stream3r = require_path(self.stream3r_model_path, "STream3R checkpoint", kind="dir")
        require_path(stream3r / "config.json", "STream3R config", kind="file")
        require_path(stream3r / "model.safetensors", "STream3R weights", kind="file")
        dinov3 = require_path(self.dinov3_model_path, "DINOv3 checkpoint", kind="dir")
        require_path(dinov3 / "config.json", "DINOv3 config", kind="file")
        require_path(dinov3 / "model.safetensors", "DINOv3 weights", kind="file")
        if sam3.is_dir():
            require_path(sam3 / "sam3.pt", "SAM3 weights", kind="file")

        entities = input_dir / "entities.txt"
        storyline = input_dir / "storyline.json"
        entities.write_text("Nothing\n", encoding="utf-8")
        storyline.write_text("[]\n", encoding="utf-8")
        sample_config = {
            "geometry_file_name": str(geometry),
            "first_frame_image": str(image),
            "entities_file": str(entities),
            "storyline_file": str(storyline),
            "iter_input": {"0": {"scene_text": prompt.strip(), "fg_text": ""}},
        }
        sample_path = input_dir / "sample" / "infer_scripts" / "wrbench.yaml"
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        sample_path.write_text(yaml.safe_dump(sample_config, sort_keys=False), encoding="utf-8")
        system_path = self._write_system_config(
            input_dir / "system.yaml",
            checkpoint_root=checkpoint_root,
            base_model=base,
            qwen=qwen,
            sam3=sam3,
            stream3r=stream3r,
            dinov3=dinov3,
            width=width,
            height=height,
            num_frames=num_frames,
            fps=fps,
            seed=seed,
        )
        native_output = input_dir / "outputs"
        command = [
            self.python_executable,
            "scripts/infer.py",
            "--config",
            sample_path,
            "--system-config",
            system_path,
            "--output-root",
            native_output,
            "--device",
            self.device,
        ]
        result = execute_in_tree(
            command,
            cwd=self.repo_root,
            output_path=output,
            search_roots=(native_output,),
            preferred_names=("final_video.mp4",),
            env=self.env,
            python_paths=(self.repo_root,),
        )
        return result if return_dict else result.get("video")


__all__ = ["LiveWorldRuntime"]
