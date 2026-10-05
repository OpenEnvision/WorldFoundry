"""Run BAAI Emu3.5's official interleaved text/image implementation.

The official source is an external, pinned checkout. No model weights or generated
artifacts are bundled with WorldFoundry.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

SOURCE_URL = "https://github.com/baaivision/Emu3.5"
SOURCE_REVISION = "8fcf1da48f9c5252fe0a0b8dc842e07b6efcd745"
SOURCE_LICENSE = "Apache-2.0"
ROOT = Path(__file__).resolve().parents[4]
TASKS = {"story", "howto"}


def _cuda_device(device: str, cuda_visible_devices: str | None = None) -> str:
    """Resolve a logical CUDA device after any CUDA_VISIBLE_DEVICES filtering."""
    value = str(device).strip().lower()
    if value == "cuda":
        value = "cuda:0"
    if not value.startswith("cuda:") or not value[5:].isdigit():
        raise ValueError(f"Emu3.5 requires a CUDA device, got {device!r}")
    visible = cuda_visible_devices if cuda_visible_devices is not None else os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        count = len([item for item in visible.split(",") if item.strip()])
        if int(value[5:]) >= count:
            raise ValueError(f"{value} is unavailable with CUDA_VISIBLE_DEVICES={visible!r}")
    return value


def _path(value: Any) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    return Path(value).expanduser().resolve()


def _checkpoint_root() -> Path:
    return Path(os.environ.get("WORLDFOUNDRY_HFD_ROOT") or ROOT.parent / "ckpts").expanduser().resolve()


def _source_revision(path: Path | None) -> str | None:
    if path is None or not (path / ".git").exists():
        return None
    completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, text=True, capture_output=True, check=False)
    return completed.stdout.strip() if completed.returncode == 0 else None


class Emu35Runtime:
    """Bridge a pinned official checkout to a protobuf interleaved response."""

    def __init__(
        self,
        *,
        checkpoint_path: Any = None,
        repo_root: Any = None,
        vision_tokenizer_path: Any = None,
        python_executable: Any = None,
        device: str = "cuda",
    ) -> None:
        base = _checkpoint_root()
        self.checkpoint_path = _path(checkpoint_path) or base / "BAAI--Emu3.5"
        self.repo_root = _path(repo_root or os.environ.get("WORLDFOUNDRY_EMU35_REPO_ROOT")) or (
            ROOT.parent / "repos" / "Emu3.5"
        )
        self.vision_tokenizer_path = _path(vision_tokenizer_path) or base / "BAAI--Emu3.5-VisionTokenizer"
        self.python_executable = str(python_executable or sys.executable)
        self.device = str(device)

    def preflight(self) -> dict[str, Any]:
        missing: list[str] = []
        repo = self.repo_root
        checkpoint = self.checkpoint_path
        vision = self.vision_tokenizer_path
        for relative in (
            "inference.py",
            "src/emu3p5/configuration_emu3.py",
            "src/emu3p5/modeling_emu3.py",
            "src/utils/generation_utils.py",
            "src/tokenizer_emu3_ibq/emu3.tiktoken",
            "src/tokenizer_emu3_ibq/emu3_vision_tokens.txt",
            "src/tokenizer_emu3_ibq/tokenization_emu3.py",
            "src/tokenizer_emu3_ibq/tokenizer_config.json",
        ):
            if not (repo / relative).is_file():
                missing.append(f"official source missing: {repo / relative}")
        revision = _source_revision(repo)
        if revision != SOURCE_REVISION:
            missing.append(
                f"official source must be checked out at {SOURCE_REVISION}; found {revision or 'unverified'}"
            )

        index_path = checkpoint / "model.safetensors.index.json"
        if not index_path.is_file():
            missing.append(f"main checkpoint index missing: {index_path}")
        else:
            try:
                names = set(json.loads(index_path.read_text())["weight_map"].values())
                for name in sorted(names):
                    shard = checkpoint / name
                    if not shard.is_file() or shard.stat().st_size < 1024:
                        missing.append(f"main checkpoint shard missing/incomplete: {shard}")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                missing.append(f"main checkpoint index invalid: {type(exc).__name__}: {exc}")
        config_path = vision / "config.json"
        if not config_path.is_file():
            missing.append(f"vision tokenizer config missing: {config_path}")
        else:
            try:
                if json.loads(config_path.read_text()).get("model_type") != "Emu3p5VisionVQ":
                    missing.append(f"vision tokenizer config has unexpected architecture: {config_path}")
            except (OSError, ValueError, TypeError) as exc:
                missing.append(f"vision tokenizer config invalid: {type(exc).__name__}: {exc}")
        weight_path = vision / "model.safetensors"
        if not weight_path.is_file() or weight_path.stat().st_size < 1024:
            missing.append(f"vision tokenizer weights missing/incomplete: {weight_path}")
        for relative in ("configuration_emu3p5visionvq.py", "modeling_emu3p5visionvq.py"):
            path = vision / relative
            if not path.is_file():
                missing.append(f"vision tokenizer code missing: {path}")
        if not Path(self.python_executable).is_file():
            missing.append(f"Python interpreter missing: {self.python_executable}")
        return {
            "status": "ready" if not missing else "blocked",
            "missing": missing,
            "model_id": "emu3.5",
            "modality": "interleaved_text_image",
            "source_url": SOURCE_URL,
            "source_revision": revision,
            "source_license": SOURCE_LICENSE,
            "repo_root": str(repo),
            "checkpoint_path": str(checkpoint),
            "vision_tokenizer_path": str(vision),
            "python_executable": self.python_executable,
        }

    def predict(
        self,
        *,
        prompt: str,
        output_path: str | Path,
        image_path: str | Path | None = None,
        task_type: str = "story",
        max_new_tokens: int = 4096,
        target_height: int | None = None,
        target_width: int | None = None,
        seed: int = 6666,
        cuda_visible_devices: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        if task_type not in TASKS:
            raise ValueError(f"Emu3.5 task_type must be one of {sorted(TASKS)}")
        if not 1 <= int(max_new_tokens) <= 32768:
            raise ValueError("Emu3.5 max_new_tokens must be between 1 and 32768")
        if (target_height is None) != (target_width is None):
            raise ValueError("Emu3.5 target_height and target_width must be set together")
        if target_height is not None and (not 1 <= int(target_height) <= 99 or not 1 <= int(target_width) <= 99):
            raise ValueError("Emu3.5 target_height and target_width must be between 1 and 99")
        device = _cuda_device(self.device, cuda_visible_devices)
        image = _path(image_path)
        if image_path is not None and (image is None or not image.is_file()):
            raise FileNotFoundError(f"Emu3.5 input image missing: {image_path}")
        output = Path(output_path).expanduser().resolve().with_suffix(".pb")
        report = self.preflight()
        if report["status"] != "ready":
            return {
                "status": "blocked",
                "blocked_reasons": report["missing"],
                "artifact_path": str(output),
                "metadata": report,
            }
        output.parent.mkdir(parents=True, exist_ok=True)
        log_path = output.with_suffix(".pb.log")
        command = [
            self.python_executable,
            "-m",
            "worldfoundry.synthesis.visual_generation.emu35.official_driver",
            "--repo-root",
            str(self.repo_root),
            "--checkpoint-path",
            str(self.checkpoint_path),
            "--vision-tokenizer-path",
            str(self.vision_tokenizer_path),
            "--output-path",
            str(output),
            "--prompt",
            str(prompt),
            "--task-type",
            task_type,
            "--max-new-tokens",
            str(int(max_new_tokens)),
            "--device",
            device,
            "--seed",
            str(int(seed)),
        ]
        if target_height is not None:
            command.extend(["--target-height", str(int(target_height)), "--target-width", str(int(target_width))])
        if image is not None:
            command.extend(["--image-path", str(image)])
        env = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join([str(ROOT), str(self.repo_root), os.environ.get("PYTHONPATH", "")]),
        }
        env["NUMEXPR_NUM_THREADS"] = "8"
        env["OMP_NUM_THREADS"] = "8"
        env["OPENBLAS_NUM_THREADS"] = "8"
        env["MKL_NUM_THREADS"] = "8"
        if cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command, cwd=self.repo_root, env=env, stdout=log, stderr=subprocess.STDOUT, check=False
            )
        metadata = {
            "model_id": "emu3.5",
            "modality": "interleaved_text_image",
            "task_type": task_type,
            "source_revision": SOURCE_REVISION,
            "checkpoint_path": str(self.checkpoint_path),
            "vision_tokenizer_path": str(self.vision_tokenizer_path),
            "log_path": str(log_path),
            "requested_max_new_tokens": int(max_new_tokens),
            "device": device,
            "target_height": target_height,
            "target_width": target_width,
        }
        if completed.returncode or not output.is_file() or output.stat().st_size == 0:
            return {
                "status": "failed",
                "error": f"official Emu3.5 inference exited {completed.returncode}; see {log_path}",
                "artifact_path": str(output),
                "metadata": metadata,
            }
        digest = hashlib.sha256(output.read_bytes()).hexdigest()
        return {
            "status": "succeeded",
            "artifact_path": str(output),
            "artifact_sha256": digest,
            "metadata": metadata,
        }


__all__ = ["Emu35Runtime", "SOURCE_REVISION", "SOURCE_URL"]
