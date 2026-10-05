"""Bridge a local, pinned HunyuanWorld-Voyager checkout to its official CLI.

Tencent's community-licensed source and model weights stay outside this package.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

SOURCE_URL = "https://github.com/Tencent-Hunyuan/HunyuanWorld-Voyager"
SOURCE_REVISION = "6218ccd36df77d0254dcf24ad1fb77312f75b04a"
SOURCE_DIGEST = "c0c51217278231bac4c8956cc2cbf00053e3608cd814732ef6297dcb40129e29"
ROOT = Path(__file__).resolve().parents[4]
CHECKPOINT_FILES = (
    "Voyager/transformers/mp_rank_00_model_states.pt",
    "hunyuan-video-i2v-720p/vae/pytorch_model.pt",
    "text_encoder_i2v/model-00001-of-00004.safetensors",
    "text_encoder_i2v/model-00002-of-00004.safetensors",
    "text_encoder_i2v/model-00003-of-00004.safetensors",
    "text_encoder_i2v/model-00004-of-00004.safetensors",
    "text_encoder_2/model.safetensors",
)


def _source_digest(source_root: Path) -> tuple[str, int]:
    files = sorted(
        [
            *source_root.glob("voyager/**/*.py"),
            *(source_root / name for name in ("sample_image2video.py", "requirements.txt", "LICENSE", "NOTICE")),
        ]
    )
    digest = hashlib.sha256()
    for path in files:
        if not path.is_file():
            return "", len(files)
        digest.update(str(path.relative_to(source_root)).encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest(), len(files)


def _visible_gpu(device: str, visible: str | None) -> str:
    """Select one physical GPU; the official CLI always uses logical cuda:0."""
    value = str(device).strip().lower()
    if value == "cuda":
        index = 0
    elif value.startswith("cuda:") and value[5:].isdigit():
        index = int(value[5:])
    else:
        raise ValueError(f"Voyager official inference requires a CUDA device, got {device!r}")
    if visible is None:
        return str(index)
    devices = [part.strip() for part in visible.split(",") if part.strip()]
    if index >= len(devices):
        raise ValueError(f"{device!r} is unavailable with CUDA_VISIBLE_DEVICES={visible!r}")
    return devices[index]


class VoyagerOfficialRuntime:
    """Run the upstream 49-frame sampler without vendoring its source."""

    def __init__(
        self,
        *,
        source_root: str | Path | None = None,
        checkpoint_root: str | Path | None = None,
        python_executable: str | Path | None = None,
        device: str = "cuda",
    ) -> None:
        source = source_root or os.environ.get("WORLDFOUNDRY_VOYAGER_SOURCE_ROOT")
        checkpoints = checkpoint_root or os.environ.get("WORLDFOUNDRY_VOYAGER_CHECKPOINT_ROOT")
        self.source_root = Path(source or ROOT.parent / "repos" / "HunyuanWorld-Voyager").expanduser().resolve()
        self.checkpoint_root = (
            Path(checkpoints or ROOT.parent / "ckpts" / "tencent--HunyuanWorld-Voyager").expanduser().resolve()
        )
        self.python_executable = str(
            Path(python_executable or os.environ.get("WORLDFOUNDRY_VOYAGER_PYTHON") or sys.executable)
            .expanduser()
            .resolve()
        )
        self.device = str(device)

    def preflight(self, condition_dir: str | Path | None = None, *, use_context_block: bool = False) -> dict[str, Any]:
        missing: list[str] = []
        source_digest, source_file_count = _source_digest(self.source_root)
        if source_digest != SOURCE_DIGEST or source_file_count != 41:
            missing.append(
                f"official source must match {SOURCE_REVISION} (41-file digest {SOURCE_DIGEST}); "
                f"found {source_file_count} files and digest {source_digest or 'missing'} at {self.source_root}"
            )
        weights = list(CHECKPOINT_FILES)
        if use_context_block:
            weights.append("Voyager/transformers/mp_rank_00_model_states_context.pt")
        for relative in weights:
            path = self.checkpoint_root / relative
            if not path.is_file() or path.stat().st_size < 1_000_000:
                missing.append(f"checkpoint missing or incomplete: {path}")
        if not Path(self.python_executable).is_file():
            missing.append(f"Python interpreter missing: {self.python_executable}")
        condition = Path(condition_dir).expanduser().resolve() if condition_dir else None
        if condition is not None:
            for relative in ("ref_image.png", "ref_depth.exr"):
                if not (condition / relative).is_file():
                    missing.append(f"condition asset missing: {condition / relative}")
            for prefix, suffix in (("render", "png"), ("mask", "png"), ("depth", "exr")):
                paths = sorted((condition / "video_input").glob(f"{prefix}_*.{suffix}"))
                if len(paths) < 49 or any(
                    not (condition / "video_input" / f"{prefix}_{i:04d}.{suffix}").is_file() for i in range(49)
                ):
                    missing.append(f"condition requires 49 numbered {prefix} frames: {condition / 'video_input'}")
        return {
            "status": "ready" if not missing else "blocked",
            "missing": missing,
            "model_id": "hunyuanworld-voyager",
            "source_url": SOURCE_URL,
            "source_revision": SOURCE_REVISION if source_digest == SOURCE_DIGEST else None,
            "source_digest": source_digest,
            "source_root": str(self.source_root),
            "checkpoint_root": str(self.checkpoint_root),
            "condition_dir": str(condition) if condition else None,
            "python_executable": self.python_executable,
        }

    def predict(
        self,
        *,
        prompt: str,
        condition_dir: str | Path,
        output_path: str | Path,
        infer_steps: int = 50,
        seed: int = 0,
        height: int = 512,
        width: int = 768,
        use_context_block: bool = False,
        cuda_visible_devices: str | None = None,
        timeout_seconds: int = 7200,
    ) -> dict[str, Any]:
        if not 1 <= int(infer_steps) <= 100:
            raise ValueError("Voyager infer_steps must be between 1 and 100")
        if int(height) < 128 or int(width) < 128:
            raise ValueError("Voyager height and width must be at least 128")
        if int(timeout_seconds) < 1:
            raise ValueError("Voyager timeout_seconds must be positive")
        report = self.preflight(condition_dir, use_context_block=use_context_block)
        output = Path(output_path).expanduser().resolve().with_suffix(".mp4")
        if report["status"] != "ready":
            return {"status": "blocked", "blocked_reasons": report["missing"], "metadata": report}
        visible = cuda_visible_devices if cuda_visible_devices is not None else os.environ.get("CUDA_VISIBLE_DEVICES")
        selected_gpu = _visible_gpu(self.device, visible)
        output.parent.mkdir(parents=True, exist_ok=True)
        log_path = output.with_suffix(".mp4.log")
        with tempfile.TemporaryDirectory(prefix=f".{output.stem}-official-", dir=output.parent) as run_dir:
            command = [
                self.python_executable,
                str(self.source_root / "sample_image2video.py"),
                "--model",
                "HYVideo-T/2",
                "--model-base",
                str(self.checkpoint_root),
                "--input-path",
                str(Path(condition_dir).expanduser().resolve()),
                "--prompt",
                str(prompt),
                "--i2v-stability",
                "--infer-steps",
                str(int(infer_steps)),
                "--flow-reverse",
                "--flow-shift",
                "7.0",
                "--seed",
                str(int(seed)),
                "--embedded-cfg-scale",
                "6.0",
                "--use-cpu-offload",
                "--video-size",
                str(int(height)),
                str(int(width)),
                "--video-length",
                "49",
                "--save-path",
                run_dir,
            ]
            if use_context_block:
                command.append("--use-context-block")
            env = {
                **os.environ,
                "MODEL_BASE": str(self.checkpoint_root),
                "CUDA_VISIBLE_DEVICES": selected_gpu,
                "PYTHONNOUSERSITE": "1",
            }
            with log_path.open("w", encoding="utf-8") as log:
                try:
                    completed = subprocess.run(
                        command,
                        cwd=self.source_root,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                        timeout=int(timeout_seconds),
                    )
                except subprocess.TimeoutExpired:
                    return {
                        "status": "failed",
                        "error": f"official inference timed out; see {log_path}",
                        "metadata": report,
                    }
            videos = list(Path(run_dir).glob("*.mp4"))
            if completed.returncode or len(videos) != 1 or videos[0].stat().st_size == 0:
                return {
                    "status": "failed",
                    "error": f"official inference exited {completed.returncode} with {len(videos)} video artifacts; see {log_path}",
                    "metadata": report,
                }
            shutil.move(str(videos[0]), str(output))
        digest = hashlib.sha256(output.read_bytes()).hexdigest()
        return {
            "status": "succeeded",
            "artifact_path": str(output),
            "artifact_sha256": digest,
            "metadata": {
                **report,
                "log_path": str(log_path),
                "physical_gpu": selected_gpu,
                "infer_steps": int(infer_steps),
            },
        }


__all__ = ["VoyagerOfficialRuntime", "SOURCE_REVISION", "SOURCE_URL"]
