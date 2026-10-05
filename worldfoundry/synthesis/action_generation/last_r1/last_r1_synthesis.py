"""Offline LaST-R1 action inference through its pinned official model code."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from worldfoundry.core.io.paths import official_runtime_repo_path, project_root, resolve_worldfoundry_path
from worldfoundry.evaluation.models.runtime.profiles import load_runtime_profile
from worldfoundry.synthesis.action_generation.base_action_synthesis import ActionModelSynthesis


class LastR1Synthesis(ActionModelSynthesis):
    """One image-and-instruction policy step; no LIBERO success-rate claim."""

    MODEL_ID = "last-r1"
    REVISION = "ba4e939632427e066f734bf9f7abfa11cc4911f6"
    VARIANT = "LaST-R1-RL/last-r1-rl-libero_10"

    def __init__(self, profile, *, checkpoint_path: Path, source_root: Path, python_executable: str, device: str):
        super().__init__(profile, device=device)
        self.checkpoint_path = checkpoint_path
        self.source_root = source_root
        self.python_executable = python_executable

    @classmethod
    def from_pretrained(cls, pretrained_model_path=None, args=None, device="cuda", **kwargs):
        del args
        options = dict(pretrained_model_path) if isinstance(pretrained_model_path, Mapping) else {}
        if pretrained_model_path is not None and not isinstance(pretrained_model_path, Mapping):
            options["checkpoint_path"] = pretrained_model_path
        options.update(kwargs)
        profile = load_runtime_profile(cls.MODEL_ID, profile_path=options.get("profile_path"))
        default_checkpoint = f"${{WORLDFOUNDRY_CKPT_DIR}}/chenhao01--LaST-R1/{cls.VARIANT}"
        checkpoint = resolve_worldfoundry_path(options.get("checkpoint_path") or default_checkpoint).resolve()
        if not options.get("checkpoint_path") and not os.environ.get("WORLDFOUNDRY_CKPT_DIR"):
            sibling = (project_root().parent / "ckpts" / "chenhao01--LaST-R1" / cls.VARIANT).resolve()
            if not (checkpoint / "config.json").is_file() and (sibling / "config.json").is_file():
                checkpoint = sibling
        source = resolve_worldfoundry_path(
            options.get("source_root")
            or official_runtime_repo_path("LaST-R1", specific_env="WORLDFOUNDRY_LAST_R1_SOURCE_DIR")
        ).resolve()
        if not options.get("source_root") and not os.environ.get("WORLDFOUNDRY_LAST_R1_SOURCE_DIR"):
            sibling_source = (project_root().parent / "model" / "LaST-R1").resolve()
            if not (source / "transformers/models/qwen3_vl/modeling_qwen3_vl.py").is_file() and (
                sibling_source / "transformers/models/qwen3_vl/modeling_qwen3_vl.py"
            ).is_file():
                source = sibling_source
        executable = str(options.get("python_executable") or sys.executable)
        executable = str(Path(shutil.which(executable) or executable).expanduser().absolute())
        selected_device = str(device or options.get("device") or "cuda")
        if selected_device != "cuda" and not (selected_device.startswith("cuda:") and selected_device[5:].isdigit()):
            raise ValueError("LaST-R1 requires device='cuda' or a single explicit CUDA index")
        return cls(profile, checkpoint_path=checkpoint, source_root=source, python_executable=executable, device=selected_device)

    def preflight(self) -> dict[str, Any]:
        required = [
            self.source_root / "transformers/models/qwen3_vl/modeling_qwen3_vl.py",
            self.source_root / "verl/workers/actor/action_tokenizer.py",
            self.checkpoint_path / "config.json",
            self.checkpoint_path / "statistics.json",
            self.checkpoint_path / "model.safetensors.index.json",
            Path(self.python_executable),
        ]
        missing = [str(path) for path in required if not path.is_file()]
        index_path = self.checkpoint_path / "model.safetensors.index.json"
        if index_path.is_file():
            try:
                index = json.loads(index_path.read_text())
                for shard in sorted(set(index["weight_map"].values())):
                    if not (self.checkpoint_path / shard).is_file():
                        missing.append(str(self.checkpoint_path / shard))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                missing.append(f"invalid weight index: {exc}")
        revision = None
        if (self.source_root / ".git").exists():
            try:
                revision = subprocess.check_output(
                    ["git", "-C", str(self.source_root), "rev-parse", "HEAD"],
                    text=True,
                    timeout=10,
                ).strip()
            except (OSError, subprocess.SubprocessError):
                pass
        return {
            "status": "ready" if not missing and revision == self.REVISION else "blocked",
            "model_id": self.MODEL_ID,
            "checkpoint": str(self.checkpoint_path),
            "source_root": str(self.source_root),
            "source_revision": revision,
            "source_matches_pin": revision == self.REVISION,
            "missing_paths": missing,
            "variant": "libero_10",
            "validation_scope": "offline_action_inference_only",
        }

    @staticmethod
    def _save_image(value: Any, destination: Path) -> None:
        if isinstance(value, (str, Path)):
            with Image.open(value) as image:
                image.convert("RGB").save(destination)
            return
        if isinstance(value, Image.Image):
            value.convert("RGB").save(destination)
            return
        array = np.asarray(value)
        if array.ndim != 3 or array.shape[-1] != 3 or array.dtype != np.uint8:
            raise ValueError("LaST-R1 images must be an RGB image path, PIL image, or uint8 H×W×3 array")
        Image.fromarray(array, "RGB").save(destination)

    def predict(
        self,
        prompt: str | None = None,
        images: Any = None,
        video: Any = None,
        interactions: Any = None,
        output_path: str | Path | None = None,
        fps: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del fps
        if video is not None:
            raise ValueError("LaST-R1 offline policy inference accepts one image, not video")
        if interactions:
            raise ValueError("LaST-R1 offline policy inference does not consume action history")
        instruction = str(prompt or kwargs.get("instruction") or "").strip()
        if not instruction:
            raise ValueError("LaST-R1 requires a nonempty instruction")
        image = images if images is not None else kwargs.get("image")
        if image is None:
            image = kwargs.get("ref_image_path")
        if isinstance(image, (list, tuple)):
            if len(image) != 1:
                raise ValueError("LaST-R1 requires exactly one image")
            image = image[0]
        if image is None:
            raise ValueError("LaST-R1 requires an image")
        preflight = self.preflight()
        if preflight["status"] != "ready":
            raise FileNotFoundError(f"LaST-R1 preflight blocked: {preflight}")

        requested = Path(output_path).expanduser().resolve() if output_path else None
        run_root = requested.parent if requested else (project_root() / "tmp" / "last-r1")
        run_dir = run_root / uuid.uuid4().hex
        run_dir.mkdir(parents=True, exist_ok=False)
        image_path = run_dir / "input.png"
        self._save_image(image, image_path)
        request_path = run_dir / "request.json"
        request_path.write_text(
            json.dumps(
                {
                    "run_dir": str(run_dir),
                    "source_root": str(self.source_root),
                    "checkpoint_path": str(self.checkpoint_path),
                    "image_path": str(image_path),
                    "instruction": instruction,
                    "seed": int(kwargs.get("seed", 42)),
                    "temperature": float(kwargs.get("temperature", 1.6)),
                    "center_crop": bool(kwargs.get("center_crop", True)),
                },
                indent=2,
            ) + "\n"
        )
        env = os.environ.copy()
        env.update(
            PYTHONPATH=os.pathsep.join((str(self.source_root), str(project_root()), env.get("PYTHONPATH", ""))),
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            TOKENIZERS_PARALLELISM="false",
            OMP_NUM_THREADS="4",
            NUMEXPR_MAX_THREADS="64",
        )
        if self.device.startswith("cuda:"):
            env["CUDA_VISIBLE_DEVICES"] = self.device[5:]
        command = [self.python_executable, "-m", "worldfoundry.synthesis.action_generation.last_r1.worker", str(request_path)]
        (run_dir / "plan.json").write_text(json.dumps({"command": command, "workdir": str(project_root()), "source_revision": self.REVISION, "preflight": preflight}, indent=2) + "\n")
        stdout_path, stderr_path = run_dir / "stdout.log", run_dir / "stderr.log"
        timeout = int(kwargs.get("timeout_seconds", 3600))
        if timeout <= 0:
            raise ValueError("timeout_seconds must be positive")
        with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
            process = subprocess.Popen(command, cwd=project_root(), env=env, stdout=stdout, stderr=stderr, start_new_session=True)
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise TimeoutError(f"LaST-R1 inference exceeded {timeout} seconds; see {stderr_path}")
        result_path = run_dir / "result.json"
        if code != 0 or not result_path.is_file():
            raise RuntimeError(f"LaST-R1 inference failed (exit={code}); see {stderr_path}")
        result = json.loads(result_path.read_text())
        if result.get("status") != "success" or result.get("action_shape") != [1, 8, 7] or not result.get("all_finite"):
            raise RuntimeError(f"LaST-R1 inference returned invalid actions; see {result_path}")
        artifact = requested or (run_dir / "action_trace.json")
        artifact.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(result_path, artifact)
        result.update(
            artifact_kind="action_trace",
            artifact_path=str(artifact),
            artifact_files=[str(artifact)],
            metadata_path=str(result_path),
            stderr_path=str(stderr_path),
            validation_scope="offline_action_inference_only",
            source_matches_pin=True,
            observation_state_used=False,
        )
        return result
