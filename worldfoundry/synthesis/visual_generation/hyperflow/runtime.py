"""Local-only HyperFlow inference using the pinned upstream H3 implementation.

The upstream package owns the two-time embedder, LoRA and fixed sigma grid;
WorldFoundry shares its existing H3 geometry and process/artifact machinery.
No H3 transformer, text encoder or VAE is copied into this adapter.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from worldfoundry.core.io.paths import hfd_root_path
from worldfoundry.runtime.assets import local_file_is_ready
from worldfoundry.runtime.in_tree_cli import execute_in_tree, require_path
from worldfoundry.runtime.probes import python_module_probe

UPSTREAM_REVISION = "1dd2f342aba5ab51da02b62885939655e8e268da"
WEIGHTS_FILENAME = "minimax_h3_hyperflow_8step_v1.0.safetensors"
WORKFLOWS = ("t2va", "fl2va", "ref2va")
_REFERENCE_EXTENSIONS = {
    "image": {".png", ".jpg", ".jpeg", ".webp", ".bmp"},
    "video": {".mp4", ".mov", ".mkv", ".webm", ".avi"},
    "audio": {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"},
}


def _reference(value: Any) -> tuple[str, Path]:
    if isinstance(value, Mapping):
        kind, location = value.get("kind"), value.get("path")
    else:
        kind, sep, location = str(value).partition(":")
        if not sep or kind not in _REFERENCE_EXTENSIONS:
            kind, location = None, value
    path = require_path(location, "HyperFlow reference", kind="file")
    kind = kind or next(
        (k for k, extensions in _REFERENCE_EXTENSIONS.items() if path.suffix.lower() in extensions), None
    )
    if kind not in _REFERENCE_EXTENSIONS:
        raise ValueError("Reference type must be image, video or audio; supply kind and path.")
    return kind, path


class HyperFlowRuntime:
    """Build and execute the official T2VA/FL2VA/Ref2VA inference commands."""

    def __init__(self, *, model_path=None, weights_path=None, python_executable=None, device="cuda", gpus=1):
        self.model_path = Path(model_path or hfd_root_path("MiniMaxAI--MiniMax-H3")).expanduser().resolve()
        self.weights_path = Path(weights_path or hfd_root_path("videorebirth--hyperflow")).expanduser().resolve()
        if self.weights_path.is_dir() or self.weights_path.suffix != ".safetensors":
            self.weights_path = self.weights_path / WEIGHTS_FILENAME
        self.python_executable = str(Path(python_executable or sys.executable).expanduser().absolute())
        self.device = str(device)
        self.gpus = int(gpus)
        if self.gpus not in (1, 2, 4):
            raise ValueError("HyperFlow supports 1, 2 or 4 GPUs.")
        if not self.device.startswith("cuda"):
            raise ValueError("HyperFlow inference requires CUDA.")

    def preflight(self, workflow="t2va"):
        if workflow not in WORKFLOWS:
            raise ValueError(f"Unknown HyperFlow workflow: {workflow}")
        root = self.model_path
        required = [root / "modular_model_index.json", self.weights_path, Path(self.python_executable)]
        components = ("transformer_ref" if workflow == "ref2va" else "transformer", "vae", "audio_vae", "text_encoder")
        for name in components:
            folder = root / name
            required.append(folder / "config.json")
            indices = sorted(folder.glob("*.safetensors.index.json"))
            if indices:
                try:
                    for index in indices:
                        shards = set(json.loads(index.read_text())["weight_map"].values())
                        if not shards:
                            raise ValueError("empty weight map")
                        required.extend(folder / shard for shard in shards)
                except (OSError, ValueError, KeyError, TypeError):
                    required.append(folder / "invalid-weight-index")
            else:
                shards = list(folder.glob("*.safetensors"))
                required.extend(shards or [folder / "missing-weights.safetensors"])
        required.extend(
            root / name / filename
            for name, filename in (
                ("tokenizer", "tokenizer_config.json"),
                ("tokenizer", "tokenizer.json"),
                ("processor", "preprocessor_config.json"),
                ("scheduler", "scheduler_config.json"),
                ("audio_scheduler", "scheduler_config.json"),
            )
        )
        missing = [str(path) for path in required if not local_file_is_ready(path)]
        environment = python_module_probe(
            Path(self.python_executable),
            ("hyperflow_h3", "diffusers", "transformers", "peft", "av", "torch"),
            pythonpath=[],
            timeout=30,
        )
        if environment["ok"]:
            # The upstream lower bound accepts Transformers 4.57, whose
            # Qwen3VLProcessor lacks create_mm_token_type_ids used by 0.40.
            code = (
                "import json; from importlib.metadata import version; "
                "from packaging.specifiers import SpecifierSet; "
                "required={'diffusers':'==0.40.0','transformers':'==5.12.1',"
                "'peft':'==0.18.0','safetensors':'>=0.8'}; "
                "versions={name:version(name) for name in required}; "
                "print(json.dumps({'versions':versions,'compatible':all("
                "versions[name] in SpecifierSet(spec) for name,spec in required.items())}))"
            )
            try:
                checked = subprocess.run(
                    [self.python_executable, "-s", "-c", code], capture_output=True, text=True, timeout=30, check=True
                )
                environment.update(json.loads(checked.stdout))
                environment["ok"] = environment["compatible"]
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                environment.update(ok=False, error=f"Cannot validate HyperFlow dependency versions: {exc}")
        return {
            "status": "blocked" if missing or not environment["ok"] else "ready",
            "model_id": "hyperflow",
            "workflow": workflow,
            "missing_or_incomplete_files": missing,
            "source_revision": UPSTREAM_REVISION,
            "python_executable": self.python_executable,
            "environment": environment,
            "runtime_requirement": "hyperflow-h3 at the recorded revision, diffusers==0.40.0, transformers==5.12.1, peft==0.18.0, safetensors>=0.8, av",
        }

    def build_plan(
        self,
        *,
        prompt,
        output_path,
        images=None,
        image=None,
        last_image=None,
        references=None,
        workflow="auto",
        num_frames=124,
        height=None,
        width=None,
        seed=42,
        fps=24,
        num_inference_steps=8,
        memory_reserve_margin="24GB",
        offload=True,
        attention_backend="sdpa",
        sol_attn=False,
    ):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("HyperFlow requires a non-empty prompt.")
        if num_inference_steps != 8:
            raise ValueError("HyperFlow v1.0 requires its fixed 8-step grid.")
        if fps != 24:
            raise ValueError("HyperFlow uses the official H3 24 fps output.")
        if isinstance(images, (list, tuple)):
            if not 1 <= len(images) <= 2:
                raise ValueError("Use one first image or two first/last images; use references for Ref2VA.")
            if image is not None or last_image is not None:
                raise ValueError("Pass images or image/last_image, not both.")
            image, last_image = images[0], images[1] if len(images) == 2 else None
        elif images is not None:
            if image is not None:
                raise ValueError("Pass images or image, not both.")
            image = images
        refs = [] if references is None else references
        if not isinstance(refs, (list, tuple)):
            raise TypeError("references must be an ordered list of local paths or kind/path mappings.")
        if refs and (image is not None or last_image is not None):
            raise ValueError("Ref2VA references cannot be combined with FL2VA keyframes.")
        selected = "ref2va" if refs else "fl2va" if image is not None or last_image is not None else "t2va"
        if workflow not in ("auto", selected):
            raise ValueError(f"Inputs require workflow={selected}, got {workflow}.")
        if isinstance(num_frames, bool) or int(num_frames) != num_frames or num_frames <= 0:
            raise ValueError("num_frames must be a positive integer.")
        from worldfoundry.base_models.diffusion_model.schedulers.minimax_h3 import minimax_h3_align_frame_count

        frames = minimax_h3_align_frame_count(int(num_frames))
        if not 120 <= frames <= 360:
            raise ValueError("HyperFlow requires 5–15 seconds at 24 fps after frame alignment (124–345 frames).")
        output = Path(output_path).expanduser().resolve()
        if output.suffix.lower() != ".mp4":
            raise ValueError("HyperFlow output_path must end in .mp4.")
        module = "generate_ref2va" if refs else "generate_fl2va"
        command = [
            self.python_executable,
            "-m",
            f"hyperflow_h3.examples.{module}",
            "--model",
            str(self.model_path),
            "--weights",
            str(self.weights_path),
            "--prompt",
            prompt,
            "--output",
            str(output),
            "--num-frames",
            str(frames),
            "--seed",
            str(seed),
            "--gpus",
            str(self.gpus),
            "--device",
            self.device,
            "--attention-backend",
            attention_backend,
            "--memory-reserve-margin",
            str(memory_reserve_margin),
        ]
        for key, value in (("height", height), ("width", width)):
            if value is not None:
                if isinstance(value, bool) or int(value) != value or value <= 0 or value % 32:
                    raise ValueError(f"{key} must be a positive multiple of 32.")
                command.extend((f"--{key}", str(value)))
        for flag, value in (("--image", image), ("--last-image", last_image)):
            if value is not None:
                command.extend((flag, str(require_path(value, "HyperFlow keyframe", kind="file"))))
        counts = dict.fromkeys(_REFERENCE_EXTENSIONS, 0)
        for value in refs:
            kind, path = _reference(value)
            counts[kind] += 1
            command.extend(("--ref", f"{kind}:{path}"))
        if len(refs) > 12 or counts["image"] > 9 or counts["video"] > 3 or counts["audio"] > 3:
            raise ValueError("Ref2VA supports at most 9 images, 3 videos, 3 audio clips and 12 references total.")
        if counts["audio"] and not (counts["image"] or counts["video"]):
            raise ValueError("Audio references require an image or video reference.")
        if not offload:
            command.append("--no-offload")
        if sol_attn:
            command.append("--sol-attn")
        return {
            "command": command,
            "workflow": selected,
            "num_frames": frames,
            "output_path": str(output),
            "env": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONNOUSERSITE": "1"},
        }

    def run(self, plan, *, timeout_seconds=7200):
        preflight = self.preflight(plan["workflow"])
        if preflight["status"] != "ready":
            raise RuntimeError(
                f"HyperFlow preflight failed: files={preflight['missing_or_incomplete_files']}; "
                f"environment={preflight['environment']}"
            )
        result = execute_in_tree(
            plan["command"],
            cwd=Path(__file__).parent,
            output_path=plan["output_path"],
            env=plan["env"],
            timeout=timeout_seconds,
        )
        if result["status"] != "succeeded":
            raise RuntimeError(result["error"])
        result.update(
            model_id="hyperflow",
            workflow=plan["workflow"],
            num_frames=plan["num_frames"],
            fps=24,
            source_revision=UPSTREAM_REVISION,
        )
        return result
