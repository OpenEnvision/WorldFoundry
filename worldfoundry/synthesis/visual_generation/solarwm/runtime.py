"""SolarWM's released inference routes, without vendored backbones or trainers."""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

import yaml

from worldfoundry.core.io import resolve_data_path
from worldfoundry.runtime.official_inference import InferencePlan, OfficialInferenceRuntime, local_path


class SolarWMRuntime(OfficialInferenceRuntime):
    MODEL_ID = "solarwm"
    REVISION = "2fe1ea9aca072f787bbba07ee9380b0e1ea7563b"
    REQUIRED_SOURCE_FILES = ("src/solarwm/__main__.py", "src/solarwm/cli.py")
    ASSET_KEYS = {
        "model.codec.video_vae_path",
        "model.codec.gemma4_path",
        "data.silence_latents_path",
        "data.encoder_contract_path",
        "inference.negative_caption_cache",
        "validation.inference.negative_caption_cache",
        "inference.plan",
        "inference.dataset_root",
    }
    REQUEST_KEYS = {
        "data.test_index",
        "data.index",
        "inference.sample_count",
        "inference.seed",
        "validation.sample_count",
        "validation.noise_seed",
        "validation.selection_seed",
        "inference.selection_seed",
        "checkpoint.weights",
        "checkpoint.weight_source",
        "model.adapter_weights",
    }

    def __init__(
        self,
        *,
        source_root="${WORLDFOUNDRY_MODEL_SOURCE_DIR}/Junchao-cs--SolarWM",
        checkpoint_dir=None,
        base_path=None,
        data_root=None,
        variant="wan-5b-dmd",
        assets=None,
        python_executable=None,
        device="cuda",
    ):
        super().__init__(source_root=source_root, python_executable=python_executable, device=device)
        config = yaml.safe_load(Path(resolve_data_path("models/runtime/configs/solarwm/routes.yaml")).read_text())
        if variant not in config["variants"]:
            raise ValueError(f"Unknown SolarWM variant {variant!r}; choose {list(config['variants'])}")
        self.variant, self.route = variant, config["variants"][variant]
        self.checkpoint_dir = local_path(checkpoint_dir) if checkpoint_dir is not None else None
        self.base_path = local_path(base_path) if base_path is not None else None
        self.data_root = local_path(data_root) if data_root is not None else None
        self.assets = dict(assets or {})
        unknown = self.assets.keys() - self.ASSET_KEYS
        if unknown:
            raise ValueError(f"Unsupported SolarWM asset keys: {sorted(unknown)}")
        self.assets = {key: str(local_path(value)) for key, value in self.assets.items()}
        self.config_path = self.source_root / self.route["config"]

    def required_assets(self):
        return (
            self.config_path,
            *(p for p in (self.checkpoint_dir, self.base_path, self.data_root) if p),
            *(Path(value) for value in self.assets.values()),
        )

    def preflight(self):
        result = super().preflight()
        missing = [key for key in ("checkpoint_dir", "base_path", "data_root") if getattr(self, key) is None]
        required = (
            {"model.codec.video_vae_path", "inference.negative_caption_cache"}
            if self.variant == "ltx"
            else {"data.silence_latents_path", "data.encoder_contract_path"}
            if self.variant.startswith("h3")
            else set()
        )
        if self.variant == "h3-dmd-long":
            required.update({"inference.plan", "inference.dataset_root"})
        missing.extend(sorted(required - self.assets.keys()))
        result["missing_options"] = missing
        result["variant"] = self.variant
        if missing:
            result["status"] = "blocked"
        return result

    def run_plan(self, plan, *, timeout_seconds=7200):
        # Upstream publishes with renameat2(RENAME_NOREPLACE). Shared/FUSE
        # filesystems may reject that operation even though ordinary writes
        # work. Preserve its publication protocol on local scratch, then copy
        # the completed artifacts and diagnostics to the requested directory.
        output = Path(plan.output_dir)
        if output.exists() and any(output.rglob("*.mp4")):
            raise FileExistsError(f"Inference output already contains videos: {output}")
        with tempfile.TemporaryDirectory(prefix="worldfoundry-solarwm-") as scratch:
            command = list(plan.command)
            for index, argument in enumerate(command):
                if index and command[index - 1] == "--set":
                    key, _, value = argument.partition("=")
                    if key == "runtime.output_dir":
                        command[index] = f"{key}={json.dumps(scratch)}"
                    elif key == "inference.work_dir":
                        command[index] = f"{key}={json.dumps(str(Path(scratch) / 'conditioning'))}"
            staged = replace(plan, command=tuple(command), output_dir=scratch)
            result = super().run_plan(staged, timeout_seconds=timeout_seconds)
            shutil.copytree(scratch, output, dirs_exist_ok=True)

            def published(value):
                if isinstance(value, str) and value.startswith(scratch + "/"):
                    return str(output / Path(value).relative_to(scratch))
                if isinstance(value, list):
                    return [published(item) for item in value]
                return value

            result = {key: published(value) for key, value in result.items()}
            Path(result["metadata_path"]).write_text(json.dumps(result, indent=2) + "\n")
            return result

    def build_plan(self, *, request, output_dir):
        request = dict(request)
        # Upstream cases carry their own caption, image and camera trajectory.
        if set(request) - {"overrides"}:
            raise ValueError(
                "SolarWM reads image/prompt/camera from its data index; pass only overrides for indexed inference."
            )
        overrides = dict(request.get("overrides") or {})
        unknown = overrides.keys() - self.REQUEST_KEYS
        if unknown:
            raise ValueError(f"Unsupported SolarWM inference overrides: {sorted(unknown)}")
        if self.checkpoint_dir is None or self.base_path is None or self.data_root is None:
            raise ValueError("SolarWM requires checkpoint_dir, base_path and data_root.")
        if not self.config_path.is_file():
            raise FileNotFoundError(self.config_path)
        config = yaml.safe_load(self.config_path.read_text())
        if config.get("action") != "infer" or config.get("model", {}).get("family") != self.route["family"]:
            raise ValueError("SolarWM route must resolve to the pinned inference action and model family.")
        processes = int(self.route["processes"])
        if self.device.startswith("cuda:") and len(self.device[5:].split(",")) != processes:
            raise ValueError(f"{self.variant} requires {processes} CUDA devices with the released route.")
        values = {
            self.route["base_key"]: str(self.base_path),
            self.route["checkpoint_key"]: str(self.checkpoint_dir),
            "data.index_root": str(self.data_root),
            "data.transport.root": str(self.data_root),
            **self.assets,
            **overrides,
            "runtime.output_dir": str(local_path(output_dir)),
        }
        # Isolated output directories allow the upstream create-only publication contract.
        if self.variant == "wan-5b-dmd":
            values["inference.run_id"] = "worldfoundry"
        if self.variant == "h3-dmd-long":
            values["inference.work_dir"] = str(local_path(output_dir) / "conditioning")
        command = [
            self.python_executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={processes}",
            "-m",
            "solarwm",
            "infer",
            "--config",
            str(self.config_path),
        ]
        for key, value in values.items():
            command.extend(("--set", f"{key}={json.dumps(value, ensure_ascii=False)}"))
        if self.variant == "wan-5b-dmd":
            artifacts, markers = ("generate/**/*.mp4",), ("runs/worldfoundry/COMPLETE.json",)
        elif self.variant == "h3-dmd-long":
            artifacts, markers = ("*/*.mp4",), ("WORKER_COMPLETE.json", "run-result.json")
        elif self.variant.startswith("h3"):
            artifacts, markers = (
                ("inference-parts/**/generated.mp4",),
                ("inference-parts/**/COMPLETE.json", "run-result.json"),
            )
        elif self.variant == "ltx":
            artifacts, markers = ("inference/**/video.mp4",), ("inference/COMPLETE.json", "run-result.json")
        else:
            artifacts, markers = ("generation/**/video.mp4",), ("generation/COMPLETE.json", "run-result.json")
        return InferencePlan(
            self.MODEL_ID,
            tuple(command),
            str(self.source_root),
            str(local_path(output_dir)),
            self.REVISION,
            self.environment(self.source_root / "src"),
            artifact_globs=artifacts,
            completion_globs=markers,
        )
