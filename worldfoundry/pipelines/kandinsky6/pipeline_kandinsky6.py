"""Public artifact pipeline for native Kandinsky-6 audio-video recipes."""

import json
import time
from pathlib import Path

from worldfoundry.operators.kandinsky6_operator import Kandinsky6Operator
from worldfoundry.pipelines.pipeline_utils import PipelineABC


class Kandinsky6Pipeline(PipelineABC):
    MODEL_ID = "kandinsky6-pro-distill"

    def __init__(self, native_pipeline, device="cuda"):
        super().__init__(model_id=native_pipeline.model_id, operator=Kandinsky6Operator(), device=device)
        self.native_pipeline = native_pipeline
        self.synthesis_model = native_pipeline

    @classmethod
    def from_pretrained(cls, model_path=None, required_components=None, device="cuda", model_id=None, **kwargs):
        import torch

        from worldfoundry.base_models.diffusion_model.optimizations import RuntimePolicy
        from worldfoundry.base_models.diffusion_model.optimizations.policy import (
            parse_attention_backend,
            parse_offload_policy,
        )
        from worldfoundry.base_models.diffusion_model.pipeline import NativeDiffusionPipeline

        options = cls._runtime_options(model_path, required_components, kwargs)
        resolved_id = str(model_id or options.pop("model_id", cls.MODEL_ID))
        root = options.get("checkpoint_dir", model_path if isinstance(model_path, (str, Path)) else None)
        if root is None and options.get("hf_models_root"):
            from worldfoundry.base_models.diffusion_model.recipes.registry import default_native_diffusion_registry

            recipe = default_native_diffusion_registry().resolve(resolved_id)
            repo = recipe.checkpoints["denoiser"].repo_id
            root = Path(options["hf_models_root"]) / repo.replace("/", "--")
        overrides = options.get("checkpoint_overrides")
        if overrides is None and root is not None:
            overrides = {role: str(Path(root).expanduser()) for role in ("denoiser", "conditioner", "decoder")}
        native = NativeDiffusionPipeline.from_pretrained(
            resolved_id,
            policy=RuntimePolicy(
                device=torch.device(device),
                dtype=torch.bfloat16,
                attention=parse_attention_backend(options.get("attention_backend", "auto")),
                offload=parse_offload_policy(options.get("offload_mode", "block"), allow_disk=False),
            ),
            checkpoint_overrides=overrides,
        )
        return cls(native, device=device)

    def process(self, prompt="", images=None, video=None, operator_kwargs=None, **kwargs):
        options = dict(operator_kwargs or {})
        options.update(kwargs)
        options.pop("task_name", None)
        options.pop("sample_id", None)
        distilled = "distill" in self.model_id
        options.setdefault("num_inference_steps", 10 if distilled else 50)
        options.setdefault("guidance_scale", 1.0 if distilled else 5.0)
        return self.operator.prepare(prompt, images=images, video=video, **options)

    def __call__(
        self,
        prompt="",
        images=None,
        video=None,
        output_path=None,
        output_dir=None,
        return_dict=False,
        operator_kwargs=None,
        interactions=None,
        ref_image_path=None,
        **kwargs,
    ):
        if interactions:
            raise ValueError("Kandinsky-6 does not accept action controls")
        if images is None:
            images = ref_image_path
        requested = Path(output_path) if output_path else Path(output_dir or ".") / (self.model_id + ".mp4")
        requested = requested.expanduser().resolve()
        report = requested if requested.suffix == ".json" else requested.with_suffix(".json")
        artifact = requested.with_suffix(".mp4")
        artifact.parent.mkdir(parents=True, exist_ok=True)
        request = self.process(
            prompt, images=images, video=video, operator_kwargs=operator_kwargs, save_path=str(artifact), **kwargs
        )
        started = time.monotonic()
        output = self.native_pipeline(request)
        result = {
            "status": "success",
            "model_id": self.model_id,
            "artifact_kind": "generated_video",
            "artifact_path": str(artifact),
            "artifact_files": [str(artifact)],
            "metadata_path": str(report),
            "runtime": "worldfoundry-native-diffusion",
            "elapsed_seconds": time.monotonic() - started,
            "has_audio": output.artifacts.get("audio") is not None,
            "metadata": dict(output.metadata),
        }
        report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        return result if return_dict else str(artifact)
