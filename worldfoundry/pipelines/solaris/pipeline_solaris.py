"""Solaris visual generation pipeline module."""

from __future__ import annotations

from typing import Any

import torch

from ...synthesis.visual_generation.solaris.runtime_env import normalize_eval_types
from ...synthesis.visual_generation.solaris.solaris_synthesis import SolarisSynthesis
from ..pipeline_utils import PipelineABC


class SolarisPipeline(PipelineABC):
    """WorldFoundry wrapper for the official Solaris inference runtime."""

    def __init__(
        self,
        synthesis_model: SolarisSynthesis | None = None,
        device: str = "cuda",
        # Use bfloat16 precision to balance memory efficiency and numeric range
        weight_dtype=torch.bfloat16,
    ) -> None:
        """Initialize the pipeline and configure runtime components."""
        self.synthesis_model = synthesis_model
        self.device = device
        self.weight_dtype = weight_dtype

    @classmethod
    def from_pretrained(  # ty: ignore[invalid-method-override]  # model-specific loader arguments, passed by keyword
        cls,
        model_path: str | None = None,
        required_components: dict[str, Any] | None = None,
        device: str = "cuda",
        # Use bfloat16 precision to balance memory efficiency and numeric range
        weight_dtype=torch.bfloat16,
        **kwargs,
    ) -> SolarisPipeline:
        """Load the pipeline from pretrained checkpoints and configurations."""
        required_components = required_components or {}
        synthesis_model = SolarisSynthesis.from_pretrained(
            pretrained_model_path=model_path,
            device=device,
            runtime_root=required_components.get("runtime_root"),
            pretrained_model_dir=required_components.get("pretrained_model_dir"),
            eval_data_dir=required_components.get("eval_data_dir"),
            output_dir=required_components.get("output_dir"),
            checkpoint_dir=required_components.get("checkpoint_dir"),
            jax_cache_dir=required_components.get("jax_cache_dir"),
            model_weights_path=required_components.get("model_weights_path"),
            python_executable=required_components.get("python_executable"),
            enable_jax_cache=required_components.get("enable_jax_cache", False),
            **kwargs,
        )
        return cls(
            synthesis_model=synthesis_model,
            device=device,
            weight_dtype=weight_dtype,
        )

    def process(self, eval_types: object | None = None) -> dict[str, Any]:
        """Process and normalize input arguments and conditions for inference."""
        return {
            "eval_types": normalize_eval_types(eval_types),
        }

    def __call__(
        self,
        eval_types: object | None = None,
        *,
        experiment_name: str = "solaris_worldfoundry",
        output_dir: str | None = None,
        eval_num_samples: int | None = None,
        num_workers: int | None = None,
        num_frames_eval: int | None = None,
        enable_jax_cache: bool | None = None,
        checkpoint_dir: str | None = None,
        jax_cache_dir: str | None = None,
        model_weights_path: str | None = None,
        return_dict: bool = False,
        show_progress: bool = True,
    ):
        """Execute the complete pipeline generation flow."""
        if self.synthesis_model is None:
            raise RuntimeError("Synthesis model is not loaded. Use from_pretrained() first.")

        processed = self.process(eval_types=eval_types)
        result = self.synthesis_model.predict(
            eval_types=processed["eval_types"],
            experiment_name=experiment_name,
            output_dir=output_dir,
            eval_num_samples=eval_num_samples,
            num_workers=num_workers,
            num_frames_eval=num_frames_eval,
            enable_jax_cache=enable_jax_cache,
            checkpoint_dir=checkpoint_dir,
            jax_cache_dir=jax_cache_dir,
            model_weights_path=model_weights_path,
            return_dict=True,
            show_progress=show_progress,
        )
        if return_dict:
            return result
        return result["model_output_dir"]


__all__ = ["SolarisPipeline"]
