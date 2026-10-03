"""Inspatio World visual generation pipeline module."""

from __future__ import annotations

from ..pipeline_utils import PipelineABC
from typing import Any, Dict, Optional

import torch

from ...synthesis.visual_generation.inspatio_world.inspatio_world_synthesis import (
    DEFAULT_CHECKPOINT_REPO,
    DEFAULT_DA3_MODEL_REPO,
    DEFAULT_FLORENCE_MODEL_REPO,
    DEFAULT_WAN_MODEL_REPO,
    InspatioWorldSynthesis,
)


class InspatioWorldPipeline(PipelineABC):
    """InSpatio-World pipeline for video + trajectory controlled novel-view synthesis."""

    def __init__(
        self,
        synthesis_model: Optional[Any] = None,
        device: str = "cuda",
        # Use bfloat16 precision to balance memory efficiency and numeric range
        weight_dtype=torch.bfloat16,
    ) -> None:
        """Initialize the pipeline and configure runtime components."""
        self.synthesis_model = synthesis_model
        self.device = device
        self.weight_dtype = weight_dtype

    @classmethod
    def from_pretrained(
        cls,
        model_path: str = DEFAULT_CHECKPOINT_REPO,
        required_components: Optional[Dict[str, Any]] = None,
        device: str = "cuda",
        # Use bfloat16 precision to balance memory efficiency and numeric range
        weight_dtype=torch.bfloat16,
        **kwargs,
    ) -> "InspatioWorldPipeline":
        """Load the pipeline from pretrained checkpoints and configurations."""
        required_components = dict(required_components or {})
        wan_model_path = kwargs.pop(
            "wan_model_path",
            required_components.pop("wan_model_path", DEFAULT_WAN_MODEL_REPO),
        )
        da3_model_path = kwargs.pop(
            "da3_model_path",
            required_components.pop("da3_model_path", DEFAULT_DA3_MODEL_REPO),
        )
        florence_model_path = kwargs.pop(
            "florence_model_path",
            required_components.pop("florence_model_path", DEFAULT_FLORENCE_MODEL_REPO),
        )
        config_path = kwargs.pop("config_path", required_components.pop("config_path", None))
        tae_checkpoint_path = kwargs.pop(
            "tae_checkpoint_path",
            required_components.pop("tae_checkpoint_path", None),
        )
        default_traj_txt_path = kwargs.pop(
            "default_traj_txt_path",
            required_components.pop("traj_txt_path", None),
        )
        kwargs.update(required_components)
        synthesis_model = InspatioWorldSynthesis.from_pretrained(
            pretrained_model_path=model_path or DEFAULT_CHECKPOINT_REPO,
            device=device,
            wan_model_path=wan_model_path,
            da3_model_path=da3_model_path,
            florence_model_path=florence_model_path,
            config_path=config_path,
            tae_checkpoint_path=tae_checkpoint_path,
            default_traj_txt_path=default_traj_txt_path,
            **kwargs,
        )
        return cls(
            synthesis_model=synthesis_model,
            device=device,
            weight_dtype=weight_dtype,
        )

    def process(
        self,
        videos=None,
        traj_txt_path: Optional[str] = None,
        prompt: str = "",
    ) -> Dict[str, Any]:
        """Process and normalize input arguments and conditions for inference."""
        if videos is None:
            raise ValueError("InSpatio-World requires `videos` as input.")
        return {
            "videos": videos,
            "traj_txt_path": traj_txt_path,
            "prompt": prompt or "",
        }

    def __call__(
        self,
        videos=None,
        traj_txt_path: Optional[str] = None,
        prompt: str = "",
        output_dir: Optional[str] = None,
        return_dict: bool = False,
        seed: Optional[int] = None,
        deterministic: Optional[bool] = None,
        **kwargs,
    ):
        """Execute the complete pipeline generation flow."""
        if self.synthesis_model is None:
            raise RuntimeError("Synthesis model is not loaded. Use from_pretrained() first.")

        processed = self.process(
            videos=videos,
            traj_txt_path=traj_txt_path,
            prompt=prompt,
        )
        result = self.synthesis_model.predict(
            visual_input=processed["videos"],
            traj_txt_path=processed["traj_txt_path"],
            prompt=processed["prompt"],
            output_root=output_dir,
            return_dict=True,
            seed=seed,
            deterministic=deterministic,
            **kwargs,
        )
        if return_dict:
            return result
        return result["video"] if result.get("video") is not None else result["videos"]
