"""Multiworld Ittakestwo visual generation pipeline module."""

from __future__ import annotations

from ..pipeline_utils import PipelineABC
from typing import Any, Dict, Optional

import torch

from ...synthesis.visual_generation.multiworld.multiworld_ittakestwo_synthesis import (
    MultiWorldItTakesTwoSynthesis,
)
from ...synthesis.visual_generation.multiworld.ittakestwo_runtime import (
    load_ittakestwo_action_csv,
)


class MultiWorldItTakesTwoPipeline(PipelineABC):
    """WorldFoundry wrapper for the MultiWorld ItTakesTwo multi-agent runtime."""

    def __init__(
        self,
        synthesis_model: Optional[MultiWorldItTakesTwoSynthesis] = None,
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
        model_path: Optional[str] = None,
        required_components: Optional[Dict[str, Any]] = None,
        device: str = "cuda",
        # Use bfloat16 precision to balance memory efficiency and numeric range
        weight_dtype=torch.bfloat16,
        **kwargs,
    ) -> "MultiWorldItTakesTwoPipeline":
        """Load the pipeline from pretrained checkpoints and configurations."""
        required_components = required_components or {}
        synthesis_model = MultiWorldItTakesTwoSynthesis.from_pretrained(
            pretrained_model_path=model_path,
            device=device,
            runtime_root=required_components.get("runtime_root"),
            config_path=required_components.get("config_path"),
            checkpoint_path=required_components.get("checkpoint_path") or model_path,
            base_model_root=required_components.get("base_model_root"),
            vggt_root=required_components.get("vggt_root"),
            python_executable=required_components.get("python_executable"),
            derive_env_obv_from_image=required_components.get("derive_env_obv_from_image", True),
            weight_dtype=weight_dtype,
            **kwargs,
        )
        return cls(
            synthesis_model=synthesis_model,
            device=device,
            weight_dtype=weight_dtype,
        )

    def process(
        self,
        images,
        action: Optional[Dict[str, Any]] = None,
        action_path: Optional[str] = None,
        num_frames: int = 81,
        env_obv: Any = None,
    ) -> Dict[str, Any]:
        """Process and normalize input arguments and conditions for inference."""
        if images is None:
            raise ValueError("MultiWorld ItTakesTwo requires `images` as the first-frame input.")
        if action is None and action_path:
            action = load_ittakestwo_action_csv(action_path, num_frames=int(num_frames))
        if not isinstance(action, dict) or not action:
            raise ValueError("MultiWorld ItTakesTwo requires a non-empty `action` dict or action_path.")
        return {
            "image": images,
            "action": action,
            "env_obv": env_obv,
        }

    def __call__(
        self,
        images,
        action: Optional[Dict[str, Any]] = None,
        action_path: Optional[str] = None,
        env_obv: Any = None,
        output_dir: Optional[str] = None,
        save_name: str = "multiworld_ittakestwo",
        return_dict: bool = False,
        **kwargs,
    ):
        """Execute the complete pipeline generation flow."""
        if self.synthesis_model is None:
            raise RuntimeError("Synthesis model is not loaded. Use from_pretrained() first.")

        runtime_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key
            in {
                "num_frames",
                "height",
                "width",
                "fps",
                "num_inference_steps",
                "inference_seed",
                "derive_env_obv_from_image",
                "show_progress",
            }
        }
        num_frames = kwargs.get("num_frames", 81)
        processed = self.process(
            images=images,
            action=action,
            action_path=action_path,
            num_frames=int(num_frames),
            env_obv=env_obv,
        )
        result = self.synthesis_model.predict(
            image=processed["image"],
            action=processed["action"],
            env_obv=processed["env_obv"],
            output_dir=output_dir,
            save_name=save_name,
            return_dict=True,
            **runtime_kwargs,
        )
        if return_dict:
            return result
        return result["video"]


__all__ = ["MultiWorldItTakesTwoPipeline"]
