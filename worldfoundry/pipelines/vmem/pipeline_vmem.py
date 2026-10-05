"""Vmem visual generation pipeline module."""

from __future__ import annotations

from ..pipeline_utils import PipelineABC
from typing import Any, Optional, Sequence

import torch

from ...synthesis.visual_generation.memory.stream import VisualFrameMemory
from ...operators.vmem_operator import VMemOperator
from ...synthesis.visual_generation.vmem.vmem_synthesis import (
    DEFAULT_VMEM_REPO,
    DEFAULT_VMEM_SURFEL_REPO,
    VMemSynthesis,
)


class VMemPipeline(PipelineABC):
    """WorldFoundry wrapper for the official VMem interactive navigation runtime."""

    def __init__(
        self,
        operator: Optional[VMemOperator] = None,
        synthesis_model: Optional[VMemSynthesis] = None,
        memory_module: Optional[Any] = None,
        device: str = "cuda",
        weight_dtype: torch.dtype = torch.float32,
    ):
        """Initialize the pipeline and configure runtime components."""
        self.operator = operator or VMemOperator()
        self.synthesis_model = synthesis_model
        self.memory_module = memory_module or VisualFrameMemory(model_id="vmem")
        self.device = device
        self.weight_dtype = weight_dtype

    @classmethod
    def from_pretrained(
        cls,
        model_path: Optional[str] = None,
        required_components: Optional[dict] = None,
        device: str = "cuda",
        weight_dtype: torch.dtype = torch.float32,
        **kwargs,
    ) -> "VMemPipeline":
        """Load the pipeline from pretrained checkpoints and configurations."""
        component_options = {**(required_components or {}), **kwargs}
        synthesis_model = VMemSynthesis.from_pretrained(
            pretrained_model_path=model_path or DEFAULT_VMEM_REPO,
            surfel_model_path=component_options.pop(
                "surfel_model_path",
                DEFAULT_VMEM_SURFEL_REPO,
            ),
            config_path=component_options.pop("config_path", None),
            runtime_root=component_options.pop("runtime_root", None),
            visualization_dir=component_options.pop("visualization_dir", None),
            device=device,
            weight_dtype=weight_dtype,
            **component_options,
        )
        return cls(
            operator=VMemOperator(),
            synthesis_model=synthesis_model,
            memory_module=VisualFrameMemory(model_id="vmem"),
            device=device,
            weight_dtype=weight_dtype,
        )

    def process(self, images, interactions: Sequence[str], prompt: str = ""):
        """Process and normalize input arguments and conditions for inference."""
        if self.synthesis_model is None:
            raise RuntimeError("Synthesis model is not loaded. Use from_pretrained() first.")

        perception = self.operator.process_perception(images)
        self.operator.get_interaction(interactions)
        try:
            interaction = self.operator.process_interaction()
        finally:
            self.operator.delete_last_interaction()

        return {
            "image": perception["image"],
            "actions": interaction["actions"],
            "commands": interaction["commands"],
            "prompt": prompt or "",
        }

    def __call__(
        self,
        images,
        interactions: Sequence[str],
        prompt: str = "",
        reset_state: bool = True,
        return_dict: bool = False,
        **kwargs,
    ):
        """Execute the complete pipeline generation flow."""
        if not interactions:
            raise ValueError("interactions must be provided for VMem generation.")

        processed = self.process(images=images, interactions=interactions, prompt=prompt)
        result = self.synthesis_model.predict(
            image=processed["image"],
            actions=processed["commands"],
            reset_state=reset_state,
            return_dict=True,
            **kwargs,
        )
        result.update(
            {
                "actions": processed["actions"],
                "commands": processed["commands"],
                "prompt": processed["prompt"],
            }
        )
        if return_dict:
            return result
        return result["video"]

    def stream(
        self,
        images: Optional[Any],
        interactions: Sequence[str],
        prompt: str = "",
        reset_memory: bool = False,
        return_dict: bool = False,
        **kwargs,
    ):
        """Stream visual generation outputs chunk by chunk."""
        if reset_memory:
            self.memory_module.manage(action="reset")
            self.synthesis_model.reset()

        if images is not None:
            result = self.__call__(
                images=images,
                interactions=interactions,
                prompt=prompt,
                reset_state=True,
                return_dict=True,
                **kwargs,
            )
        else:
            if not self.synthesis_model.is_initialized():
                raise ValueError("No active VMem state. Provide 'images' on the first stream turn.")

            self.operator.get_interaction(interactions)
            try:
                interaction = self.operator.process_interaction()
            finally:
                self.operator.delete_last_interaction()

            result = self.synthesis_model.predict(
                image=None,
                actions=interaction["commands"],
                reset_state=False,
                return_dict=True,
                **kwargs,
            )
            result.update(
                {
                    "actions": interaction["actions"],
                    "commands": interaction["commands"],
                    "prompt": prompt or "",
                }
            )

        self.memory_module.record(
            result,
            metadata={"prompt": prompt, "interactions": list(interactions)},
        )
        if return_dict:
            return result
        return result["video"]


__all__ = ["VMemPipeline"]
