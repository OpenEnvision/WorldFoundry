"""Lyra1 visual generation pipeline module."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import torch

from worldfoundry.base_models.three_dimensions.point_clouds.lyra.runtime_v1 import (
    Lyra1Representation,
)
from worldfoundry.base_models.three_dimensions.point_clouds.lyra.utils import (
    resolve_lyra1_repo_root,
)
from worldfoundry.core.io.paths import scratch_directory

from ...operators.lyra1_operator import Lyra1Operator
from ...synthesis.visual_generation.lyra_1.synthesis import Lyra1Synthesis
from ...synthesis.visual_generation.memory.stream import VisualFrameMemory
from ..pipeline_utils import PipelineABC
from .checkpoints import prepare_lyra1_checkpoint_root


class Lyra1Pipeline(PipelineABC):
    """Lyra-1 pipeline supporting static image or dynamic video reconstruction."""
    MODEL_ID = "lyra-1"

    def __init__(
        self,
        synthesis_model: Optional[Lyra1Synthesis] = None,
        representation_model: Optional[Lyra1Representation] = None,
        operator: Optional[Lyra1Operator] = None,
        memory_module: Optional[Any] = None,
        device: str = "cuda",
        # Use bfloat16 precision to balance memory efficiency and numeric range
        weight_dtype=torch.bfloat16,
        default_mode: str = "static",
    ):
        """Initialize the pipeline and configure runtime components."""
        self.synthesis_model = synthesis_model
        self.representation_model = representation_model
        self.operator = operator or Lyra1Operator()
        self.memory_module = memory_module or VisualFrameMemory(model_id="lyra-1")
        self.device = device
        self.weight_dtype = weight_dtype
        self.default_mode = str(default_mode).lower()

    @classmethod
    def from_pretrained(
        cls,
        model_path: Optional[str] = None,
        required_components: Optional[Dict[str, Any]] = None,
        device: str = "cuda",
        # Use bfloat16 precision to balance memory efficiency and numeric range
        weight_dtype=torch.bfloat16,
        **kwargs,
    ) -> "Lyra1Pipeline":
        """Load the pipeline from pretrained checkpoints and configurations."""
        required_components = required_components or {}
        repo_root = required_components.get("repo_root") or model_path
        default_mode = str(
            required_components.get("default_mode", kwargs.get("default_mode", "static"))
        ).lower()

        synthesis_model = Lyra1Synthesis.from_pretrained(
            pretrained_model_path=repo_root,
            device=device,
            checkpoint_dir=required_components.get("checkpoint_dir"),
            default_mode=default_mode,
            **{
                key: value
                for key, value in {**required_components, **kwargs}.items()
                if key not in {"repo_root", "checkpoint_dir", "default_mode"}
            },
        )
        representation_keys = {
            "static_ckpt_path",
            "dynamic_ckpt_path",
            "static_config_path",
            "dynamic_config_path",
            "inference_config_path",
        }
        load_representation = bool(
            required_components.get("load_representation")
            or kwargs.get("load_representation")
            or any(required_components.get(key) for key in representation_keys)
        )
        representation_model = None
        if load_representation:
            representation_model = Lyra1Representation.from_pretrained(
                pretrained_model_path=repo_root,
                checkpoint_dir=prepare_lyra1_checkpoint_root(
                    checkpoint_dir=required_components.get("checkpoint_dir") or synthesis_model.checkpoint_dir,
                    repo_root=resolve_lyra1_repo_root(repo_root),
                ),
                device=device,
                static_ckpt_path=required_components.get("static_ckpt_path"),
                dynamic_ckpt_path=required_components.get("dynamic_ckpt_path"),
                static_config_path=required_components.get("static_config_path"),
                dynamic_config_path=required_components.get("dynamic_config_path"),
                inference_config_path=required_components.get("inference_config_path"),
            )
        return cls(
            synthesis_model=synthesis_model,
            representation_model=representation_model,
            operator=Lyra1Operator(),
            memory_module=VisualFrameMemory(model_id="lyra-1"),
            device=device,
            weight_dtype=weight_dtype,
            default_mode=default_mode,
        )

    def process(
        self,
        images=None,
        videos=None,
        interactions: Optional[Sequence[Union[str, Dict[str, Any]]]] = None,
        prompt: str = "",
        mode: Optional[str] = None,
        trajectory: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process and normalize input arguments and conditions for inference."""
        mode = str(mode or self.default_mode).lower()
        if trajectory is not None and interactions is None:
            interactions = [{"trajectory": trajectory, "caption": prompt or ""}]
        if interactions is None:
            raise ValueError("Lyra-1 expects `interactions` or an explicit `trajectory`.")

        visual_input = self.operator.process_perception(
            images=images,
            videos=videos,
            mode=mode,
        )
        self.operator.get_interaction(interactions)
        try:
            operator_condition = self.operator.process_interaction(prompt=prompt)
        finally:
            self.operator.delete_last_interaction()

        return {
            "mode": mode,
            "visual_input": visual_input,
            "prompt": prompt or "",
            **operator_condition,
        }

    def __call__(
        self,
        images=None,
        videos=None,
        interactions: Optional[Sequence[Union[str, Dict[str, Any]]]] = None,
        prompt: str = "",
        mode: Optional[str] = None,
        trajectory: Optional[str] = None,
        reconstruct_3d: bool = False,
        output_dir: Optional[str] = None,
        return_dict: bool = False,
        **kwargs,
    ):
        """Execute the complete pipeline generation flow."""
        if self.synthesis_model is None:
            raise RuntimeError("Synthesis model is not loaded. Use from_pretrained() first.")

        processed = self.process(
            images=images,
            videos=videos,
            interactions=interactions,
            prompt=prompt,
            mode=mode,
            trajectory=trajectory,
        )

        mode = processed["mode"]
        output_root = Path(output_dir).expanduser().resolve() if output_dir else None
        if output_root is not None:
            output_root.mkdir(parents=True, exist_ok=True)
            generated_root = output_root / "generated"
            reconstruction_root = output_root / "reconstruction"
        else:
            generated_root = scratch_directory(f"lyra1_{mode}_generated_")
            reconstruction_root = scratch_directory(f"lyra1_{mode}_recon_")

        synthesis_kwargs = dict(kwargs)
        multi_trajectory = bool(synthesis_kwargs.pop("multi_trajectory", reconstruct_3d))
        synthesis_result = self.synthesis_model.predict(
            visual_input=processed["visual_input"],
            mode=mode,
            prompt=processed["prompt"],
            trajectory=processed["trajectory"],
            output_root=str(generated_root),
            multi_trajectory=multi_trajectory,
            **synthesis_kwargs,
        )
        if synthesis_result.get("status") == "planned":
            return {
                **synthesis_result,
                "mode": mode,
                "prompt": processed["prompt"],
                "actions": processed["actions"],
                "mapped_trajectories": processed["mapped_trajectories"],
                "trajectory": processed["trajectory"],
            }

        reconstruction_result = None
        if reconstruct_3d:
            if self.representation_model is None:
                raise RuntimeError("Representation model is not loaded. Use from_pretrained() first.")
            representation_input = {
                "generated_root": synthesis_result["generated_root"],
                "mode": mode,
                "output_dir": str(reconstruction_root),
                **kwargs,
            }
            reconstruction_result = self.representation_model.get_representation(
                representation_input
            )

        result = {
            "mode": mode,
            "prompt": processed["prompt"],
            "actions": processed["actions"],
            "mapped_trajectories": processed["mapped_trajectories"],
            "trajectory": processed["trajectory"],
            "generated_root": synthesis_result["generated_root"],
            "generated_video_path": synthesis_result["generated_video_path"],
            "video": synthesis_result["video"],
            "fps": synthesis_result["fps"],
            "reconstruction": reconstruction_result,
            "input_image": processed["visual_input"] if mode == "static" else None,
            "input_video": processed["visual_input"] if mode == "dynamic" else None,
        }
        if reconstruct_3d or return_dict:
            return result
        return synthesis_result["video"]

    def stream(
        self,
        interactions: Sequence[Union[str, Dict[str, Any]]],
        images=None,
        videos=None,
        prompt: str = "",
        mode: Optional[str] = None,
        trajectory: Optional[str] = None,
        reconstruct_3d: bool = False,
        output_dir: Optional[str] = None,
        reset_memory: bool = False,
        return_dict: bool = False,
        **kwargs,
    ):
        """Stream visual generation outputs chunk by chunk."""
        mode = str(mode or self.default_mode).lower()
        if reset_memory:
            self.memory_module.manage(action="reset")

        if images is not None:
            self.memory_module.record(images, metadata={"mode": "static"})
        if videos is not None:
            self.memory_module.record(videos, metadata={"mode": "dynamic"})

        current_visual = self.memory_module.select(mode=mode)
        if current_visual is None:
            raise ValueError(
                "No input found in memory. Provide `images` for static mode or `videos` for dynamic mode on the first turn."
            )

        result = self.__call__(
            images=current_visual if mode == "static" else None,
            videos=current_visual if mode == "dynamic" else None,
            interactions=interactions,
            prompt=prompt,
            mode=mode,
            trajectory=trajectory,
            reconstruct_3d=reconstruct_3d,
            output_dir=output_dir,
            return_dict=True,
            **kwargs,
        )
        self.memory_module.record(
            result,
            metadata={
                "mode": mode,
                "prompt": prompt,
                "interactions": list(interactions),
            },
        )
        if reconstruct_3d or return_dict:
            return result
        return result["video"]
