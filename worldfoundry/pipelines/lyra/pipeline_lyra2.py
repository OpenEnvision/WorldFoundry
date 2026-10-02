"""Lyra2 visual generation pipeline module."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import torch
from PIL import Image

from worldfoundry.base_models.three_dimensions.point_clouds.lyra.runtime_v2 import (
    Lyra2Representation,
)
from worldfoundry.core.io import save_video_frames
from worldfoundry.core.io.image import load_pil_image

from ...operators.lyra_operator import LyraOperator
from ...synthesis.visual_generation.lyra_2.synthesis import Lyra2Synthesis
from ...synthesis.visual_generation.memory.stream import VisualFrameMemory
from ..pipeline_utils import PipelineABC


class Lyra2Pipeline(PipelineABC):
    """Lyra-2 pipeline for action-conditioned navigation video and optional 3D reconstruction."""

    MODEL_ID = "lyra-2"

    def __init__(
        self,
        synthesis_model: Optional[Lyra2Synthesis] = None,
        representation_model: Optional[Lyra2Representation] = None,
        operator: Optional[LyraOperator] = None,
        memory_module: Optional[Any] = None,
        device: str = "cuda",
        # Use bfloat16 precision to balance memory efficiency and numeric range
        weight_dtype=torch.bfloat16,
    ):
        """Initialize the pipeline and configure runtime components."""
        self.synthesis_model = synthesis_model
        self.representation_model = representation_model
        self.operator = operator or LyraOperator()
        self.memory_module = memory_module or VisualFrameMemory(model_id=self.MODEL_ID)
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
    ) -> "Lyra2Pipeline":
        """Load the pipeline from pretrained checkpoints and configurations."""
        required_components = required_components or {}
        repo_root = required_components.get("repo_root") or model_path
        da3_model_name = required_components.get("da3_model_name") or kwargs.get(
            "da3_model_name",
            "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        )

        synthesis_model = Lyra2Synthesis.from_pretrained(
            pretrained_model_path=repo_root,
            device=device,
            weight_dtype=weight_dtype,
            checkpoint_dir=required_components.get("checkpoint_dir"),
            negative_prompt_path=required_components.get("negative_prompt_path"),
            da3_model_name=da3_model_name,
            da3_model_path_custom=required_components.get("da3_model_path_custom"),
            experiment=required_components.get("experiment", kwargs.get("experiment", "lyra2")),
            context_parallel_size=required_components.get(
                "context_parallel_size",
                kwargs.get("context_parallel_size", 1),
            ),
            guidance=required_components.get("guidance", kwargs.get("guidance", 5.0)),
            shift=required_components.get("shift", kwargs.get("shift", 5.0)),
            num_sampling_step=required_components.get(
                "num_sampling_step",
                kwargs.get("num_sampling_step", 35),
            ),
            prompt_suffix=required_components.get("prompt_suffix", kwargs.get("prompt_suffix", "")),
            offload=required_components.get("offload", kwargs.get("offload", False)),
            offload_when_prompt=required_components.get(
                "offload_when_prompt",
                kwargs.get("offload_when_prompt", False),
            ),
            load_runtime=required_components.get("load_runtime", kwargs.get("load_runtime", False)),
        )

        representation_model = Lyra2Representation.from_pretrained(
            pretrained_model_path=repo_root,
            device=device,
            da3_model_name=da3_model_name,
            da3_model_path_custom=required_components.get("da3_model_path_custom"),
            no_vipe=required_components.get("no_vipe", kwargs.get("no_vipe", False)),
            force=required_components.get("force", kwargs.get("force", False)),
            render_fps=required_components.get("render_fps", kwargs.get("render_fps")),
            render_chunk_size=required_components.get(
                "render_chunk_size",
                kwargs.get("render_chunk_size", 1),
            ),
            vipe_overrides=required_components.get("vipe_overrides"),
        )

        return cls(
            synthesis_model=synthesis_model,
            representation_model=representation_model,
            operator=LyraOperator(),
            memory_module=VisualFrameMemory(model_id=cls.MODEL_ID),
            device=device,
            weight_dtype=weight_dtype,
        )

    def process(
        self,
        images,
        interactions: Optional[Sequence[Union[str, Dict[str, str]]]] = None,
        prompt: str = "",
        camera_path: Optional[Dict[str, Any]] = None,
        region_hint: str = "",
    ) -> Dict[str, Any]:
        """Process and normalize input arguments and conditions for inference."""
        if self.synthesis_model is None:
            raise RuntimeError("Synthesis model is not loaded. Use from_pretrained() first.")

        image = self.operator.process_perception(images)
        if camera_path is not None:
            operator_condition = self.operator.process_camera_path(
                camera_path,
                prompt=prompt,
                region_hint=region_hint,
            )
            return {
                "image": image,
                "prompt": prompt or "",
                "region_hint": region_hint or "",
                **operator_condition,
            }
        if not interactions:
            raise ValueError("Provide interactions or a camera_path.")
        self.operator.get_interaction(interactions)
        try:
            operator_condition = self.operator.process_interaction(prompt=prompt)
        finally:
            self.operator.delete_last_interaction()

        return {
            "image": image,
            "prompt": prompt or "",
            **operator_condition,
        }

    def __call__(
        self,
        images,
        interactions: Optional[Sequence[Union[str, Dict[str, str]]]] = None,
        prompt: str = "",
        camera_path: Optional[Dict[str, Any]] = None,
        region_hint: str = "",
        fps: int = 16,
        resolution: Sequence[int] = (480, 832),
        reconstruct_3d: bool = False,
        output_dir: Optional[str] = None,
        seed: int = 1,
        return_dict: bool = False,
        show_progress: bool = True,
        **kwargs,
    ):
        """Execute the complete pipeline generation flow."""
        processed = self.process(
            images=images,
            interactions=interactions,
            prompt=prompt,
            camera_path=camera_path,
            region_hint=region_hint,
        )

        synthesis_result = self.synthesis_model.predict(
            input_image=processed["image"],
            camera_w2c=processed["camera_w2c"],
            prompt=processed["prompt"],
            chunk_captions=processed["chunk_captions"],
            zoom_factors=processed["zoom_factors"],
            fps=fps,
            seed=seed,
            resolution=resolution,
            guidance=kwargs.get("guidance"),
            shift=kwargs.get("shift"),
            num_sampling_step=kwargs.get("num_sampling_step"),
            offload=kwargs.get("offload"),
            offload_when_prompt=kwargs.get("offload_when_prompt"),
            show_progress=show_progress,
            execute=kwargs.get("execute", True),
        )

        generated_video_path = None
        reconstruction_result = None
        if output_dir:
            output_dir_path = Path(output_dir).expanduser().resolve()
            output_dir_path.mkdir(parents=True, exist_ok=True)
            generated_video_path = str(output_dir_path / "generated.mp4")
            save_video_frames(synthesis_result["video"], generated_video_path, fps=fps)

        if reconstruct_3d:
            if self.representation_model is None:
                raise RuntimeError("Representation model is not loaded. Use from_pretrained() first.")
            recon_output_dir = None
            if output_dir:
                recon_output_dir = str(Path(output_dir).expanduser().resolve() / "reconstruction")
            reconstruction_result = self.representation_model.get_representation(
                {
                    "video_path": generated_video_path,
                    "video": synthesis_result["video"] if generated_video_path is None else None,
                    "video_tensor": synthesis_result["video_tensor"],
                    "fps": fps,
                    "output_dir": recon_output_dir,
                }
            )

        result = {
            "video": synthesis_result["video"],
            "video_tensor": synthesis_result["video_tensor"],
            "generated_video_path": generated_video_path,
            "reconstruction": reconstruction_result,
            "prompt": processed["prompt"],
            "actions": processed["actions"],
            "camera_w2c": processed["camera_w2c"],
            "zoom_factors": processed["zoom_factors"],
            "chunk_captions": processed["chunk_captions"],
            "camera_path": processed.get("camera_path"),
            "region_hint": processed.get("region_hint", ""),
        }
        if reconstruct_3d or return_dict:
            return result
        return synthesis_result["video"]

    def stream(
        self,
        interactions: Optional[Sequence[Union[str, Dict[str, str]]]] = None,
        images: Optional[Union[Image.Image, Any]] = None,
        prompt: str = "",
        camera_path: Optional[Dict[str, Any]] = None,
        region_hint: str = "",
        fps: int = 16,
        resolution: Sequence[int] = (480, 832),
        reconstruct_3d: bool = False,
        output_dir: Optional[str] = None,
        seed: int = 1,
        reset_memory: bool = False,
        return_dict: bool = False,
        **kwargs,
    ):
        """Stream visual generation outputs chunk by chunk."""
        if reset_memory:
            self.memory_module.manage(action="reset")

        if images is not None:
            self.memory_module.record(load_pil_image(images), metadata={"mode": "init"})

        current_image = self.memory_module.select()
        if current_image is None:
            raise ValueError("No image in memory. Provide 'images' on the first stream turn.")

        result = self.__call__(
            images=current_image,
            interactions=interactions,
            prompt=prompt,
            camera_path=camera_path,
            region_hint=region_hint,
            fps=fps,
            resolution=resolution,
            reconstruct_3d=reconstruct_3d,
            output_dir=output_dir,
            seed=seed,
            return_dict=True,
            **kwargs,
        )
        self.memory_module.record(
            result,
            metadata={"prompt": prompt, "interactions": list(interactions)},
        )
        if reconstruct_3d or return_dict:
            return result
        return result["video"]


class LyraPipeline(Lyra2Pipeline):
    """Pipeline implementation for Lyra visual generation."""
    MODEL_ID = "lyra"
