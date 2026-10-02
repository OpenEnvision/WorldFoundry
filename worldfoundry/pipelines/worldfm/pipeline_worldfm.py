"""Worldfm visual generation pipeline module."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import torch

from worldfoundry.base_models.three_dimensions.point_clouds.worldfm.runtime import (
    DEFAULT_WORLDFM_MOGE2_REPO,
    WorldFMRepresentation,
)

from ...operators.worldfm_operator import WorldFMOperator
from ...synthesis.visual_generation.memory.stream import SceneStateMemory
from ...synthesis.visual_generation.worldfm.worldfm_synthesis import (
    DEFAULT_WORLDFM_REPO,
    WorldFMSynthesis,
)
from ..pipeline_utils import PipelineABC


class WorldFMPipeline(PipelineABC):
    """WorldFM pipeline for image-conditioned novel-view generation with official MoGe-2."""

    def __init__(
        self,
        operator: Optional[WorldFMOperator] = None,
        representation_model: Optional[WorldFMRepresentation] = None,
        synthesis_model: Optional[WorldFMSynthesis] = None,
        memory_module: Optional[Any] = None,
        device: str = "cuda",
        weight_dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Initialize the pipeline and configure runtime components."""
        self.operator = operator or WorldFMOperator()
        self.representation_model = representation_model
        self.synthesis_model = synthesis_model
        self.memory_module = memory_module or SceneStateMemory(model_id="worldfm")
        self.device = device
        self.weight_dtype = weight_dtype

    @classmethod
    def from_pretrained(
        cls,
        model_path: str = DEFAULT_WORLDFM_REPO,
        required_components: Optional[Dict[str, Any]] = None,
        device: str = "cuda",
        weight_dtype: Optional[torch.dtype] = None,
        **kwargs,
    ) -> "WorldFMPipeline":
        """Load the pipeline from pretrained checkpoints and configurations."""
        required_components = required_components or {}

        representation_model = WorldFMRepresentation.from_pretrained(
            device=device,
            hw_path=required_components.get("hw_path"),
            moge_path=required_components.get("moge_path"),
            realesrgan_path=required_components.get("realesrgan_path"),
            zim_path=required_components.get("zim_path"),
            moge_pretrained=required_components.get("moge_pretrained", DEFAULT_WORLDFM_MOGE2_REPO),
            render_size=required_components.get("render_size", 512),
            resolution_level=required_components.get("resolution_level", 30),
            fov_deg=required_components.get("fov_deg", 45.0),
            num_views=required_components.get("num_views", 42),
            merge_max_width=required_components.get("merge_max_width", 4096),
            merge_max_height=required_components.get("merge_max_height", 2048),
            batch_size=required_components.get("batch_size", 4),
            panogen_seed=required_components.get("panogen_seed", 42),
            panogen_fp8_attention=required_components.get("panogen_fp8_attention", False),
            panogen_fp8_gemm=required_components.get("panogen_fp8_gemm", False),
            panogen_cache=required_components.get("panogen_cache", False),
            sample_grid=required_components.get("sample_grid", 10),
            center_grid=required_components.get("center_grid", 15),
            center_frac=required_components.get("center_frac", 0.5),
            eps_rel=required_components.get("eps_rel", 0.02),
            eps_abs=required_components.get("eps_abs", 0.0),
            px_radius=required_components.get("px_radius", 0),
            max_view_angle_deg=required_components.get("max_view_angle_deg", 180.0),
            use_distance_weight=required_components.get("use_distance_weight", True),
            dist_min_m=required_components.get("dist_min_m", 1.0),
            dist_max_m=required_components.get("dist_max_m", 20.0),
            weight_near=required_components.get("weight_near", 1.0),
            weight_far=required_components.get("weight_far", 0.0),
        )

        synthesis_model = WorldFMSynthesis.from_pretrained(
            pretrained_model_path=model_path or DEFAULT_WORLDFM_REPO,
            device=device,
            vae_path=required_components.get("vae_path"),
            checkpoint_filename=required_components.get("checkpoint_filename"),
            step=required_components.get("step", 2),
            image_size=required_components.get("image_size", 512),
            version=required_components.get("version", "sigma"),
            cfg_scale=required_components.get("cfg_scale", 4.5),
            weight_dtype=weight_dtype,
            **kwargs,
        )

        return cls(
            operator=WorldFMOperator(),
            representation_model=representation_model,
            synthesis_model=synthesis_model,
            memory_module=SceneStateMemory(model_id="worldfm"),
            device=device,
            weight_dtype=weight_dtype,
        )

    def _resolve_interactions(self, interactions, default_interactions):
        """Resolve interactions for WorldFMPipeline."""
        active_interactions = interactions if interactions is not None else default_interactions
        if active_interactions is None:
            raise ValueError("WorldFM requires `interactions` or a meta file containing `c2w`.")
        self.operator.get_interaction(active_interactions)
        try:
            return self.operator.process_interaction()
        finally:
            self.operator.delete_last_interaction()

    def _build_scene_context(
        self,
        *,
        images=None,
        K=None,
        meta_path=None,
        scene_name=None,
        panorama_image=None,
        panorama_path=None,
        output_dir=None,
    ):
        """Build scene context for WorldFMPipeline."""
        perception = self.operator.process_perception(
            images=images,
            K=K,
            meta_path=meta_path,
            scene_name=scene_name,
            panorama_image=panorama_image,
            panorama_path=panorama_path,
        )
        scene_context = self.representation_model.build_scene_context(
            images=perception["images"],
            image_path=perception["image_path"],
            panorama_image=perception["panorama_image"],
            panorama_path=perception["panorama_path"],
            K=perception["K"],
            scene_name=perception["scene_name"],
            output_dir=output_dir,
        )
        return perception, scene_context

    def process(
        self,
        images=None,
        interactions: Optional[Sequence] = None,
        K=None,
        prompt: str = "",
        meta_path=None,
        scene_name: Optional[str] = None,
        panorama_image=None,
        panorama_path=None,
        output_dir: Optional[str] = None,
        scene_context: Optional[Dict[str, Any]] = None,
    ):
        """Process and normalize input arguments and conditions for inference."""
        if self.representation_model is None or self.synthesis_model is None:
            raise RuntimeError("WorldFM models are not loaded. Use from_pretrained() first.")

        if scene_context is None:
            perception, scene_context = self._build_scene_context(
                images=images,
                K=K,
                meta_path=meta_path,
                scene_name=scene_name,
                panorama_image=panorama_image,
                panorama_path=panorama_path,
                output_dir=output_dir,
            )
        else:
            perception = {
                "scene_name": scene_context.get("scene_name", scene_name or "worldfm_scene"),
                "default_interactions": None,
            }

        c2w_list = self._resolve_interactions(interactions, perception.get("default_interactions"))
        representation = self.representation_model.get_representation(
            {
                "scene_context": scene_context,
                "c2w_list": c2w_list,
            }
        )
        representation["prompt"] = prompt or ""
        representation["output_dir"] = output_dir
        return representation

    def __call__(
        self,
        images=None,
        interactions: Optional[Sequence] = None,
        K=None,
        prompt: str = "",
        meta_path=None,
        scene_name: Optional[str] = None,
        panorama_image=None,
        panorama_path=None,
        output_dir: Optional[str] = None,
        save_mode: str = "video",
        fps: int = 30,
        return_dict: bool = False,
        scene_context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Execute the complete pipeline generation flow."""
        processed = self.process(
            images=images,
            interactions=interactions,
            K=K,
            prompt=prompt,
            meta_path=meta_path,
            scene_name=scene_name,
            panorama_image=panorama_image,
            panorama_path=panorama_path,
            output_dir=output_dir,
            scene_context=scene_context,
        )
        result = self.synthesis_model.predict(
            frame_conditions=processed["rendered_conditions"],
            output_dir=output_dir,
            scene_name=processed["scene_name"],
            save_mode=save_mode,
            fps=fps,
            return_dict=True,
            **kwargs,
        )
        result.update(
            {
                "scene_context": processed["scene_context"],
                "K": processed["K"],
                "c2w_list": processed["c2w_list"],
            }
        )
        if return_dict:
            return result
        return result["frames"]

    def stream(
        self,
        images=None,
        interactions: Optional[Sequence] = None,
        K=None,
        prompt: str = "",
        meta_path=None,
        scene_name: Optional[str] = None,
        panorama_image=None,
        panorama_path=None,
        output_dir: Optional[str] = None,
        save_mode: str = "video",
        fps: int = 30,
        reset_memory: bool = False,
        return_dict: bool = False,
        **kwargs,
    ):
        """Stream visual generation outputs chunk by chunk."""
        if reset_memory:
            self.memory_module.manage(action="reset")

        default_interactions = None
        scene_context = self.memory_module.select()

        if images is not None or meta_path is not None or panorama_image is not None or panorama_path is not None:
            perception, scene_context = self._build_scene_context(
                images=images,
                K=K,
                meta_path=meta_path,
                scene_name=scene_name,
                panorama_image=panorama_image,
                panorama_path=panorama_path,
                output_dir=output_dir,
            )
            default_interactions = perception.get("default_interactions")
            self.memory_module.record(
                {"scene_context": scene_context},
                metadata={"type": "scene_context", "scene_name": scene_context["scene_name"]},
            )

        if scene_context is None:
            raise ValueError("No WorldFM scene context is cached. Provide `images`/`meta_path` on the first turn.")

        active_interactions = interactions if interactions is not None else default_interactions
        if active_interactions is None:
            raise ValueError("WorldFM stream requires target camera poses for each turn.")

        result = self.__call__(
            images=None,
            interactions=active_interactions,
            K=scene_context["K"],
            prompt=prompt,
            scene_name=scene_context["scene_name"],
            output_dir=output_dir,
            save_mode=save_mode,
            fps=fps,
            return_dict=True,
            scene_context=scene_context,
            **kwargs,
        )
        self.memory_module.record(
            result,
            metadata={"type": "generation", "scene_name": scene_context["scene_name"]},
        )
        if return_dict:
            return result
        return result["frames"]
