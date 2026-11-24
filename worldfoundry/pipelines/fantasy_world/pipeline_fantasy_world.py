"""Fantasy World visual generation pipeline module."""

from __future__ import annotations

import importlib
from typing import Any

import torch

from ...operators.fantasy_world_operator import FantasyWorldOperator
from ...synthesis.visual_generation.memory.stream import SceneStateMemory
from ..pipeline_utils import PipelineABC


class FantasyWorldPipelineBase(PipelineABC):
    """FantasyWorldPipelineBase implementation for WorldFoundry pipelines."""

    SYNTHESIS_CLS = None
    SYNTHESIS_TARGET = None
    DEFAULT_SCENE_NAME = "fantasyworld_scene"

    def __init__(
        self,
        *,
        operator: FantasyWorldOperator | None = None,
        synthesis_model=None,
        memory_module: Any | None = None,
        device: str = "cuda",
        # Use bfloat16 precision to balance memory efficiency and numeric range
        weight_dtype: torch.dtype = torch.bfloat16,
    ):
        """Initialize the pipeline and configure runtime components."""
        self.operator = operator or FantasyWorldOperator()
        self.synthesis_model = synthesis_model
        memory_model_id = self.MODEL_ID or "fantasyworld"
        self.memory_module = memory_module or SceneStateMemory(model_id=memory_model_id)
        self.device = device
        self.weight_dtype = weight_dtype

    @classmethod
    def _synthesis_cls(cls):
        """Synthesis cls for FantasyWorldPipelineBase."""
        if cls.SYNTHESIS_CLS is not None:
            return cls.SYNTHESIS_CLS
        if cls.SYNTHESIS_TARGET is None:
            raise RuntimeError(f"{cls.__name__} must define SYNTHESIS_CLS or SYNTHESIS_TARGET.")
        module_name, class_name = cls.SYNTHESIS_TARGET.split(":", 1)
        # Dynamically load target module to prevent eager-import overhead
        module = importlib.import_module(module_name)
        cls.SYNTHESIS_CLS = getattr(module, class_name)
        return cls.SYNTHESIS_CLS

    @classmethod
    def from_pretrained(  # ty: ignore[invalid-method-override]  # model-specific loader arguments, passed by keyword
        cls,
        model_path: Any | None = None,
        required_components: dict | None = None,
        device: str = "cuda",
        # Use bfloat16 precision to balance memory efficiency and numeric range
        weight_dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ):
        """Load the pipeline from pretrained checkpoints and configurations."""
        component_options = dict(required_components or {})
        if isinstance(model_path, dict):
            component_options.update(model_path)
            model_path = component_options.pop(
                "model_path",
                component_options.pop("pretrained_model_path", None),
            )
        kwargs = cls._strip_framework_loading_options({**component_options, **kwargs})
        synthesis_model = cls._synthesis_cls().from_pretrained(
            pretrained_model_path=model_path,
            device=device,
            weight_dtype=weight_dtype,
            **kwargs,
        )
        return cls(
            operator=FantasyWorldOperator(),
            synthesis_model=synthesis_model,
            memory_module=SceneStateMemory(model_id=cls.MODEL_ID or "fantasyworld"),
            device=device,
            weight_dtype=weight_dtype,
        )

    def _resolve_camera_source(
        self,
        *,
        interactions=None,
        camera_json_path=None,
        camera_data=None,
        camera_poses=None,
    ):
        """Resolve camera source for FantasyWorldPipelineBase."""
        camera_source = self.operator.resolve_interaction_source(
            interactions=interactions,
            camera_json_path=camera_json_path,
            camera_data=camera_data,
            camera_poses=camera_poses,
        )
        self.operator.get_interaction(camera_source)
        try:
            return self.operator.process_interaction()
        finally:
            self.operator.delete_last_interaction()

    def _build_request_state(
        self,
        *,
        images,
        end_image=None,
        interactions=None,
        camera_json_path=None,
        camera_data=None,
        camera_poses=None,
        K=None,
        scene_name=None,
        prompt: str = "",
    ):
        """Build request state for FantasyWorldPipelineBase."""
        perception = self.operator.process_perception(
            images=images,
            end_image=end_image,
            scene_name=scene_name,
        )
        camera_source = self._resolve_camera_source(
            interactions=interactions,
            camera_json_path=camera_json_path,
            camera_data=camera_data,
            camera_poses=camera_poses,
        )
        return {
            "image": perception["image"],
            "end_image": perception["end_image"],
            "camera_source": camera_source,
            "K": K,
            "scene_name": perception["scene_name"],
            "prompt": prompt or "",
        }

    def process(self, **kwargs):
        """Process and normalize input arguments and conditions for inference."""
        if self.synthesis_model is None:
            raise RuntimeError("FantasyWorld synthesis model is not loaded. Use from_pretrained() first.")
        return self._build_request_state(**kwargs)

    def _predict_from_state(
        self,
        state: dict[str, Any],
        *,
        prompt: str,
        output_dir=None,
        return_dict: bool = False,
        **kwargs,
    ):
        """Predict from state for FantasyWorldPipelineBase."""
        result = self.synthesis_model.predict(
            image=state["image"],
            end_image=state.get("end_image"),
            prompt=prompt or state.get("prompt") or "",
            camera_source=state["camera_source"],
            K=state.get("K"),
            scene_name=state.get("scene_name") or self.DEFAULT_SCENE_NAME,
            output_dir=output_dir,
            return_dict=True,
            **kwargs,
        )
        if return_dict:
            return result
        return result["frames"]

    def __call__(
        self,
        *,
        images,
        interactions=None,
        prompt: str = "",
        end_image=None,
        camera_json_path=None,
        camera_data=None,
        camera_poses=None,
        K=None,
        scene_name=None,
        output_dir=None,
        return_dict: bool = False,
        **kwargs,
    ):
        """Execute the complete pipeline generation flow."""
        state = self.process(
            images=images,
            end_image=end_image,
            interactions=interactions,
            camera_json_path=camera_json_path,
            camera_data=camera_data,
            camera_poses=camera_poses,
            K=K,
            scene_name=scene_name,
            prompt=prompt,
        )
        result = self._predict_from_state(
            state,
            prompt=prompt,
            output_dir=output_dir,
            return_dict=True,
            **kwargs,
        )
        if return_dict:
            return result
        return result["frames"]

    def stream(
        self,
        *,
        images=None,
        interactions=None,
        prompt: str = "",
        end_image=None,
        camera_json_path=None,
        camera_data=None,
        camera_poses=None,
        K=None,
        scene_name=None,
        output_dir=None,
        reset_memory: bool = False,
        return_dict: bool = False,
        **kwargs,
    ):
        """Stream visual generation outputs chunk by chunk."""
        if reset_memory:
            self.memory_module.manage(action="reset")

        state = self.memory_module.select()
        if (
            images is not None
            or end_image is not None
            or interactions is not None
            or camera_json_path is not None
            or camera_data is not None
            or camera_poses is not None
            or K is not None
            or scene_name is not None
        ):
            if state is None and images is None:
                raise ValueError("FantasyWorld stream initialization requires `images` on the first turn.")

            current_camera_source = self.operator.resolve_interaction_source(
                interactions=interactions,
                camera_json_path=camera_json_path,
                camera_data=camera_data,
                camera_poses=camera_poses,
            )
            if current_camera_source is None and state is not None:
                current_camera_source = state.get("camera_source")

            state = self._build_request_state(
                images=images if images is not None else state["image"],
                end_image=end_image if end_image is not None else (state.get("end_image") if state else None),
                interactions=current_camera_source,
                K=K if K is not None else (state.get("K") if state else None),
                scene_name=scene_name if scene_name is not None else (state.get("scene_name") if state else None),
                prompt=prompt or (state.get("prompt") if state else ""),
            )
            self.memory_module.record(
                {"kind": "request_state", "state": state},
                metadata={"type": "other"},
            )

        if state is None:
            raise ValueError(
                "No FantasyWorld stream state in memory. Provide `images` and camera trajectory on the first turn."
            )

        if K is not None:
            state["K"] = K
        if scene_name is not None:
            state["scene_name"] = scene_name
        if prompt:
            state["prompt"] = prompt

        result = self._predict_from_state(
            state,
            prompt=prompt or state.get("prompt") or "",
            output_dir=output_dir,
            return_dict=True,
            **kwargs,
        )
        self.memory_module.record(result, metadata={"type": "video"})
        if return_dict:
            return result
        return result["frames"]


class FantasyWorldWan21Pipeline(FantasyWorldPipelineBase):
    """Pipeline implementation for FantasyWorldWan21 visual generation."""

    MODEL_ID = "fantasyworld-wan21"
    SYNTHESIS_TARGET = (
        "worldfoundry.synthesis.visual_generation.fantasy_world.fantasy_world_wan21_synthesis:"
        "FantasyWorldWan21Synthesis"
    )
    DEFAULT_SCENE_NAME = "fantasyworld_wan21_scene"


class FantasyWorldWan22Pipeline(FantasyWorldPipelineBase):
    """Pipeline implementation for FantasyWorldWan22 visual generation."""

    MODEL_ID = "fantasyworld-wan22"
    SYNTHESIS_TARGET = (
        "worldfoundry.synthesis.visual_generation.fantasy_world.fantasy_world_wan22_synthesis:"
        "FantasyWorldWan22Synthesis"
    )
    DEFAULT_SCENE_NAME = "fantasyworld_wan22_scene"


class FantasyWorldPipeline(FantasyWorldWan21Pipeline):
    """Pipeline implementation for FantasyWorld visual generation."""

    MODEL_ID = "fantasyworld"


__all__ = [
    "FantasyWorldPipeline",
    "FantasyWorldPipelineBase",
    "FantasyWorldWan21Pipeline",
    "FantasyWorldWan22Pipeline",
]
