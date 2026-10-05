"""Recammaster visual generation pipeline module."""

from ..pipeline_utils import PipelineABC
import torch
from typing import Any, Optional


def _recammaster_operator_cls():
    """Recammaster operator cls helper function."""
    from ...operators.recammaster_operator import ReCamMasterOperator

    return ReCamMasterOperator


def _recammaster_synthesis_cls():
    """Recammaster synthesis cls helper function."""
    from ...synthesis.visual_generation.kling.recammaster_synthesis import ReCamMasterSynthesis

    return ReCamMasterSynthesis


class ReCamMasterPipeline(PipelineABC):
    """Pipeline implementation for ReCamMaster visual generation."""
    def __init__(self,
                 operator: Optional[Any] = None,
                 synthesis_model: Optional[Any] = None,
                 device: str = "cuda",
                 # Use bfloat16 precision to balance memory efficiency and numeric range
                 weight_dtype = torch.bfloat16,):
        """Initialize the pipeline and configure runtime components."""
        self.synthesis_model = synthesis_model 
        self.operator = operator or _recammaster_operator_cls()()
        self.device = device
        self.weight_dtype = weight_dtype

    @classmethod
    def from_pretrained(cls,
                        model_path="KlingTeam/ReCamMaster-Wan2.1",
                        required_components=None,
                        device="cuda",
                        # Use bfloat16 precision to balance memory efficiency and numeric range
                        weight_dtype = torch.bfloat16,
                        **kwargs):
        """Load the pipeline from pretrained checkpoints and configurations."""
        component_options = dict(required_components or {})
        if isinstance(model_path, dict):
            component_options.update(model_path)
            model_path = component_options.pop(
                "model_path",
                component_options.pop(
                    "pretrained_model_path",
                    component_options.pop("recammaster_ckpt_path", None),
                ),
            )
        component_options = cls._strip_framework_loading_options({**component_options, **kwargs})
        wan_model_path = component_options.pop("wan_model_path", "Wan-AI/Wan2.1-T2V-1.3B")
        recammaster_ckpt_path = component_options.pop(
            "recammaster_ckpt_path",
            model_path or "KlingTeam/ReCamMaster-Wan2.1",
        )
        synthesis_model = _recammaster_synthesis_cls().from_pretrained(pretrained_model_path=wan_model_path,
                                                         recammaster_ckpt_path=recammaster_ckpt_path,
                                                         device=device,
                                                         weight_dtype=weight_dtype,
                                                         **component_options)
        operator = _recammaster_operator_cls()()
        return cls(operator, synthesis_model, device, weight_dtype)

    def process(self,
                interaction,
                video_path,
                textual_prompt):
        """Process and normalize input arguments and conditions for inference."""
        video = self.operator.process_perception(video_path).to(self.weight_dtype)

        self.operator.get_interaction(interaction, textual_prompt)
        cam_trajectory_emb = self.operator.process_interaction().to(self.weight_dtype)

        self.operator.delete_last_interaction()

        return video, cam_trajectory_emb, textual_prompt

    def __call__(self,
                 camera_trajectory,
                 video_path,
                 prompt,
                 num_frames=81,
                 num_inference_steps=50,
                 cfg_scale=5.0,
                 fps=8,
                 max_num_frames=81,
                 frame_interval=1,
                 size=(480, 832),
                 ):
        """Execute the complete pipeline generation flow."""
        height, width = size
        self.operator.max_num_frames = max_num_frames
        self.operator.frame_interval = frame_interval
        self.operator.num_frames = num_frames
        self.operator.height = height
        self.operator.width = width

        video, cam_trajectory_emb, textual_prompt = self.process(camera_trajectory,
                                                                 video_path,
                                                                 prompt)
        
        output_video = self.synthesis_model.predict(
                                            textual_prompt,
                                            video,
                                            cam_trajectory_emb,
                                            num_frames=num_frames,
                                            height=height,
                                            width=width,
                                            num_inference_steps=num_inference_steps,
                                            cfg_scale=cfg_scale)
        return output_video
