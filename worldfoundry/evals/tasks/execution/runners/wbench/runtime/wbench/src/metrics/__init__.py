"""
Evaluation metrics module.

Organized by 5 core dimensions:
- video_quality: Video quality (Renderer)
- interaction: Interaction adherence (Player)
- consistency: Consistency (Memory)
- physical: Physical plausibility (Engine)
- setting_adherence: Setting adherence (Director)
- vlm: VLM-based evaluation
"""


def get_vlm_evaluator():
    from .vlm import VLMEvaluator

    return VLMEvaluator


__all__ = [
    "consistency",
    "get_vlm_evaluator",
    "interaction",
    "physical",
    "setting_adherence",
    "video_quality",
    "vlm",
]
