from .prompt.action import action_prompt
from .prompt.color import color_prompt
from .prompt.object_class import object_class_prompt
from .prompt.overall_consistency import overall_consistency_prompt
from .prompt.PromptTemplate4GPTeval import (
    Prompt4AestheticQuality,
    Prompt4ImagingQuality,
    Prompt4MotionEffects,
    Prompt4TemporalConsistency,
)
from .prompt.scene import scene_prompt

prompt = {
    "overall_consistency": overall_consistency_prompt,
    "color": color_prompt,
    "object_class": object_class_prompt,
    "scene": scene_prompt,
    "action": action_prompt,
    "imaging_quality": Prompt4ImagingQuality,
    "aesthetic_quality": Prompt4AestheticQuality,
    "temporal_consistency": Prompt4TemporalConsistency,
    "motion_effects": Prompt4MotionEffects,
}
