"""StepVideo STEP1 + CLIP :class:`~...contracts.ConditionEncoder`.

Package entry for StepVideo prompt conditioning.
:class:`~.component.StepVideoPromptConditioner` runs the
STEP1 decoder-only LLM (:mod:`.step_llm`) plus Hunyuan CLIP
(:mod:`.clip`, max length 77).

``encode`` returns the concatenated / paired tensors the
StepVideo denoiser expects.  Tokenizer / BPE is
:mod:`.tokenizer`.

This is StepVideo's own stack, not Wan UMT5.
"""

from .clip import HunyuanClip
from .step_llm import STEP1TextEncoder
from .component import StepVideoPromptConditioner, build_step_video_prompt_conditioner

__all__ = [
    "HunyuanClip",
    "STEP1TextEncoder",
    "StepVideoPromptConditioner",
    "build_step_video_prompt_conditioner",
]
