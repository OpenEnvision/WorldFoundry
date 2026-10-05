"""HunyuanVideo / 1.5 condition-encoder factories.

Package entry for runner-facing prompt conditioners.
:class:`~.component.HunyuanVideoPromptConditioner` is the
original Llama+CLIP path.  :class:`HunyuanVideo15PromptConditioner`
adds LLM + ByT5 glyph + optional SigLIP vision.

``encode`` returns :class:`~...contracts.Conditioning` maps
for the matching Hunyuan denoiser.  Inner graphs live in
:mod:`.original`, :mod:`.h15_text`, :mod:`.h15_vision`.

Chinese template literals in :mod:`.constants` are ABI —
do not rewrite them.
"""

from .component import (
    HunyuanVideo15PromptConditioner,
    HunyuanVideoPromptConditioner,
    build_hunyuan_video15_prompt_conditioner,
    build_hunyuan_video_prompt_conditioner,
)

__all__ = [
    "HunyuanVideo15PromptConditioner",
    "HunyuanVideoPromptConditioner",
    "build_hunyuan_video15_prompt_conditioner",
    "build_hunyuan_video_prompt_conditioner",
]
