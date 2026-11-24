# Model registration precedes scorer initialization.
from .model import MPLUGOwl2LlamaForCausalLM as MPLUGOwl2LlamaForCausalLM

# isort: split
from .evaluate import (
    QAlignAestheticScorer as QAlignAestheticScorer,
)
from .evaluate import (
    QAlignScorer as QAlignScorer,
)
from .evaluate import (
    QAlignVideoScorer as QAlignVideoScorer,
)
from .load_video import load_video as load_video
