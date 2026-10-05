"""Research Wan DiT forks layered on the native or official Wan graphs.

Each submodule is a checkpoint-shaped variant.  Typical additions:

- Action: :mod:`.action_21`, :mod:`.action_22`, :mod:`.scope_action`,
  :mod:`.multiworld`, DreamZero / MinWM action encoders.
- Camera: :mod:`.camera_21`, :mod:`.dualcamctrl`, :mod:`.fantasy_world`,
  :mod:`.geometry`.
- VACE / dual control: :mod:`.spatia`, :mod:`.dual_control`, :mod:`.neoverse`.
- Causal / memory: :mod:`.abot_world`, :mod:`.forcing`, :mod:`.echo_infinity`,
  :mod:`.lingbot`, :mod:`.magic_world`, :mod:`.moverse`, :mod:`.dreamx_world`.
- Linear attention / TeaCache-friendly processors: :mod:`.linear`,
  :mod:`.linear_attention`, :mod:`.sage_attention`.
- Other recipes: :mod:`.s2v`, :mod:`.pusa`, :mod:`.sama`, :mod:`.anyflow`,
  :mod:`.versecrafter`, :mod:`.video_x_fun`.

Import the concrete class from the submodule; this package re-exports
only the variants that native recipes construct by name.
"""

from .dual_control import WanDualControlModel, WanModelDualControl
from .dualcamctrl import WanControlNet
from .neoverse import NeoVerseControlBranch, NeoVerseControlBranchDictConverter
from .s2v import WanS2VModel, WanS2VModelStateDictConverter, rope_precompute
from .scope_action import ScopeActionBlock, ScopeActionModule, ScopeActionWanModel
from .pusa import PusaWanModel
from .sama import SamaWanModel, SemanticDiffusionHead, SigLIPFeatureProjection
from .spatia import SpatiaWanModel
from .multiworld import ItTakesTwoActionEncoder, MultiWorldDiTBlock, MultiWorldWanModel
from .fantasy_world import FantasyWorldCameraCondition, FantasyWorldFusionModel, IRGBlock

__all__ = [
    "WanDualControlModel",
    "WanModelDualControl",
    "WanControlNet",
    "NeoVerseControlBranch",
    "NeoVerseControlBranchDictConverter",
    "WanS2VModel",
    "WanS2VModelStateDictConverter",
    "rope_precompute",
    "ScopeActionBlock",
    "ScopeActionModule",
    "ScopeActionWanModel",
    "PusaWanModel",
    "SamaWanModel",
    "SemanticDiffusionHead",
    "SigLIPFeatureProjection",
    "SpatiaWanModel",
    "ItTakesTwoActionEncoder",
    "MultiWorldDiTBlock",
    "MultiWorldWanModel",
    "FantasyWorldCameraCondition",
    "FantasyWorldFusionModel",
    "IRGBlock",
]
