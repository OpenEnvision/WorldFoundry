"""EfficientViT / DC-AE neural-net primitives (norm, act, ops).

Subpackage entry for the building blocks imported by
:mod:`~..efficientvit.dc_ae` and
:mod:`~..efficientvit.dc_ae_with_temporal`.

Includes 2-D ops (:mod:`.ops`), 3-D ops (:mod:`.ops_3d`),
normalization (:mod:`.norm`), activations (:mod:`.act`),
and drop-path (:mod:`.drop`).

No encode/decode entry points live here.
"""

from .act import *
from .norm import *
from .ops import *
