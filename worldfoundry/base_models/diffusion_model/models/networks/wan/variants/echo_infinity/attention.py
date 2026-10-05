# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Echo-Infinity attention re-exports (shared WorldFoundry varlen kernels).

Echo-Infinity is a causal Wan 2.x memory variant.  Its transformer files
import ``flash_attention`` / ``attention`` from this module so the
historical relative path keeps working.  The kernels themselves live in
``worldfoundry.core.attention.sequence.varlen``.

No extra action, camera, VACE, linear-attention, or TeaCache logic.
"""

from worldfoundry.core.attention.sequence.varlen import attention, flash_attention

__all__ = ["attention", "flash_attention"]
