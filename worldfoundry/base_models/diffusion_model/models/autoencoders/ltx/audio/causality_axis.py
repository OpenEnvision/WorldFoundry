"""Which axis is causal for LTX audio convolutions.

:class:`CausalityAxis` names the time-like dimension that
must not see the future when :class:`~.causal_conv_2d.CausalConv2d`
pads.  Encoder, decoder, and vocoder blocks share this enum
so checkpoint configs stay consistent.

No tensor math lives here — it is a configuration token for
:mod:`.downsample`, :mod:`.upsample`, and :mod:`.resnet`.

Part of the LTX audio VAE family under :mod:`~..audio`.
"""

from enum import Enum


class CausalityAxis(Enum):
    """Enum for specifying the causality axis in causal convolutions."""

    NONE = None
    WIDTH = "width"
    HEIGHT = "height"
    WIDTH_COMPATIBILITY = "width-compatibility"
