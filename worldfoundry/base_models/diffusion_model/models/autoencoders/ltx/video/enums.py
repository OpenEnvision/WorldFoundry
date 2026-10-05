"""LTX video VAE enums: log-variance, norm type, padding mode.

:class:`NormLayerType`, :class:`LogVarianceType`, and
:class:`PaddingModeType` are the tokens serialized in LTX
video checkpoint JSON.  Configurators in
:mod:`.model_configurator` map them onto CNN constructors.

No tensor math lives here.  Audio has a separate
:class:`~..audio.causality_axis.CausalityAxis`.

Shared configuration vocabulary for the LTX video VAE family.
"""

from enum import Enum


class NormLayerType(Enum):
    """Normalization kind serialized in LTX video checkpoint JSON."""
    GROUP_NORM = "group_norm"
    PIXEL_NORM = "pixel_norm"


class LogVarianceType(Enum):
    """How the video VAE parameterizes posterior log-variance."""
    PER_CHANNEL = "per_channel"
    UNIFORM = "uniform"
    CONSTANT = "constant"
    NONE = "none"


class PaddingModeType(Enum):
    """Spatial/temporal padding mode serialized in LTX video config."""
    ZEROS = "zeros"
    REFLECT = "reflect"
    REPLICATE = "replicate"
    CIRCULAR = "circular"
