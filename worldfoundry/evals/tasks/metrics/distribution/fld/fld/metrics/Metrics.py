from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .AuthPct import AuthPct
    from .CTTest import CTTest
    from .FID import FID
    from .FLD import FLD
    from .KID import KID
    from .PrecisionRecall import PrecisionRecall
else:
    from fld.metrics.AuthPct import AuthPct
    from fld.metrics.CTTest import CTTest
    from fld.metrics.FID import FID
    from fld.metrics.FLD import FLD
    from fld.metrics.KID import KID
    from fld.metrics.PrecisionRecall import PrecisionRecall

# For testing
ALL_METRICS = [
    AuthPct(),
    CTTest(),
    FID("train"),
    FID("test"),
    FLD("train"),
    FLD("test"),
    KID("train"),
    KID("test"),
    PrecisionRecall("Precision"),
    PrecisionRecall("Recall"),
]
ALL_METRICS_DICT = {metric.name: metric for metric in ALL_METRICS}
