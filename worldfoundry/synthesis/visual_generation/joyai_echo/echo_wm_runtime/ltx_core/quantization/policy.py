from dataclasses import dataclass

from ..loader.module_ops import ModuleOps
from ..loader.sd_ops import SDOps
from .fp8_cast import TRANSFORMER_LINEAR_DOWNCAST_MAP, UPCAST_DURING_INFERENCE
from .fp8_scaled_mm import FP8_PREPARE_MODULE_OPS, FP8_TRANSPOSE_SD_OPS


@dataclass(frozen=True)
class QuantizationPolicy:
    """Configuration for model quantization during loading.
    Attributes:
        sd_ops: State dict operations for weight transformation.
        module_ops: Post-load module transformations.
    """

    sd_ops: SDOps | None = None
    module_ops: tuple[ModuleOps, ...] = ()

    @classmethod
    def fp8_cast(cls) -> "QuantizationPolicy":
        """Create policy using FP8 casting with upcasting during inference."""
        return cls(
            sd_ops=TRANSFORMER_LINEAR_DOWNCAST_MAP,
            module_ops=(UPCAST_DURING_INFERENCE,),
        )

    @classmethod
    def fp8_scaled_mm(cls) -> "QuantizationPolicy":
        """Create policy using FP8 scaled matrix multiplication."""
        try:
            import tensorrt_llm  # noqa: F401
        except ImportError as e:
            raise ImportError("tensorrt_llm is not installed, skipping FP8 scaled MM quantization") from e

        return cls(
            sd_ops=FP8_TRANSPOSE_SD_OPS,
            module_ops=(FP8_PREPARE_MODULE_OPS,),
        )
