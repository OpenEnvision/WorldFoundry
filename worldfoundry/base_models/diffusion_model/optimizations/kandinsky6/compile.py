"""Kandinsky-6 compile configuration values."""
from typing import Any, Literal
from pydantic import Field
from .base import ConfigModel
CompileStrategyName = Literal["none", "torch"]

class AotiCompileConfig(ConfigModel):
    """Regional AOTInductor knobs for ``kandy export`` / ``compile_aoti``.

    Defaults match ``torch.compile(mode="max-autotune-no-cudagraphs")`` plus
    packaging with weights outside the SO. Prefer ``cudagraphs=false`` with
    MagCache + FA3 (graphs break on skip / custom-op callouts).
    """

    max_autotune: bool = True
    coordinate_descent_tuning: bool = True
    epilogue_fusion: bool = True
    shape_padding: bool = True
    package_constants_in_so: bool = False
    cudagraphs: bool = False
    force_same_precision: bool = False
    # None → the export CLI uses ``$KANDINSKY_HOME/export/inductor``.
    cache_dir: str | None = None

    def inductor_configs(self) -> dict[str, Any]:
        """Flat dict for ``torch._inductor.aoti_compile_and_package``."""
        cfg: dict[str, Any] = {
            "aot_inductor.package_constants_in_so": self.package_constants_in_so,
            "max_autotune": self.max_autotune,
            "coordinate_descent_tuning": self.coordinate_descent_tuning,
            "epilogue_fusion": self.epilogue_fusion,
            "shape_padding": self.shape_padding,
            "force_same_precision": self.force_same_precision,
        }
        if self.cudagraphs:
            cfg["triton.cudagraphs"] = True
        return cfg

class CompileConfig(ConfigModel):
    """DiT compile: runtime ``torch.compile`` + optional AOTI export settings.

    ``strategy`` — ``DiTCompiler`` at pipeline load (``none`` | ``torch``).
    Ignored when ``paths.dit_export`` is set (regional block AOTI replaces it).

    ``aoti`` — Inductor knobs for ``kandy export`` (always read by the export CLI).
    """

    strategy: CompileStrategyName = "torch"
    aoti: AotiCompileConfig = Field(default_factory=AotiCompileConfig)
