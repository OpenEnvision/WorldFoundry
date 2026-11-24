from typing import cast

from worldfoundry.base_models.diffusion_model.models.networks.ltx.variants.solarwm import SolarWMModel as LTXModel
from worldfoundry.base_models.diffusion_model.optimizations.ltx.compile import (
    CompilationConfig,
)
from worldfoundry.base_models.diffusion_model.optimizations.ltx.compile import (
    compile_transformer as compile_ltx_transformer,
)
from worldfoundry.base_models.diffusion_model.optimizations.ltx.cudagraph import CudaGraphRunner as CanonicalGraphRunner

from ...loader.module_ops import ModuleOps
from ...loader.sd_ops import SDOps
from .cudagraph_capture import CudaGraphRunner


def compile_transformer(model: LTXModel, config: CompilationConfig) -> LTXModel:
    """Reuse native compilation while keeping SolarWM's per-capture input ownership."""
    compile_ltx_transformer(model, config)
    if config.capture:
        runner = cast(CanonicalGraphRunner, model._process_transformer_blocks)
        model._process_transformer_blocks = CudaGraphRunner(runner._block_loop, model.block_input_processor)
    return model


def build_compile_transformer_op(config: CompilationConfig) -> ModuleOps:
    """Build a ``ModuleOps`` that compiles transformer blocks with the given settings."""
    return ModuleOps(
        name="compile_transformer",
        matcher=lambda model: isinstance(model, LTXModel),
        mutator=lambda model: compile_transformer(model, config),
    )


def modify_sd_ops_for_compilation(original_sd_ops: SDOps, number_of_blocks: int = 48) -> SDOps:
    for i in range(number_of_blocks):
        original_sd_ops = original_sd_ops.with_replacement(
            f"transformer_blocks.{i}.", f"transformer_blocks.{i}._orig_mod."
        )
    return original_sd_ops
