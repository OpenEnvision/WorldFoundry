"""Precision-aware generator loading for Echo 1.5 inference."""

from __future__ import annotations

import gc
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch

from worldfoundry.base_models.diffusion_model.models.networks.ltx.model import X0Model
from worldfoundry.synthesis.visual_generation.joyai_echo.echo_longvideo_runtime.ltx_core.model.transformer import (
    LTXModelConfigurator,
)

from ..ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader
from .models.ltx_wrapper import LTX2DiffusionWrapper, create_ltx2_wrapper
from .quantization import build_prequant_fp8_policy, inspect_prequant_fp8_checkpoint
from .release_checkpoint import ReleaseCheckpoint, resolve_release_checkpoint

BF16 = "bf16"
FP8 = "fp8"
FP4 = "fp4"


@dataclass(frozen=True)
class GeneratorLoadReport:
    """Serializable record of the generator topology and weights used at runtime."""

    mode: str
    checkpoint: str
    backend: str
    format: str
    quantized_modules: int | None = None
    missing_keys: tuple[str, ...] = ()
    unexpected_keys: tuple[str, ...] = ()


def _create_wrapper(
    checkpoint: Path,
    gemma_path: Path,
    device: torch.device,
    dtype: torch.dtype,
    video_height: int,
    video_width: int,
    *,
    quantization=None,
):
    return create_ltx2_wrapper(
        checkpoint_path=str(checkpoint),
        gemma_path=str(gemma_path),
        device=device,
        dtype=dtype,
        video_height=video_height,
        video_width=video_width,
        quantization=quantization,
    )


def _load_fp4_modelopt_generator(
    *,
    components: Path,
    checkpoint: Path,
    device: torch.device,
    video_height: int,
    video_width: int,
) -> tuple[torch.nn.Module, int]:
    try:
        from modelopt.torch.opt.conversion import restore_from_modelopt_state
        from modelopt.torch.quantization.plugins.diffusion.ltx2 import (
            register_ltx2_quant_linear,
        )
        from modelopt.torch.utils import safe_load
    except ImportError as error:
        raise ImportError("FP4 inference requires NVIDIA ModelOpt 0.45.0; install requirements-fp4.txt") from error

    config = SafetensorsModelStateDictLoader().metadata(str(components))
    if not config:
        raise ValueError(f"FP4 components checkpoint is missing LTX config metadata: {components}")

    # Build only the topology. No BF16 DiT parameters are materialized: ModelOpt
    # mutates this meta graph and the packed checkpoint tensors are then assigned
    # directly into it. This is the important distinction from modelopt.restore(),
    # which first required a complete BF16 velocity model.
    with torch.device("meta"):
        velocity_model = LTXModelConfigurator.from_config(config)

    register_ltx2_quant_linear()
    packed = safe_load(
        str(checkpoint),
        map_location="cpu",
        mmap=True,
        # Official ModelOpt checkpoints contain its QTensor subclasses and
        # conversion metadata. Only load checkpoints from the trusted release.
        weights_only=False,
    )
    if not isinstance(packed, dict) or not {
        "modelopt_state",
        "model_state_dict",
    }.issubset(packed):
        raise ValueError(f"invalid packed ModelOpt checkpoint: {checkpoint}")
    velocity_model = restore_from_modelopt_state(
        velocity_model,
        packed["modelopt_state"],
    )
    incompatible = velocity_model.load_state_dict(
        packed["model_state_dict"],
        strict=True,
        assign=True,
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "packed FP4 state does not match the LTX transformer topology: "
            f"missing={len(incompatible.missing_keys)} "
            f"unexpected={len(incompatible.unexpected_keys)}"
        )
    del packed
    gc.collect()

    generator = LTX2DiffusionWrapper(
        model=X0Model(velocity_model),
        video_height=video_height,
        video_width=video_width,
    )
    generator.to(device)
    # ModelOpt emits this once per unsupported matrix shape and can flood a
    # single inference log with thousands of multi-line warnings. The public
    # README documents the fallback; retain all other ModelOpt warnings.
    warnings.filterwarnings(
        "ignore",
        message=r"RealQuantLinear: No real-quant GEMM found:.*",
        category=UserWarning,
        module=r"modelopt\.torch\.quantization\.nn\.modules\.quant_linear",
    )
    quantized_modules = sum(module.__class__.__name__ == "TensorQuantizer" for module in generator.modules())
    return generator, quantized_modules


def load_inference_generator(
    *,
    checkpoint: str | Path | ReleaseCheckpoint,
    gemma_path: str | Path,
    device: torch.device,
    dtype: torch.dtype,
    video_height: int,
    video_width: int,
    load_on_cpu: bool = False,
) -> tuple[torch.nn.Module, GeneratorLoadReport]:
    """Build a generator from one of the three public checkpoint directories."""

    release = checkpoint if isinstance(checkpoint, ReleaseCheckpoint) else resolve_release_checkpoint(checkpoint)
    mode = release.precision
    model_path = release.model_path
    gemma = Path(gemma_path).expanduser().resolve()
    load_device = torch.device("cpu") if load_on_cpu else device

    if mode == BF16:
        generator = _create_wrapper(model_path, gemma, load_device, dtype, video_height, video_width)
        generator.eval()
        return generator, GeneratorLoadReport(
            mode=mode,
            checkpoint=str(release.root),
            backend="torch-bfloat16",
            format="full_dmd_merged",
        )

    if mode == FP8:
        if not hasattr(torch, "_scaled_mm"):
            raise RuntimeError("this PyTorch build does not provide torch._scaled_mm")
        info = inspect_prequant_fp8_checkpoint(model_path)
        generator = _create_wrapper(
            model_path,
            gemma,
            load_device,
            dtype,
            video_height,
            video_width,
            quantization=build_prequant_fp8_policy(info),
        )
        generator.eval()
        return generator, GeneratorLoadReport(
            mode=mode,
            checkpoint=str(release.root),
            backend="torch-scaled-mm",
            format="full_prequant_e4m3_scaled_mm",
            quantized_modules=info.module_count,
        )

    if release.modelopt_path is None:
        raise ValueError("echo15_fp4 is missing its packed ModelOpt state")
    target_device = torch.device("cpu") if load_on_cpu else device
    generator, quantized_modules = _load_fp4_modelopt_generator(
        components=model_path,
        checkpoint=release.modelopt_path,
        device=target_device,
        video_height=video_height,
        video_width=video_width,
    )
    generator.eval()
    return generator, GeneratorLoadReport(
        mode=mode,
        checkpoint=str(release.root),
        backend="modelopt-nvfp4-packed",
        format="modelopt_nvfp4_e2m1_block16_fp8_scale",
        quantized_modules=quantized_modules,
    )
