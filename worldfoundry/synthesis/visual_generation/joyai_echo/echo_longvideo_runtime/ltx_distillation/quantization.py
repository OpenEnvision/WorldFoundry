"""Quantized transformer building blocks used by Echo 1.5 inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import safetensors
import torch
from torch import nn

from worldfoundry.base_models.diffusion_model.models.networks.ltx.variants.joyai_echo.model import (
    JoyAIEchoModel as LTXModel,
)

from ..ltx_core.loader.module_ops import ModuleOps
from ..ltx_core.quantization.policy import QuantizationPolicy

_CHECKPOINT_PREFIX = "model.diffusion_model."


@dataclass(frozen=True)
class FP8CheckpointInfo:
    """Transformer layers carrying direct E4M3 weights and per-tensor scales."""

    checkpoint: str
    module_names: tuple[str, ...]

    @property
    def module_count(self) -> int:
        return len(self.module_names)


def inspect_prequant_fp8_checkpoint(checkpoint_path: str | Path) -> FP8CheckpointInfo:
    """Read only the safetensors header and discover pre-quantized Linear layers."""

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"FP8 checkpoint not found: {path}")
    if path.suffix != ".safetensors":
        raise ValueError("direct FP8 checkpoints must use the safetensors format")

    with safetensors.safe_open(str(path), framework="pt", device="cpu") as handle:
        module_names = tuple(
            sorted(
                key.removeprefix(_CHECKPOINT_PREFIX).removesuffix(".weight_scale")
                for key in handle.keys()  # noqa: SIM118 - safe_open handles are not iterable.
                if key.startswith(_CHECKPOINT_PREFIX) and key.endswith(".weight_scale")
            )
        )
    if not module_names:
        raise ValueError(f"no FP8 weight_scale tensors found in {path}")
    return FP8CheckpointInfo(checkpoint=str(path), module_names=module_names)


class PrequantFP8ScaledMMLinear(nn.Module):
    """Linear with checkpoint-provided E4M3 weights and dynamic activation scaling."""

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = nn.Parameter(
            torch.empty(
                (linear.out_features, linear.in_features),
                dtype=torch.float8_e4m3fn,
                device=linear.weight.device,
            ),
            requires_grad=False,
        )
        self.weight_scale = nn.Parameter(
            torch.empty((), dtype=torch.float32, device=linear.weight.device),
            requires_grad=False,
        )
        self.bias = (
            nn.Parameter(
                torch.empty(
                    (linear.out_features,),
                    dtype=linear.bias.dtype,
                    device=linear.bias.device,
                ),
                requires_grad=False,
            )
            if linear.bias is not None
            else None
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        original_shape = value.shape
        value_2d = value.reshape(-1, value.shape[-1])
        fp8_info = torch.finfo(torch.float8_e4m3fn)
        max_abs = value_2d.float().abs().amax()
        input_scale = torch.where(max_abs > 0, max_abs / fp8_info.max, torch.ones_like(max_abs))
        quantized_input = torch.clamp(
            value_2d.float() / input_scale,
            fp8_info.min,
            fp8_info.max,
        ).to(torch.float8_e4m3fn)
        output = torch._scaled_mm(
            quantized_input,
            self.weight.t(),
            scale_a=input_scale,
            scale_b=self.weight_scale,
            out_dtype=value.dtype,
            use_fast_accum=True,
        )
        # PyTorch releases that expose the amax result return a tuple here.
        if isinstance(output, tuple):
            output = output[0]
        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        return output.reshape(*original_shape[:-1], self.out_features)


def build_prequant_fp8_policy(info: FP8CheckpointInfo) -> QuantizationPolicy:
    """Replace exactly the Linear modules described by an FP8 checkpoint."""

    scale_modules = frozenset(info.module_names)

    def mutate(model: nn.Module) -> nn.Module:
        replacements: list[tuple[nn.Module, str, nn.Linear]] = []
        found_linears: set[str] = set()
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            found_linears.add(name)
            if name in scale_modules:
                parent_name, attr_name = name.rsplit(".", 1)
                replacements.append((model.get_submodule(parent_name), attr_name, module))

        if len(replacements) != len(scale_modules):
            missing = sorted(scale_modules - found_linears)
            raise ValueError(
                "FP8 checkpoint/model layer mismatch: "
                f"scales={len(scale_modules)} replacements={len(replacements)} "
                f"missing_sample={missing[:10]}"
            )
        for parent, attr_name, linear in replacements:
            setattr(parent, attr_name, PrequantFP8ScaledMMLinear(linear))
        return model

    return QuantizationPolicy(
        sd_ops=None,
        module_ops=(
            ModuleOps(
                name="echo15_prequant_fp8_scaled_mm",
                matcher=lambda model: isinstance(model, LTXModel),
                mutator=mutate,
            ),
        ),
    )
