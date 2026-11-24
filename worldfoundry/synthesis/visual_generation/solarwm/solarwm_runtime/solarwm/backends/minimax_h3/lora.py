"""Exact PEFT LoRA-384 target and checkpoint contract for MiniMax-H3."""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ...errors import BackendContractError

H3_LORA_TARGET_COUNT = 312
H3_LORA_TRAINABLE_PARAMETERS = 2_075_394_048
H3_LORA_SUFFIXES = (
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
)


def _indexed_prefixes(modules: Mapping[str, Any], pattern: str, expected: int, label: str) -> tuple[str, ...]:
    expression = re.compile(pattern)
    found: dict[int, str] = {}
    for name in modules:
        match = expression.fullmatch(name)
        if match:
            index = int(match.group(1))
            if index in found:
                raise BackendContractError(f"duplicate {label} block index {index}")
            found[index] = name
    required = set(range(expected))
    if set(found) != required:
        raise BackendContractError(
            f"{label} topology differs: missing={sorted(required - set(found))} extra={sorted(set(found) - required)}"
        )
    return tuple(found[index] for index in range(expected))


def discover_h3_lora_targets(model: Any) -> tuple[str, ...]:
    """Discover and audit all 50 main + 2 refiner block QKVO/FFN linears."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional local dependency
        raise BackendContractError("PyTorch is required for H3 LoRA discovery") from exc
    modules = dict(model.named_modules())
    main = _indexed_prefixes(
        modules,
        r"(?:.*\.)?transformer_blocks\.(\d+)",
        50,
        "MiniMax-H3 main transformer",
    )
    refiner = _indexed_prefixes(
        modules,
        r"(?:.*\.)?token_refiner\.refiner_blocks\.(\d+)",
        2,
        "MiniMax-H3 token refiner",
    )
    blocks = main + refiner
    for block in blocks:
        attention = modules.get(f"{block}.attn")
        if attention is None or getattr(attention, "fused_projections", None) is not False:
            raise BackendContractError(f"H3 LoRA requires split Q/K/V projections at {block!r}")
    targets = tuple(sorted(f"{block}.{suffix}" for block in blocks for suffix in H3_LORA_SUFFIXES))
    if len(targets) != H3_LORA_TARGET_COUNT or len(set(targets)) != len(targets):
        raise AssertionError("internal H3 LoRA target-count error")
    invalid = [name for name in targets if name not in modules or not isinstance(modules[name], torch.nn.Linear)]
    if invalid:
        raise BackendContractError(f"H3 LoRA expected nn.Linear targets; first invalid={invalid[:8]}")
    return targets


@dataclass
class H3LoRARuntime:
    model: Any
    targets: tuple[str, ...]
    parameter_by_key: OrderedDict[str, Any]
    peft_config: Any
    peft_module: Any
    base_identity: Mapping[str, Any]
    rank: int
    alpha: int

    @property
    def parameters(self) -> tuple[Any, ...]:
        return tuple(self.parameter_by_key.values())

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters)

    def state_dict(self) -> OrderedDict[str, Any]:
        return OrderedDict((key, parameter.detach()) for key, parameter in self.parameter_by_key.items())

    def load_state_dict(self, values: Mapping[str, Any], *, broadcast: bool = True) -> None:
        import torch
        import torch.distributed as dist

        if set(values) != set(self.parameter_by_key):
            raise BackendContractError(
                "H3 LoRA checkpoint keys differ: "
                f"missing={sorted(set(self.parameter_by_key) - set(values))[:8]} "
                f"extra={sorted(set(values) - set(self.parameter_by_key))[:8]}"
            )
        with torch.no_grad():
            for key, parameter in self.parameter_by_key.items():
                value = values[key]
                if tuple(value.shape) != tuple(parameter.shape):
                    raise BackendContractError(f"H3 LoRA shape differs for {key!r}")
                if value.dtype != parameter.dtype:
                    raise BackendContractError(f"H3 LoRA dtype differs for {key!r}")
                parameter.copy_(value.to(device=parameter.device))
                if broadcast and dist.is_available() and dist.is_initialized():
                    dist.broadcast(parameter, src=0)

    def metadata(self) -> dict[str, Any]:
        return {
            "schema": "solarwm.minimax-h3-lora.v1",
            "peft_version": str(self.peft_module.__version__),
            "rank": self.rank,
            "alpha": self.alpha,
            "target_count": len(self.targets),
            "target_modules": list(self.targets),
            "state_keys": list(self.parameter_by_key),
            "state_shapes": {key: list(parameter.shape) for key, parameter in self.parameter_by_key.items()},
            "state_dtypes": {
                key: str(parameter.dtype).removeprefix("torch.") for key, parameter in self.parameter_by_key.items()
            },
            "trainable_parameters": self.parameter_count,
            "base_identity": json.loads(json.dumps(dict(self.base_identity), sort_keys=True)),
        }


def inject_h3_lora(
    model: Any,
    adapter_cfg: Mapping[str, Any],
    *,
    base_identity: Mapping[str, Any],
) -> tuple[Any, H3LoRARuntime]:
    """Inject standard BF16 PEFT LoRA and verify its realized topology/size."""

    try:
        import peft
        import torch
        import torch.distributed.tensor
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise BackendContractError("H3 LoRA requires peft==0.20.0") from exc

    rank = int(adapter_cfg.get("rank", 0))
    alpha = int(adapter_cfg.get("alpha", 0))
    if rank != 384 or alpha != 384:
        raise BackendContractError("H3 adapter requires rank=alpha=384")
    targets = discover_h3_lora_targets(model)
    configuration = peft.LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=float(adapter_cfg.get("dropout", 0.0)),
        target_modules=list(targets),
        bias="none",
        init_lora_weights=True,
    )
    wrapped = peft.get_peft_model(
        model,
        configuration,
        adapter_name="default",
        autocast_adapter_dtype=False,
    )
    realized = tuple(sorted(str(name) for name in wrapped.base_model.targeted_module_names))
    if realized != targets:
        raise BackendContractError(
            "PEFT realized a different H3 topology: "
            f"missing={sorted(set(targets) - set(realized))[:8]} "
            f"extra={sorted(set(realized) - set(targets))[:8]}"
        )
    wrapped.peft_config["default"].target_modules = set(targets)
    for parameter in wrapped.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.to(torch.bfloat16)
    keep_vars = wrapped.state_dict(keep_vars=True)
    standard = peft.get_peft_model_state_dict(
        wrapped,
        state_dict=keep_vars,
        adapter_name="default",
        save_embedding_layers=False,
    )
    parameter_by_key: OrderedDict[str, Any] = OrderedDict()
    for key in sorted(standard):
        value = standard[key]
        if not isinstance(value, torch.nn.Parameter):
            raise BackendContractError(f"PEFT state {key!r} is not a Parameter")
        parameter_by_key[key] = value
    trainable = tuple(parameter for parameter in wrapped.parameters() if parameter.requires_grad)
    if {id(value) for value in trainable} != {id(value) for value in parameter_by_key.values()}:
        raise BackendContractError("all and only H3 LoRA parameters must be trainable")
    if len(parameter_by_key) != 2 * H3_LORA_TARGET_COUNT:
        raise BackendContractError("H3 LoRA must expose one A/B tensor pair per target")
    runtime = H3LoRARuntime(
        model=wrapped,
        targets=targets,
        parameter_by_key=parameter_by_key,
        peft_config=wrapped.peft_config["default"],
        peft_module=peft,
        base_identity=dict(base_identity),
        rank=rank,
        alpha=alpha,
    )
    expected = int(adapter_cfg.get("expected_trainable_parameters", H3_LORA_TRAINABLE_PARAMETERS))
    if runtime.parameter_count != expected:
        raise BackendContractError(f"H3 LoRA trainable parameters={runtime.parameter_count:,}, expected={expected:,}")
    if any(parameter.dtype != torch.bfloat16 for parameter in runtime.parameters):
        raise BackendContractError("all H3 LoRA parameters must remain BF16")
    return wrapped, runtime


__all__ = [
    "H3_LORA_SUFFIXES",
    "H3_LORA_TARGET_COUNT",
    "H3_LORA_TRAINABLE_PARAMETERS",
    "H3LoRARuntime",
    "discover_h3_lora_targets",
    "inject_h3_lora",
]
