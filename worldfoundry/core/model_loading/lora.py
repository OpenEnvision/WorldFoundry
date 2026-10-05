"""Small, framework-independent helpers for merging inference LoRA weights.

Released adapters use several naming layouts (PEFT A/B, LightX2V pair/
diff, rank-scaled alpha, flattened ComfyUI paths, or unnamed ordered
up/down lists). :class:`GeneralLoRALoader` and
:class:`LightX2VLoRALoader` fuse named matrices in place.
:func:`merge_named_lora_`, :func:`merge_rank_scaled_lora_`,
:func:`merge_flattened_path_lora_`, and :func:`merge_ordered_lora_`
are the public entry points.

Merges mutate the base module; they do not keep a separate adapter
stack for later unload.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence

import torch
from torch import nn

LOGGER = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────
# Named PEFT A/B (or up/down) — fuse in place; no unload stack
# ──────────────────────────────────────────────────────────────────────────


class GeneralLoRALoader:
    """Merge named PEFT-style LoRA matrices into a model in place."""

    def __init__(self, device: str | torch.device = "cpu", torch_dtype: torch.dtype = torch.float32) -> None:
        """Stage LoRA matrices on ``device`` in ``torch_dtype`` before fusing."""
        self.device = device
        self.torch_dtype = torch_dtype

    @staticmethod
    def get_name_dict(lora_state_dict) -> dict[str, tuple[str, str]]:
        """Map base module names to ``(up, down)`` keys for PEFT or lora_up/down."""
        pairs = {}
        for key in lora_state_dict:
            if ".lora_up." in key:
                down_token, up_token = "lora_down", "lora_up"
            else:
                down_token, up_token = "lora_A", "lora_B"
            if f".{up_token}." not in key:
                continue
            parts = key.split(".")
            position = parts.index(up_token)
            if len(parts) > position + 2:
                parts.pop(position + 1)
            parts.pop(position)
            if parts and parts[0] == "diffusion_model":
                parts.pop(0)
            parts.pop(-1)
            pairs[".".join(parts)] = (key, key.replace(f".{up_token}.", f".{down_token}."))
        return pairs

    def convert_state_dict(self, state_dict, suffix: str = ".weight"):
        """Normalize common LoRA naming layouts to PEFT A/B names."""

        converted = {}
        for name, (up_key, down_key) in self.get_name_dict(state_dict).items():
            converted[f"{name}.lora_B{suffix}"] = state_dict[up_key]
            converted[f"{name}.lora_A{suffix}"] = state_dict[down_key]
        return converted

    @torch.no_grad()
    def fuse_lora_to_base_model(self, model: nn.Module, state_dict, alpha: float = 1.0) -> int:
        """Fuse a normalized LoRA state dict into base weights."""

        normalized = self.convert_state_dict(state_dict)
        updated = 0
        for name, module in model.named_modules():
            key_up = f"{name}.lora_B.weight"
            key_down = f"{name}.lora_A.weight"
            if key_up not in normalized or key_down not in normalized or not hasattr(module, "weight"):
                continue
            weight_up = normalized[key_up].to(device=self.device, dtype=self.torch_dtype)
            weight_down = normalized[key_down].to(device=self.device, dtype=self.torch_dtype)
            if weight_up.ndim == 4:
                weight_up = weight_up.squeeze(-1).squeeze(-1)
                weight_down = weight_down.squeeze(-1).squeeze(-1)
                delta = torch.mm(weight_up, weight_down).unsqueeze(-1).unsqueeze(-1)
            else:
                delta = torch.mm(weight_up, weight_down)
            module.weight.add_(delta.to(module.weight), alpha=float(alpha))
            updated += 1
        LOGGER.info("fused %d LoRA tensors into base weights", updated)
        return updated

    @torch.no_grad()
    def load(self, model: nn.Module, state_dict_lora, alpha: float = 1.0) -> int:
        """Fuse named LoRA A/B (or up/down) matrices into matching module weights."""
        updated = 0
        pairs = self.get_name_dict(state_dict_lora)
        for name, module in model.named_modules():
            if name not in pairs or not hasattr(module, "weight"):
                continue
            up_key, down_key = pairs[name]
            weight_up = state_dict_lora[up_key].to(device=self.device, dtype=self.torch_dtype)
            weight_down = state_dict_lora[down_key].to(device=self.device, dtype=self.torch_dtype)
            if weight_up.ndim == 4:
                weight_up = weight_up.squeeze(-1).squeeze(-1)
                weight_down = weight_down.squeeze(-1).squeeze(-1)
                delta = torch.mm(weight_up, weight_down).unsqueeze(-1).unsqueeze(-1)
            else:
                delta = torch.mm(weight_up, weight_down)
            module.weight.add_(delta.to(module.weight), alpha=float(alpha))
            updated += 1
        LOGGER.info("merged %d named LoRA tensors", updated)
        return updated


# ──────────────────────────────────────────────────────────────────────────
# Public named merge — optional expected_modules ABI check
# ──────────────────────────────────────────────────────────────────────────


def merge_named_lora_(
    model: nn.Module,
    state_dict,
    *,
    alpha: float = 1.0,
    expected_modules: int | None = None,
    owner: str = "LoRA",
    device: str | torch.device = "cpu",
    torch_dtype: torch.dtype = torch.float32,
) -> int:
    """Fuse a named A/B adapter and optionally enforce its module ABI."""

    merged = GeneralLoRALoader(device=device, torch_dtype=torch_dtype).load(
        model,
        state_dict,
        alpha=alpha,
    )
    if expected_modules is not None and merged != int(expected_modules):
        raise ValueError(f"{owner} merged {merged} modules; expected {int(expected_modules)}")
    return merged


# ──────────────────────────────────────────────────────────────────────────
# LightX2V pair/diff + rank-scaled alpha + flattened ComfyUI paths
# ──────────────────────────────────────────────────────────────────────────


class LightX2VLoRALoader(GeneralLoRALoader):
    """Merge LightX2V pair/diff checkpoint layouts without a backend dependency."""

    @staticmethod
    def get_lightx2v_mappings(
        lora_state_dict,
    ) -> tuple[dict[str, tuple[str, str]], dict[str, str]]:
        """Return paired LoRA matrices and direct-delta tensor mappings.

        This is intentionally separate from ``GeneralLoRALoader.get_name_dict``
        so subclasses retain the parent's dictionary return contract.
        """

        pairs: dict[str, tuple[str, str]] = {}
        diffs: dict[str, str] = {}
        for prefix in ("", "diffusion_model."):
            for key in lora_state_dict:
                if not key.startswith(prefix):
                    continue
                for suffix_a, suffix_b, target_suffix in (
                    ("lora_A.weight", "lora_B.weight", "weight"),
                    ("lora_down.weight", "lora_up.weight", "weight"),
                ):
                    if key.endswith(suffix_a):
                        pair_key = key[: -len(suffix_a)] + suffix_b
                        if pair_key in lora_state_dict:
                            pairs[key[len(prefix) :].replace(suffix_a, target_suffix)] = (key, pair_key)
                for suffix, target_suffix in (("diff", "weight"), ("diff_b", "bias"), ("diff_m", "modulation")):
                    if key.endswith(suffix):
                        diffs[key[len(prefix) :].replace(suffix, target_suffix)] = key
        return pairs, diffs

    @torch.no_grad()
    def load(self, model: nn.Module, state_dict_lora, alpha: float = 1.0) -> int:
        """Fuse LightX2V pair matrices and direct-delta tensors into named parameters."""
        pairs, diffs = self.get_lightx2v_mappings(state_dict_lora)
        updated = 0
        for name, parameter in model.named_parameters():
            if name in pairs:
                key_a, key_b = pairs[name]
                lora_a = state_dict_lora[key_a].to(device=parameter.device, dtype=self.torch_dtype)
                lora_b = state_dict_lora[key_b].to(device=parameter.device, dtype=self.torch_dtype)
                if lora_a.ndim != 2 or lora_b.ndim != 2:
                    raise ValueError(f"LightX2V LoRA pair for {name} must contain matrices")
                delta = lora_b @ lora_a
                if parameter.shape != delta.shape:
                    raise ValueError(
                        f"LightX2V LoRA delta for {name} has shape {tuple(delta.shape)}, "
                        f"expected {tuple(parameter.shape)}"
                    )
                parameter.add_(delta.to(dtype=parameter.dtype), alpha=float(alpha))
                updated += 1
            elif name in diffs:
                delta = state_dict_lora[diffs[name]].to(device=parameter.device, dtype=self.torch_dtype)
                if parameter.shape != delta.shape:
                    raise ValueError(
                        f"LightX2V direct delta for {name} has shape {tuple(delta.shape)}, "
                        f"expected {tuple(parameter.shape)}"
                    )
                parameter.add_(delta.to(dtype=parameter.dtype), alpha=float(alpha))
                updated += 1
        LOGGER.info("merged %d LightX2V LoRA tensors", updated)
        return updated


@torch.no_grad()
def merge_rank_scaled_lora_(
    model: nn.Module,
    state_dict_lora,
    *,
    alpha: float = 1.0,
) -> int:
    """Merge checkpoints whose stored alpha is normalized by LoRA rank.

    This is the layout used by released LightX2V acceleration weights for
    Wan2.2.  It deliberately operates parameter-by-parameter so a 14B model
    is never duplicated into a second full state dict during inference setup.
    """

    prefix = "diffusion_model." if any(key.startswith("diffusion_model.") for key in state_dict_lora) else ""
    updated = 0
    for name, parameter in model.named_parameters():
        if not name.endswith(".weight"):
            continue
        stem = prefix + name[: -len(".weight")]
        pair = None
        for down_suffix, up_suffix in (
            (".lora_down.weight", ".lora_up.weight"),
            (".lora_A.weight", ".lora_B.weight"),
        ):
            down_key = stem + down_suffix
            up_key = stem + up_suffix
            if down_key in state_dict_lora and up_key in state_dict_lora:
                pair = down_key, up_key
                break
        if pair is None:
            continue
        down = state_dict_lora[pair[0]].to(device=parameter.device, dtype=torch.float32)
        up = state_dict_lora[pair[1]].to(device=parameter.device, dtype=torch.float32)
        if down.ndim != 2 or up.ndim != 2:
            raise ValueError(f"rank-scaled LoRA for {name} must contain matrices")
        delta = up @ down
        if delta.shape != parameter.shape:
            raise ValueError(
                f"rank-scaled LoRA delta for {name} has shape {tuple(delta.shape)}, "
                f"expected {tuple(parameter.shape)}"
            )
        alpha_key = stem + ".alpha"
        layer_alpha = float(state_dict_lora[alpha_key].item()) if alpha_key in state_dict_lora else float(alpha)
        parameter.add_(delta.to(dtype=parameter.dtype), alpha=layer_alpha / down.shape[0])
        updated += 1
    LOGGER.info("merged %d rank-scaled LoRA tensors", updated)
    return updated


@torch.no_grad()
def merge_flattened_path_lora_(
    model: nn.Module,
    state_dict_lora,
    *,
    prefix: str = "lora_unet__",
    scale: float = 1.0,
) -> int:
    """Merge LoRA weights whose module paths are flattened with underscores.

    Several inference releases encode ``blocks.0.cross_attn.q`` as
    ``lora_unet__blocks_0_cross_attn_q``.  Resolving checkpoint strings back
    into attributes is ambiguous when module names contain underscores.  This
    helper instead derives the exact encoded stem from each native module path,
    which is deterministic and keeps the policy reusable across model families.
    Stored ``alpha`` values are normalized by rank, matching the common
    ComfyUI/DiffSynth layout.
    """

    updated = 0
    for name, module in model.named_modules():
        if not name or not hasattr(module, "weight"):
            continue
        stem = prefix + name.replace(".", "_")
        pair = None
        for down_suffix, up_suffix in (
            (".lora_down.weight", ".lora_up.weight"),
            (".lora_A.weight", ".lora_B.weight"),
        ):
            down_key = stem + down_suffix
            up_key = stem + up_suffix
            if down_key in state_dict_lora and up_key in state_dict_lora:
                pair = down_key, up_key
                break
        if pair is None:
            continue
        down = state_dict_lora[pair[0]].to(device=module.weight.device, dtype=torch.float32)
        up = state_dict_lora[pair[1]].to(device=module.weight.device, dtype=torch.float32)
        if down.ndim == 4:
            down = down.squeeze(-1).squeeze(-1)
            up = up.squeeze(-1).squeeze(-1)
            delta = (up @ down).unsqueeze(-1).unsqueeze(-1)
        elif down.ndim == 2 and up.ndim == 2:
            delta = up @ down
        else:
            raise ValueError(f"flattened-path LoRA for {name} must contain matrices or 1x1 kernels")
        if delta.shape != module.weight.shape:
            raise ValueError(
                f"flattened-path LoRA delta for {name} has shape {tuple(delta.shape)}, "
                f"expected {tuple(module.weight.shape)}"
            )
        alpha_key = stem + ".alpha"
        layer_alpha = (
            float(state_dict_lora[alpha_key].item())
            if alpha_key in state_dict_lora
            else float(down.shape[0])
        )
        module.weight.add_(
            delta.to(dtype=module.weight.dtype),
            alpha=float(scale) * layer_alpha / down.shape[0],
        )
        updated += 1
    LOGGER.info("merged %d flattened-path LoRA tensors", updated)
    return updated


# ──────────────────────────────────────────────────────────────────────────
# Unnamed alternating up/down lists — traversal order is part of the format
# ──────────────────────────────────────────────────────────────────────────


def _ordered_modules(
    model: nn.Module,
    *,
    ancestor_class_names: frozenset[str],
) -> Iterable[nn.Module]:
    """Yield Linear/Conv children under named ancestor classes in ``modules()`` order.

    Unnamed LoRA lists have no parameter keys; this traversal *is* the ABI.
    Changing ancestor names silently remaps weights onto the wrong layers.
    """
    for ancestor in model.modules():
        if ancestor.__class__.__name__ not in ancestor_class_names:
            continue
        for module in ancestor.modules():
            if module is ancestor:
                continue
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
                yield module


def _lora_delta(up: torch.Tensor, down: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """``up @ down`` for Linear, flattened GEMM then reshape for Conv.

    Failure: :exc:`ValueError` when ranks or the resulting shape disagree
    with *target* — fail closed rather than broadcast into the wrong layer.
    """
    if target.ndim == 2:
        if up.ndim != 2 or down.ndim != 2:
            raise ValueError("linear LoRA tensors must be matrices")
        delta = up @ down
    else:
        if up.ndim != target.ndim or down.ndim != target.ndim:
            raise ValueError("convolution LoRA tensor ranks must match the target")
        delta = (up.flatten(start_dim=1) @ down.flatten(start_dim=1)).reshape(target.shape)
    if delta.shape != target.shape:
        raise ValueError(f"LoRA delta shape {tuple(delta.shape)} does not match {tuple(target.shape)}")
    return delta


@torch.no_grad()
def merge_ordered_lora_(
    model: nn.Module,
    tensors: Sequence[torch.Tensor],
    *,
    ancestor_class_names: Sequence[str],
    scale: float = 1.0,
) -> int:
    """Merge an upstream alternating ``up, down`` tensor list into modules in traversal order.

    Some early LoRA releases stored no parameter names. Their traversal order is
    part of the checkpoint format, so callers must declare the ancestor class
    names that define that order. Shape checks fail closed if code and checkpoint
    no longer describe the same module sequence.
    """

    values = list(tensors)
    if len(values) % 2:
        raise ValueError("ordered LoRA checkpoints must contain up/down tensor pairs")
    if not all(isinstance(value, torch.Tensor) for value in values):
        raise TypeError("ordered LoRA checkpoints must contain tensors only")
    targets = frozenset(str(value) for value in ancestor_class_names if str(value))
    if not targets:
        raise ValueError("ancestor_class_names cannot be empty")

    cursor = 0
    merged = 0
    for module in _ordered_modules(model, ancestor_class_names=targets):
        if cursor >= len(values):
            break
        up, down = values[cursor : cursor + 2]
        if up.ndim != module.weight.ndim:
            continue
        delta = _lora_delta(up, down, module.weight)
        module.weight.add_(
            delta.to(device=module.weight.device, dtype=module.weight.dtype),
            alpha=float(scale),
        )
        cursor += 2
        merged += 1

    if cursor != len(values):
        remaining = len(values) - cursor
        raise ValueError(f"LoRA traversal ended with {remaining} unmatched tensors")
    return merged


__all__ = [
    "GeneralLoRALoader",
    "LightX2VLoRALoader",
    "merge_flattened_path_lora_",
    "merge_named_lora_",
    "merge_ordered_lora_",
    "merge_rank_scaled_lora_",
]
