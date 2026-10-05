"""Inference-time LoRA weight merging for VideoX-Fun pipelines."""

from collections import defaultdict

import torch
from safetensors.torch import load_file

from worldfoundry.core.model_loading.checkpoints import load_tensor_state_dict

from videox_fun.utils.group_offload import (
    _is_group_offload_enabled,
    register_auto_device_hook,
    safe_enable_group_offload,
    safe_remove_group_offloading,
)


def _load_lora(path):
    if path.endswith(".safetensors"):
        return load_file(path)
    return load_tensor_state_dict(path)


def _normalise_lora_key(key):
    if "diffusion_model." in key:
        key = key.replace("diffusion_model.", "")
    if not key.startswith(("lora_unet", "lora_te")):
        key = "lora_unet__" + key

    key = key.replace(".", "_")
    suffixes = {
        "_lora_up_weight": ".lora_up.weight",
        "_lora_down_weight": ".lora_down.weight",
        "_lora_A_default_weight": ".lora_down.weight",
        "_lora_B_default_weight": ".lora_up.weight",
        "_lora_A_weight": ".lora_down.weight",
        "_lora_B_weight": ".lora_up.weight",
        "_alpha": ".alpha",
    }
    for suffix, replacement in suffixes.items():
        if key.endswith(suffix):
            return key[:-len(suffix)] + replacement
    return key


def _organise_lora(state_dict):
    updates = defaultdict(dict)
    for key, value in state_dict.items():
        key = _normalise_lora_key(key)
        if "." not in key:
            continue
        layer, element = key.split(".", 1)
        updates[layer][element] = value
    return updates


def _resolve_module(root, flattened_name):
    """Resolve an underscore-flattened Kohya module path."""
    parts = [part for part in flattened_name.strip("_").split("_") if part]
    if not parts:
        return root
    for end in range(len(parts), 0, -1):
        name = "_".join(parts[:end])
        if not hasattr(root, name):
            continue
        child = getattr(root, name)
        if end == len(parts):
            return child
        try:
            return _resolve_module(child, "_".join(parts[end:]))
        except AttributeError:
            continue
    raise AttributeError(flattened_name)


def _target_module(pipeline, layer, sub_transformer_name, transformer_only):
    if layer.startswith("lora_te"):
        if transformer_only:
            return None
        root = pipeline.text_encoder
        name = layer[len("lora_te"):]
    else:
        root = getattr(pipeline, sub_transformer_name)
        name = layer[len("lora_unet"):]
        name = name.strip("_")
        if name == sub_transformer_name:
            return root
        prefix = f"{sub_transformer_name}_"
        if name.startswith(prefix):
            name = name[len(prefix):]
    return _resolve_module(root, name)


def _refresh_group_offload(pipeline, sub_transformer_name, device):
    transformer = getattr(pipeline, sub_transformer_name)
    if not _is_group_offload_enabled(transformer):
        return
    safe_remove_group_offloading(pipeline)
    register_auto_device_hook(transformer)
    safe_enable_group_offload(
        pipeline,
        onload_device=device,
        offload_device="cpu",
        offload_type="leaf_level",
        use_stream=True,
    )


def _apply_lora(
    pipeline,
    lora_path,
    multiplier,
    *,
    sign,
    device,
    dtype,
    state_dict=None,
    transformer_only=False,
    sub_transformer_name="transformer",
):
    if lora_path is None and state_dict is None:
        return pipeline

    state_dict = state_dict if state_dict is not None else _load_lora(lora_path)
    updates = _organise_lora(state_dict)
    transformer = getattr(pipeline, sub_transformer_name)
    transformer_device = getattr(transformer, "device", None)
    if transformer_device is None:
        parameter = next(transformer.parameters(), None)
        transformer_device = parameter.device if parameter is not None else torch.device("cpu")
    sequential_offload = transformer_device == torch.device("meta")
    offload_device = getattr(pipeline, "_offload_device", device)
    if sequential_offload:
        pipeline.remove_all_hooks()

    applied = 0
    skipped = 0
    with torch.no_grad():
        for layer, elements in updates.items():
            try:
                module = _target_module(
                    pipeline, layer, sub_transformer_name, transformer_only)
                if module is None:
                    skipped += 1
                    continue
                up = elements["lora_up.weight"].to(device=device, dtype=dtype)
                down = elements["lora_down.weight"].to(device=device, dtype=dtype)
            except (AttributeError, KeyError) as exc:
                print(f"[LoRA] Skipping {layer}: {exc}")
                skipped += 1
                continue

            original_device = module.weight.device
            original_dtype = module.weight.dtype
            module.to(device=device, dtype=dtype)
            alpha = elements.get("alpha")
            scale = 1.0 if alpha is None else alpha.item() / up.shape[1]
            if up.ndim == 4:
                delta = torch.mm(up.squeeze(3).squeeze(2), down.squeeze(3).squeeze(2))
                delta = delta.unsqueeze(2).unsqueeze(3)
            else:
                delta = torch.mm(up, down)
            module.weight.add_(delta, alpha=sign * multiplier * scale)
            module.to(device=original_device, dtype=original_dtype)
            applied += 1

    if sequential_offload:
        pipeline.enable_sequential_cpu_offload(device=offload_device)
    else:
        try:
            _refresh_group_offload(pipeline, sub_transformer_name, device)
        except Exception as exc:
            print(f"[LoRA] Failed to refresh group offload: {exc}")

    action = "merged" if sign > 0 else "unmerged"
    print(f"[LoRA] {action} {applied} layers; skipped {skipped}")
    return pipeline


def merge_lora(
    pipeline,
    lora_path,
    multiplier,
    device="cpu",
    dtype=torch.float32,
    state_dict=None,
    transformer_only=False,
    sub_transformer_name="transformer",
):
    return _apply_lora(
        pipeline,
        lora_path,
        multiplier,
        sign=1,
        device=device,
        dtype=dtype,
        state_dict=state_dict,
        transformer_only=transformer_only,
        sub_transformer_name=sub_transformer_name,
    )


def unmerge_lora(
    pipeline,
    lora_path,
    multiplier=1,
    device="cpu",
    dtype=torch.float32,
    sub_transformer_name="transformer",
):
    return _apply_lora(
        pipeline,
        lora_path,
        multiplier,
        sign=-1,
        device=device,
        dtype=dtype,
        sub_transformer_name=sub_transformer_name,
    )
