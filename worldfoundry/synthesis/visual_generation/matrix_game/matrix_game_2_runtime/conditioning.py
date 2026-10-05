"""Prove when the action Wan VAE's image-plus-blank condition becomes steady."""

from __future__ import annotations

from torch import nn

from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.variants.action_21 import (
    CACHE_T,
    AttentionBlock,
    CausalConv3d,
    Encoder3d,
    Resample,
    ResidualBlock,
    RMS_norm,
)


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _unsupported(module: nn.Module, path: str) -> ValueError:
    return ValueError(
        "Cannot prove Matrix-Game 2 blank-tail stationarity for "
        f"{type(module).__name__} at {path}."
    )


def _check_spatial_graph(module: nn.Module, path: str) -> None:
    """Check the graph run after time has been folded into the batch axis."""

    if type(module) is nn.Sequential:
        for name, child in module._modules.items():
            _check_spatial_graph(child, f"{path}.{name}")
    elif type(module) not in {nn.Conv2d, nn.ZeroPad2d, nn.Identity, RMS_norm} or module._modules:
        raise _unsupported(module, path)


def _first_stable_after(module: nn.Module, onset: int, path: str) -> int:
    # Exact types reject subclasses whose forward may introduce new temporal
    # dependencies. No generic fallback silently assumes a layer is framewise.
    if type(module) is nn.Sequential:
        # named_children deduplicates shared layers, whereas Sequential.forward
        # executes every registered occurrence of a layer.
        for name, child in module._modules.items():
            onset = _first_stable_after(child, onset, f"{path}.{name}")
        return onset
    if type(module) is CausalConv3d:
        lag = module.dilation[0] * (module.kernel_size[0] - 1)
        if module._padding[4:] != (lag, 0) or lag > CACHE_T or module.stride[0] != 1:
            # Encoder.forward caches at most CACHE_T frames. A larger kernel or
            # a new strided path would need a different chunk/cache proof.
            raise _unsupported(module, path)
        return _ceil_div(onset + lag, module.stride[0])
    if type(module) is ResidualBlock:
        if set(module._modules) != {"residual", "shortcut"}:
            raise _unsupported(module, path)
        if type(module.shortcut) is CausalConv3d and module.shortcut.kernel_size[0] != 1:
            # Unlike the residual branch, shortcut(x) receives no history.
            # A temporal shortcut would restart padding on every chunk.
            raise _unsupported(module.shortcut, f"{path}.shortcut")
        residual = _first_stable_after(module.residual, onset, f"{path}.residual")
        shortcut = _first_stable_after(module.shortcut, onset, f"{path}.shortcut")
        return max(residual, shortcut)
    if type(module) is Resample:
        expected = {"resample", "time_conv"} if module.mode == "downsample3d" else {"resample"}
        if set(module._modules) != expected:
            raise _unsupported(module, path)
        _check_spatial_graph(module.resample, f"{path}.resample")
        if module.mode in {"none", "downsample2d"}:
            return onset
        if module.mode == "downsample3d":
            conv = module.time_conv
            if (
                type(conv) is not CausalConv3d
                or conv.kernel_size[0] != 3
                or conv.stride[0] != 2
                or conv.dilation[0] != 1
                or conv._padding[4:] != (0, 0)
            ):
                raise _unsupported(module, path)
            # Output 0 bypasses time_conv. For n>=1, the cached preceding
            # frame gives exactly the input window [2*n-2, 2*n-1, 2*n].
            return max(1, _ceil_div(onset + 2, 2))
        raise _unsupported(module, path)
    if type(module) is AttentionBlock:
        if set(module._modules) != {"norm", "to_qkv", "proj"}:
            raise _unsupported(module, path)
        for name, child in module._modules.items():
            _check_spatial_graph(child, f"{path}.{name}")
        return onset
    if type(module) in {RMS_norm, nn.SiLU, nn.Identity} and not module._modules:
        # RMS_norm only reduces channels; AttentionBlock folds time into the
        # batch axis and attends to spatial positions within each frame.
        return onset
    if type(module) is nn.Dropout and not module._modules and (module.p == 0 or not module.training):
        return onset
    raise _unsupported(module, path)


def first_stable_encoder_latent(encoder: Encoder3d) -> int:
    """Return the first provably stationary latent for image + repeated zeros.

    Raw frame 0 contains the image; raw frames >=1 are identical blanks.
    Propagate that stationary onset through the actual encoder graph, including
    cold-start causal padding and the first-frame temporal-resample bypass.
    For the released encoder this is latent 29 (113-frame receptive field).
    Unknown layers fail closed rather than permitting an unproved tail reuse.
    """

    if type(encoder) is not Encoder3d or set(encoder._modules) != {
        "conv1", "downsamples", "middle", "head",
    }:
        raise _unsupported(encoder, "encoder")
    onset = 1
    for name in ("conv1", "downsamples", "middle", "head"):
        onset = _first_stable_after(getattr(encoder, name), onset, f"encoder.{name}")
    return onset


def minimum_condition_prefetch_blocks(encoder: Encoder3d, block_frames: int = 3) -> int:
    """Prefetch enough latents that every frame of the reused last block is steady."""

    if not isinstance(block_frames, int) or isinstance(block_frames, bool) or block_frames < 1:
        raise ValueError("Conditioning block_frames must be a positive integer.")
    return _ceil_div(first_stable_encoder_latent(encoder) + block_frames, block_frames)


__all__ = ["first_stable_encoder_latent", "minimum_condition_prefetch_blocks"]
