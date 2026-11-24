"""Tensor and checkpoint helpers shared by classic diffusion networks.

Shape/cast utilities for U-Net residuals. DiT modulation lives in
``diffusion_transformer``; do not mix the two vocabularies.
"""

import math
from inspect import isfunction

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.checkpoint
from torch import nn

# ──────────────────────────────────────────────────────────────────────────
# Distributed gather and AMP — U-Net residual helpers, not DiT modulation
# ──────────────────────────────────────────────────────────────────────────


def gather_data(data, return_np=True):
    """gather data from multiple processes to one list"""
    data_list = [torch.zeros_like(data) for _ in range(dist.get_world_size())]
    dist.all_gather(data_list, data)  # gather not supported with NCCL
    if return_np:
        data_list = [data.cpu().numpy() for data in data_list]
    return data_list


def autocast(f):
    """Wrap ``f`` to run under CUDA autocast with the current autocast GPU dtype and cache setting."""

    def do_autocast(*args, **kwargs):
        with torch.amp.autocast(
            "cuda",
            enabled=True,
            dtype=torch.get_autocast_gpu_dtype(),
            cache_enabled=torch.is_autocast_cache_enabled(),
        ):
            return f(*args, **kwargs)

    return do_autocast


def extract_into_tensor(a, t, x_shape):
    """Gather ``a`` at timesteps ``t`` and reshape to ``(b, 1, ..., 1)`` to broadcast against ``x_shape``.

    Args:
        a: Per-timestep values, gathered along the last dimension.
        t: Integer timesteps of shape ``[b]``.
        x_shape: Shape of the tensor the result broadcasts against.
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def extract_numpy_into_tensor(
    arr: np.ndarray, timesteps: torch.Tensor, broadcast_shape: tuple[int, ...]
) -> torch.Tensor:
    """Index a NumPy schedule on the timestep device and add a full default-dtype zero tensor."""
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + torch.zeros(broadcast_shape, device=timesteps.device)


def append_zero(x: torch.Tensor) -> torch.Tensor:
    """Append one zero without changing the schedule's dtype or device."""
    return torch.cat([x, x.new_zeros([1])])


def extract_into_tensor_2d(a, t, x_shape):
    """Extract values from a 1D tensor for 1D or [batch, time] indices."""
    if len(t.shape) == 1:
        b = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))
    if len(t.shape) == 2:
        b, t_dim = t.shape
        ndim = len(x_shape)
        # Video is B,C,T,... so gather along axis 2; 2-D images gather on axis 1.
        time_dim_idx = 2 if ndim >= 3 else 1
        flat_values = a.gather(-1, t.reshape(-1))
        values_bt = flat_values.reshape(b, t_dim)
        output_shape = [b]
        for i in range(1, ndim):
            output_shape.append(t_dim if i == time_dim_idx else 1)
        return values_bt.reshape(output_shape)
    raise ValueError(f"extract_into_tensor_2d expects 1D or 2D timesteps, got shape {tuple(t.shape)}")


# ──────────────────────────────────────────────────────────────────────────
# Noise, defaults, and small predicates used by classic U-Net graphs
# ──────────────────────────────────────────────────────────────────────────


def noise_like(shape, device, repeat=False):
    """Return standard normal noise of ``shape`` on ``device``; with ``repeat`` one sample is shared across the batch."""

    def repeat_noise():
        return torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))

    def noise():
        return torch.randn(shape, device=device)

    return repeat_noise() if repeat else noise()


def default(val, d):
    """Return ``val`` unless it is None, otherwise ``d``, calling it first when it is a function."""
    if exists(val):
        return val
    return d() if isfunction(d) else d


def exists(val):
    """Return whether ``val`` is not None."""
    return val is not None


def identity(*args, **kwargs):
    """Return a new ``nn.Identity`` module, ignoring all arguments."""
    return nn.Identity()


def uniq(arr):
    """Return the distinct items of ``arr`` in first-seen order, as a dict keys view."""
    return dict.fromkeys(arr, True).keys()


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def ismap(x):
    """Return whether ``x`` is a 4-D tensor with more than 3 channels in dimension 1."""
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] > 3)


def isimage(x):
    """Return whether ``x`` is a 4-D tensor with 1 or 3 channels in dimension 1."""
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] == 3 or x.shape[1] == 1)


def max_neg_value(t):
    """Return the most negative finite value of the floating dtype of ``t``."""
    return -torch.finfo(t.dtype).max


def shape_to_str(x):
    """Format ``x.shape`` as a string such as ``"2x3x64x64"``."""
    shape_str = "x".join([str(x) for x in x.shape])
    return shape_str


def init_(tensor):
    """Fill ``tensor`` in place from ``U(-1/sqrt(d), 1/sqrt(d))`` with ``d = tensor.shape[-1]`` and return it."""
    dim = tensor.shape[-1]
    # Fan-in uniform, same as the improved-diffusion residual Linear init.
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# ──────────────────────────────────────────────────────────────────────────
# Activation checkpoint — flag-gated; use_reentrant=False for current PyTorch
# ──────────────────────────────────────────────────────────────────────────


ckpt = torch.utils.checkpoint.checkpoint


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        return ckpt(func, *inputs, use_reentrant=False)
    else:
        return func(*inputs)


def optimized_scale(positive_flat: torch.Tensor, negative_flat: torch.Tensor) -> torch.Tensor:
    """Return row-wise guidance projection coefficients using FP32 accumulation."""
    positive_flat = positive_flat.float()
    negative_flat = negative_flat.float()
    # Calculate dot production
    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
    # Squared norm of uncondition
    squared_norm = torch.sum(negative_flat**2, dim=1, keepdim=True) + 1e-8
    # st_star = v_cond^T * v_uncond / ||v_uncond||^2
    st_star = dot_product / squared_norm
    return st_star
