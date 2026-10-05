"""Tensor and checkpoint helpers shared by classic diffusion networks.

Shape/cast utilities for U-Net residuals. DiT modulation lives in
``diffusion_transformer``; do not mix the two vocabularies.
"""

import math
from inspect import isfunction
import torch
import torch.utils.checkpoint
from torch import nn
import torch.distributed as dist


# ──────────────────────────────────────────────────────────────────────────
# Distributed gather and AMP — U-Net residual helpers, not DiT modulation
# ──────────────────────────────────────────────────────────────────────────


def gather_data(data, return_np=True):
    ''' gather data from multiple processes to one list '''
    data_list = [torch.zeros_like(data) for _ in range(dist.get_world_size())]
    dist.all_gather(data_list, data)  # gather not supported with NCCL
    if return_np:
        data_list = [data.cpu().numpy() for data in data_list]
    return data_list

def autocast(f):
    """Autocast.

    Args:
        f: The f.
    """
    def do_autocast(*args, **kwargs):
        """Do autocast."""
        with torch.amp.autocast(
            "cuda",
            enabled=True,
            dtype=torch.get_autocast_gpu_dtype(),
            cache_enabled=torch.is_autocast_cache_enabled(),
        ):
            return f(*args, **kwargs)
    return do_autocast


def extract_into_tensor(a, t, x_shape):
    """Extract into tensor.

    Args:
        a: The a.
        t: The t.
        x_shape: The x shape.
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


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
    """Noise like.

    Args:
        shape: The shape.
        device: The device.
        repeat: The repeat.
    """
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def default(val, d):
    """Default.

    Args:
        val: The val.
        d: The d.
    """
    if exists(val):
        return val
    return d() if isfunction(d) else d

def exists(val):
    """Exists.

    Args:
        val: The val.
    """
    return val is not None

def identity(*args, **kwargs):
    """Identity."""
    return nn.Identity()

def uniq(arr):
    """Uniq.

    Args:
        arr: The arr.
    """
    return{el: True for el in arr}.keys()

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def ismap(x):
    """Ismap.

    Args:
        x: The x.
    """
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] > 3)

def isimage(x):
    """Isimage.

    Args:
        x: The x.
    """
    if not isinstance(x,torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] == 3 or x.shape[1] == 1)

def max_neg_value(t):
    """Max neg value.

    Args:
        t: The t.
    """
    return -torch.finfo(t.dtype).max

def shape_to_str(x):
    """Shape to str.

    Args:
        x: The x.
    """
    shape_str = "x".join([str(x) for x in x.shape])
    return shape_str

def init_(tensor):
    """Init.

    Args:
        tensor: The tensor.
    """
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
