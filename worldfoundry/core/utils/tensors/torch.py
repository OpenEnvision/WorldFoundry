"""PyTorch utilities: seeds, checkpoints, DDP unwrap, and module helpers.

Responsibility
    Process-wide seeds / determinism, checkpoint load/save, DDP unwrap,
    freeze/clip, and small training meters.

Boundaries
    Training-oriented. Inference Graph capture is ``inference_graph``;
    device placement for VRAM is ``worldfoundry.core.vram``. Nested
    device / state-dict walks require ``dm_tree``.

Public surface
    Seed helpers, :class:`eval_mode`, :class:`DDPMethodWrapper`,
    :func:`to_state_dict` / :func:`load_state_dict`, :class:`RunningMeanStd`,
    :class:`AverageMeter`, and the other names in this module's callers.
"""

from __future__ import annotations

import os
import random
import time
from copy import deepcopy
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from typing_extensions import Literal

from worldfoundry.core.io.filesystem.file_utils import f_join
from worldfoundry.core.observability.formatting import to_readable_count_str
from worldfoundry.core.utils.python.functional_utils import assert_implements_method, implements_method
from worldfoundry.core.utils.python.tree_utils import _require_tree, tree_value_at_path


# ──────────────────────────────────────────────────────────────────────────
# Init / seeds — process-wide; PYTHONHASHSEED only affects child processes
# ──────────────────────────────────────────────────────────────────────────


def weight_init(m):
    """Custom weight initialization function for Conv2D, ConvTranspose2D, and Linear layers.

    Applies orthogonal initialization to weight parameters. Sets bias parameters to 0.0
    if they exist. Commonly used in deep reinforcement learning or high-precision policies
    to stabilize the starting gradients.
    """
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


def get_seed(
    seed: Union[int, str, None],
    handle_invalid_seed: Literal["none", "system", "raise"] = "none",
) -> Optional[int]:
    """Resolves and standardizes random seed inputs.

    Translates special verbal seed flags like `"system"` into high-entropy,
    scrambled integers based on high-precision system time to prevent collisions.

    Args:
        seed:
            - "system": Generates a scrambled integer from high-precision system time.
            - None or negative int: Deemed invalid, cascades to `handle_invalid_seed`.
            - Non-negative int: Returned directly as a valid seed.
        handle_invalid_seed: Controls action taken for invalid/missing seed parameters:
            - "none": Return None gracefully.
            - "system": Automatically generate and return a scrambled system-time integer.
            - "raise": Raise a ValueError exception immediately.
    """
    handle_invalid_seed = handle_invalid_seed.lower()
    assert handle_invalid_seed in ["none", "system", "raise"]
    if isinstance(seed, str):
        assert seed in ["system"]
        invalid = False
    else:
        assert seed is None or isinstance(seed, int)
        invalid = seed is None or seed < 0

    if seed == "system" or invalid and handle_invalid_seed == "system":
        # Scramble timestamp bits to avoid adjacent seeds if multiple workers initialize concurrently.
        # https://stackoverflow.com/questions/27276135/python-random-system-time-seed
        t = int(time.time() * 100000)
        return ((t & 0xFF000000) >> 24) + ((t & 0x00FF0000) >> 8) + ((t & 0x0000FF00) << 8) + ((t & 0x000000FF) << 24)
    elif invalid:
        if handle_invalid_seed == "none":
            return None
        elif handle_invalid_seed == "raise":
            raise ValueError(f'Invalid random seed: {seed}, must be a non-negative integer or "system"')
        else:
            raise NotImplementedError
    else:
        return seed


def _env_truthy(name: str) -> bool:
    """True for ``1`` / ``true`` / ``yes`` / ``on``; missing or other spellings are False."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def set_deterministic(flag: bool = True):
    """Enforces absolute reproducibility / determinism across PyTorch and CuDNN layers.

    Sets system environment flags for CUBLAS workspace caching and disables CuDNN auto-benchmarking
    to avoid non-deterministic execution paths during policy rollout evaluation.

    Also honoured process-wide via ``WORLDFOUNDRY_DETERMINISTIC=1`` (XC-23).
    """
    if not flag:
        return

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["HOROVOD_FUSION_THRESHOLD"] = "0"
    import torch.backends.cudnn as cudnn

    cudnn.deterministic = True
    cudnn.benchmark = False
    if hasattr(torch, "use_deterministic_algorithms"):
        # only available in PyTorch >= 1.9
        torch.use_deterministic_algorithms(True)
    elif hasattr(torch, "set_deterministic"):
        # only available in PyTorch >= 1.7
        torch.set_deterministic(True)


def apply_deterministic_from_env() -> bool:
    """Enable :func:`set_deterministic` when ``WORLDFOUNDRY_DETERMINISTIC`` is set."""

    if not _env_truthy("WORLDFOUNDRY_DETERMINISTIC"):
        return False
    set_deterministic(True)
    return True


def set_seed_everywhere(
    seed: Optional[Union[int, str]],
    deterministic=False,
    set_tensorflow=False,
    handle_invalid_seed: Literal["none", "system", "raise"] = "none",
) -> Optional[int]:
    """Inoculates random seeds globally across all standard runtime backends.

    Binds random seeds across Python builtin random, NumPy, PyTorch (CPU and CUDA),
    and optionally TensorFlow to guarantee consistency across multi-worker evaluators.

    References:
    - https://github.com/NVIDIA/framework-determinism/blob/master/pytorch.md
    - https://pytorch.org/docs/stable/notes/randomness.html
    - CUBLAS env var:
        https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility

    Args:
        seed: Target seed integer or `"system"` flag.
        deterministic: If True, enforces strict algorithmic determinism (slowing down some kernels).
        set_tensorflow: If True, attempts to seed active TF runtimes.
        handle_invalid_seed: Strategy for resolving invalid/None seeds.
    """
    if not deterministic:
        deterministic = _env_truthy("WORLDFOUNDRY_DETERMINISTIC")
    set_deterministic(deterministic)

    seed = get_seed(seed, handle_invalid_seed=handle_invalid_seed)
    if seed is None:
        return None

    # Note: PYTHONHASHSEED cannot change str/bytes hashing for the *current*
    # interpreter (it is read once at startup); setting it here only makes
    # child processes inherit the seed.
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if set_tensorflow:
        try:
            import tensorflow as tf

            tf.random.set_seed(seed)
        except ImportError:
            pass
    return seed


def set_random_seed(seed: int, by_rank: bool = False) -> int:
    """Set Python, NumPy, and PyTorch seeds with optional distributed-rank offset.

    Thin wrapper over :func:`set_seed_everywhere` (the canonical entry point).
    When the process group is unavailable, ``by_rank`` silently falls back to
    the unoffset seed.
    """

    resolved_seed = int(seed)
    if (
        by_rank
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        resolved_seed += int(torch.distributed.get_rank())
    set_seed_everywhere(resolved_seed)
    return resolved_seed


def mean_flat(tensor: torch.Tensor) -> torch.Tensor:
    """Average over all non-batch dimensions."""

    return tensor.mean(dim=list(range(1, tensor.ndim)))


def temporal_feature_consistency(features: torch.Tensor) -> torch.Tensor:
    """Average non-negative adjacent/first-frame cosine consistency on device."""

    if features.ndim != 2:
        raise ValueError(f"features must have shape [frames, channels], got {tuple(features.shape)}")
    if features.shape[0] < 2:
        raise ValueError("at least two frame features are required")
    adjacent = torch.nn.functional.cosine_similarity(features[:-1], features[1:], dim=-1).clamp_min_(0)
    first = torch.nn.functional.cosine_similarity(features[0:1], features[1:], dim=-1).clamp_min_(0)
    return ((adjacent + first) * 0.5).mean()


# ──────────────────────────────────────────────────────────────────────────
# Eval / device / IO — restore train flags; default map_location is CPU
# ──────────────────────────────────────────────────────────────────────────


class eval_mode(object):
    """Temporarily ``train(False)`` one or more modules; restore on exit."""

    def __init__(self, *models):
        """Record modules; flags are snapshotted on enter, not here."""
        self.models = models

    def __enter__(self):
        """Switch each module to eval; remember prior ``training`` bits."""
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        """Restore prior flags; never suppress exceptions."""
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def get_device(x, strict: bool = False) -> int:
    """
    Args:
        x: can be any arbitrary nested structure of np array and torch tensor
        strict: True to check all batch sizes are the same
    """
    tree = _require_tree()
    xs = tree.flatten(x)

    def _get_device(x):
        """Tensor device, first-parameter device for modules, else ``None``."""
        if torch.is_tensor(x):
            return x.device
        elif isinstance(x, nn.Module):
            return get_module_device(x)
        else:
            return None

    if strict:
        devices = [_get_device(x) for x in xs]
        assert all(b == devices[0] for b in devices), f"devices must all be the same in nested structure: {devices}"
        return devices[0]
    else:
        return _get_device(xs[0])


def load_torch(*fpath: str, map_location="cpu") -> dict:
    """
    Default maps to "cpu"
    """
    from worldfoundry.core.model_loading import load_torch_checkpoint

    fpath = str(f_join(fpath))
    try:
        return load_torch_checkpoint(fpath, map_location=map_location)
    except RuntimeError as e:
        raise RuntimeError(f"{e}\n\n --- Error loading {fpath}")


def save_torch(D, *fpath):
    """
    Supports both (D, fpath) and (fpath, D) arg order, as long as one of them is a str
    """
    if isinstance(D, str):
        assert not isinstance(fpath, str), "Either torch_save(D, fpath) or torch_save(fpath, D)"
        # ``fpath`` is a varargs tuple here; unwrap the payload so the swapped
        # call order saves the object itself instead of a 1-tuple.
        if len(fpath) != 1:
            raise ValueError("save_torch(fpath, D) expects exactly one object to save")
        fpath, D = (D,), fpath[0]
    torch.save(D, str(f_join(fpath)))


# Aliases for consistency with load_pickle, load_text, load_json/yaml, etc.
torch_load = load_torch
torch_save = save_torch
dump_torch = save_torch


def torch_compute_stats(x, precision: int = 2):
    """Human-readable mean/std/median/min/max after a float32 cast."""
    x = x.to(dtype=torch.float32)
    return (
        f"mean|std: {torch.mean(x):.{precision}f} +/- {torch.std(x):.{precision}f}, "
        f"median: {torch.median(x):.{precision}f}, "
        f"max: {torch.max(x):.{precision}f}, min: {torch.min(x):.{precision}f}"
    )


def tensor_hash(x: torch.Tensor, mode: str = "mean"):
    """Cheap abs-mean or abs-sum fingerprint for debug equality, not a crypto hash."""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    x = x.float().abs()
    if mode == "sum":
        x = x.sum()
    elif mode == "mean":
        x = x.mean()
    else:
        raise NotImplementedError
    return float(x)


def torch_flatten_indices(indices: torch.Tensor, shape: Tuple[int]):
    """
    Convert M dim indices to 1D indices with the given shape

    Args:
        indices: BxM, batch_size x M-dimensional
    """
    offsets = np.array(shape)  # e.g. [3, 4, 5, 6]
    offsets = np.append(offsets[1:], 1)  # [4, 5, 6, 1]
    offsets = np.cumprod(offsets[::-1])[::-1]  # [4*5*6, 5*6, 6, 1]
    offsets = torch.tensor(offsets.copy(), dtype=torch.long)
    assert offsets.size() == (len(shape),)
    return (indices * offsets.to(device=indices.device)).sum(dim=1)


def torch_multi_index_select(x: torch.Tensor, indices: torch.Tensor):
    """
    Args:
        x: N dim
        indices: [B x M], M <= N, will select the first M-D from N-D

    Returns:
        (N - M + 1) dim
    """
    assert indices.ndim == 2
    B, idx_dim = indices.size()
    x_shape = x.size()
    assert len(x_shape) >= idx_dim
    remainder_dim = len(x_shape) - idx_dim
    if remainder_dim == 0:
        x = torch.flatten(x)
    else:
        x = x.view(-1, *x_shape[-remainder_dim:])  # flatten the first M dims
    # convert indices to a 1D flattened array
    indices = torch_flatten_indices(indices, x_shape[:idx_dim])
    selected = x[indices]
    return selected


# ──────────────────────────────────────────────────────────────────────────
# Module ops — freeze / clip / DDP unwrap; first parameter defines device
# ──────────────────────────────────────────────────────────────────────────


def set_requires_grad(model, requires_grad):
    """Set ``requires_grad`` on a tensor or every parameter of a module."""
    if torch.is_tensor(model):
        model.requires_grad = requires_grad
    else:
        for param in model.parameters():
            param.requires_grad = requires_grad


def freeze_params(model):
    """Disable grads and, for modules, switch to eval (BN / dropout off)."""
    set_requires_grad(model, False)
    if not torch.is_tensor(model):
        model.eval()


def unfreeze_params(model):
    """Enable grads and, for modules, switch to train."""
    set_requires_grad(model, True)
    if not torch.is_tensor(model):
        model.train()


def clip_grad_value(model, max_value):
    """Clamp per-element grads in place; no return value."""
    with torch.no_grad():
        nn.utils.clip_grad_value_(model.parameters(), max_value)


def clip_grad_norm(model, max_norm, norm_type=2):
    """
    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    with torch.no_grad():
        return nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm, norm_type=norm_type)


def implements_state_dict(object, requires_load_method: bool = False):
    """True if ``state_dict`` exists; optionally also require ``load_state_dict``."""
    cond = implements_method(object, "state_dict")
    if requires_load_method:
        return cond and implements_method(object, "load_state_dict")
    else:
        return cond


def unwrap_ddp_model(model):
    """Peel one DDP/wrapper ``.module`` when it is the sole child."""
    if hasattr(model, "module") and len(list(model.children())) == 1:
        model = model.module
    return model


class DDPMethodWrapper(nn.Module):
    """
    Wraps another module's method as forward(), because DDP only works on forward()
    This module can be wrapped with DDP and directly called.
    It will not save any extra parameters
    """

    def __init__(self, net: nn.Module, method_name: str):
        """Bind ``net.method_name``; the method must already exist."""
        super().__init__()
        self.net = net
        assert_implements_method(net, method_name)
        self._method_name = method_name

    def forward(self, *args, **kwargs):
        """Dispatch to the wrapped method so DDP can allreduce that path."""
        return getattr(self.net, self._method_name)(*args, **kwargs)

    def state_dict(self):
        """Empty on purpose — the inner module owns the real weights."""
        return {}


def to_state_dict(objects, to_cpu: bool = False, copy: bool = False, unwrap_ddp: bool = False):
    """
    Anything that has state_dict() method, e.g. nn.Module, Optimizer, LRScheduler, etc.

    Args:
        to_cpu: True to copy to CPU. The original tensors will still be on GPU.
        copy: takes effect if and only if to_cpu is False
    """

    def _transfer(x):
        """Detach tensors and optionally copy to CPU or clone on-device."""
        if torch.is_tensor(x):
            x = x.detach()
            if to_cpu:
                return x.cpu()
            elif copy:
                return x.clone()
        return x

    def _to_state_dict(m):
        """Call ``state_dict`` when present; otherwise treat ``m`` as a leaf tensor."""
        if implements_state_dict(m):
            # DDP unwrapping is just a preprocessing step; every object with a
            # state_dict() method (Optimizer, LRScheduler, plain nn.Module)
            # goes through the same transfer path.
            if isinstance(m, nn.Module) and unwrap_ddp:
                m = unwrap_ddp_model(m)
            tree = _require_tree()
            return tree.map_structure(_transfer, m.state_dict())
        else:
            return _transfer(m)

    tree = _require_tree()
    return tree.map_structure(_to_state_dict, objects)


def load_state_dict(objects, states, strip_prefix=None, strict=False):
    """
    Args:
        strict: objects and states must match exactly
        strip_prefix: only match the keys that have the prefix, and strip it
    """

    def _load(paths, obj):
        """Load the matching subtree; skip missing paths unless ``strict``."""
        if not implements_method(obj, "load_state_dict"):
            raise ValueError(f"Object {type(obj)} does not support load_state_dict() method")
        try:
            state = tree_value_at_path(states, paths)
        except ValueError:  # paths do not exist in `states` structure
            if strict:
                raise
            else:
                return
        if strip_prefix:
            assert isinstance(strip_prefix, str)
            state = {k[len(strip_prefix) :]: v for k, v in state.items() if k.startswith(strip_prefix)}
        if isinstance(obj, nn.Module):
            return obj.load_state_dict(state, strict=strict)
        else:
            return obj.load_state_dict(state)

    tree = _require_tree()
    return tree.map_structure_with_path(_load, objects)


def count_parameters(model, verbose: bool = False):
    """Sum ``numel`` over parameters (includes frozen); optionally log the count."""
    count = sum(parameter.numel() for parameter in model.parameters())
    if verbose:
        from worldfoundry.core.distributed.runtime.logging import log

        log.info("parameter count: {:,}", count)
    return count


def readable_count_parameters(model, precision: int = 2):
    """Human-readable parameter count (e.g. ``12.3M``)."""
    return to_readable_count_str(count_parameters(model), precision=precision)


def get_module_device(model):
    """
    Returns:
        first model parameter's device
    """
    return next(model.parameters()).device


def maybe_transfer_module(model, device):
    """
    Transfer a module to another device if and only if they are on different devices.
    Assumes that the module's first parameter determines the module device, i.e.
    no model parallelism.

    Returns:
        True if module is transferred to a different device, False otherwise
    """
    if device is None:
        return False
    device = torch.device(device)
    if get_module_device(model) != device:
        model.to(device=device)
        return True
    else:
        return False


def clone_model(model):
    """Deep-copy a module onto its current device without sharing storage."""
    with torch.no_grad():
        new_model = deepcopy(model).to(get_module_device(model))
    # new_model.load_state_dict(model.state_dict())
    return new_model


def update_soft_params(net, target_net, tau):
    """Polyak average ``target = tau * net + (1-tau) * target`` in place."""
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def tie_weights(src, trg):
    """Alias ``trg.weight`` / ``bias`` to ``src`` (same type required). Deprecated."""
    # TODO deprecate this
    assert type(src) is type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


def torch_normalize(tensor: torch.Tensor, mean, std, inplace=False):
    """
    Adapted from https://pytorch.org/docs/stable/_modules/torchvision/transforms/functional.html#normalize

    Normalize a tensor image with mean and standard deviation.

    .. note::
        This transform acts out of place by default, i.e., it does not mutates the input tensor.

    See :class:`~torchvision.transforms.Normalize` for more details.

    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation inplace.

    Returns:
        Tensor: Normalized Tensor image.
    """
    if not torch.is_tensor(tensor):
        raise TypeError("tensor should be a torch tensor. Got {}.".format(type(tensor)))

    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    if (std == 0).any():
        raise ValueError(f"std evaluated to zero after conversion to {dtype}, leading to division by zero.")
    if mean.ndim == 1:
        mean = mean[:, None, None]
    if std.ndim == 1:
        std = std[:, None, None]
    tensor.sub_(mean).div_(std)
    return tensor


def contains_rnn(net: nn.Module) -> bool:
    """True if any submodule is an ``nn.RNNBase`` (LSTM / GRU / RNN)."""
    for m in net.modules():
        if isinstance(m, nn.RNNBase):
            return True
    return False


def multi_one_hot(x, num_classes: List[int], to_float=True):
    """
    Concatenates multiple one-hot matrices, useful for embedding MultiDiscrete action space

    Args:
        x: torch.long, [*N, D]
        num_classes: list len == D, match the last dim of x

    Returns:
        [*N, sum(num_classes)]
    """
    from torch.nn.functional import one_hot

    assert x.dtype == torch.long
    assert x.dim() >= 2, x.size()
    assert len(num_classes) == x.size(-1), f"{len(num_classes)} != {x.size(1)}"
    result = torch.cat([one_hot(t, c) for t, c in zip(torch.unbind(x, dim=-1), num_classes)], dim=-1)
    if to_float:
        return result.float()
    else:
        return result


def _random_derangement(n):
    """Sattolo-style rejection sample of a permutation with no fixed points."""
    while True:
        v = [i for i in range(n)]
        for j in range(n - 1, -1, -1):
            p = random.randint(0, j)
            if v[p] == j:
                break
            else:
                v[j], v[p] = v[p], v[j]
        else:
            if v[0] != 0:
                return tuple(v)


def random_derangement(n, format: Literal["list", "numpy", "torch"] = "torch"):
    """
    Early refusal algorithm, described at
    https://stackoverflow.com/questions/25200220/generate-a-random-derangement-of-a-list
    Derangement is permuation without fixed point, useful for constructing negative
    pairs in contrastive learning.
    """
    assert format in ["list", "numpy", "torch"]
    D = _random_derangement(n)
    if format == "list":
        return D
    elif format == "numpy":
        return np.array(D, dtype=np.int64)
    elif format == "torch":
        return torch.tensor(D, dtype=torch.long)
    else:
        raise NotImplementedError(f"Unknown format {format}")


def classify_accuracy(
    output,
    target,
    topk: Union[int, List[int], Tuple[int]] = 1,
    mask=None,
    reduction="mean",
    scale_100=False,
):
    """
    Computes the accuracy over the k top predictions for the specified values of k.
    Accuracy is a float between 0.0 and 1.0

    Args:
        topk: if int, return a single acc. If tuple, return a tuple of accs
        mask: shape [batch_size,], binary mask of whether to include this sample or not
    """
    if isinstance(topk, int):
        topk = [topk]
        is_int = True
    else:
        is_int = False

    batch_size = target.size(0)
    assert output.size(0) == batch_size
    if mask is not None:
        assert mask.dim() == 1
        assert mask.size(0) == batch_size

    assert reduction in ["sum", "mean", "none"]
    if reduction != "mean":
        assert not scale_100, f"reduce={reduction} does not support scale_100=True"

    with torch.no_grad():
        maxk = max(topk)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        if mask is not None:
            correct = mask * correct

        mult = 100.0 if scale_100 else 1.0
        res = []
        for k in topk:
            correct_k = correct[:k].int().sum(dim=0)
            if reduction == "mean":
                if mask is not None:
                    # fmt: off
                    res.append(
                        float(correct_k.float().sum().mul_(mult / mask.sum().item()).item())
                    )
                    # fmt: on
                else:
                    res.append(float(correct_k.float().sum().mul_(mult / batch_size).item()))
            elif reduction == "sum":
                res.append(int(correct_k.sum().item()))
            elif reduction == "none":
                res.append(correct_k)
            else:
                raise NotImplementedError(f"Unknown reduce={reduction}")

    if is_int:
        assert len(res) == 1, "INTERNAL"
        return res[0]
    else:
        return res


def merge_dict_list(dict_list):
    """Stack 0-D tensors and cat N-D tensors across a list of homogeneous dicts.

    Non-tensor values are taken from the first item only. A single-item list
    is returned unchanged.
    """
    if len(dict_list) == 1:
        return dict_list[0]

    merged_dict = {}
    for k, v in dict_list[0].items():
        if isinstance(v, torch.Tensor):
            if v.ndim == 0:
                merged_dict[k] = torch.stack([d[k] for d in dict_list], dim=0)
            else:
                merged_dict[k] = torch.cat([d[k] for d in dict_list], dim=0)
        else:
            # for non-tensor values, we just copy the value from the first item
            merged_dict[k] = v
    return merged_dict


def sequential_split_dataset(dataset: torch.utils.data.Dataset, split_portions: list[float]):
    """
    Split a dataset into multiple datasets, each with a different portion of the
    original dataset. Uses torch.utils.data.Subset.
    """
    from itertools import accumulate

    assert len(split_portions) > 0, "split_portions must be a non-empty list"
    assert all(0.0 <= p <= 1.0 for p in split_portions), f"{split_portions=}"
    assert abs(sum(split_portions) - 1.0) < 1e-6, f"{sum(split_portions)=} != 1.0"
    L = len(dataset)
    assert L > 0, "dataset must be non-empty"
    # split the list with proportions
    lengths = [int(p * L) for p in split_portions]
    # make sure the last split fills the full dataset
    lengths[-1] += L - sum(lengths)
    indices = list(range(L))

    return [
        torch.utils.data.Subset(dataset, indices[offset - length : offset])
        for offset, length in zip(accumulate(lengths), lengths)
    ]


class RunningMeanStd:
    """Welford / parallel variance over a stream of numpy or torch batches.

    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    Variance is always the biased (``unbiased=False``) estimator so numpy and
    torch stay aligned.
    """

    def __init__(self):
        """Start empty; first :meth:`update` allocates mean/var like the batch."""
        self._mean = None
        self._var = None
        self._count = 0

    @property
    def mean(self):
        """Running mean, or ``None`` before the first update."""
        return self._mean

    @property
    def var(self):
        """Running biased variance, or ``None`` before the first update."""
        return self._var

    @property
    def std(self):
        """``sqrt(var)`` on the same array/tensor type as the stream."""
        if isinstance(self._var, np.ndarray):
            return np.sqrt(self._var)
        else:
            return self._var.sqrt()

    @property
    def count(self):
        """Number of samples absorbed so far (leading-dim sum)."""
        return self._count

    def update(self, values: np.ndarray | torch.Tensor) -> None:
        """Fold a batch via its mean/var; leading dim is the sample axis."""
        from worldfoundry.core.utils.tensors.arrays import any_mean, any_variance, get_batch_size

        batch_mean = any_mean(values, dim=0)
        # our running var calculation currently only supports unbiased=False
        batch_var = any_variance(values, dim=0, unbiased=False)
        batch_count = get_batch_size(values)
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(
        self,
        batch_mean: np.ndarray | torch.Tensor,
        batch_var: np.ndarray | torch.Tensor,
        batch_count: int,
    ) -> None:
        """Combine precomputed moments; ``batch_count`` must be positive after add."""

        is_tensor = torch.is_tensor(batch_mean)
        _zeros = batch_mean.new_zeros if is_tensor else np.zeros
        if self._mean is None:
            self._mean = _zeros(any_get_shape(batch_mean))
        if self._var is None:
            self._var = _zeros(any_get_shape(batch_var)) + 1.0

        delta = batch_mean - self._mean
        tot_count = self._count + batch_count
        assert tot_count > 0, "count must be > 0"

        new_mean = self._mean + delta * batch_count / tot_count
        m_a = self._var * self._count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta * delta * self._count * batch_count / tot_count
        new_var = m_2 / tot_count

        self._mean = new_mean
        self._var = new_var
        self._count = tot_count


# ──────────────────────────────────────────────────────────────────────────
# Compatibility wrappers — prefer set_seed_everywhere / has_batchnorms
# ──────────────────────────────────────────────────────────────────────────


def fix_random_seeds(seed=31):
    """Compatibility wrapper for the canonical :func:`set_seed_everywhere`."""

    return set_seed_everywhere(int(seed))


def has_batchnorms(model):
    """Check whether a model contains any BatchNorm layers."""
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for _, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name="", fmt="f"):
        """Optional ``name`` prefixes ``str()``; ``fmt`` is a format spec or empty."""
        self._name = name
        self._fmt = fmt
        self.reset()

    def reset(self):
        """Clear sum and count so a new epoch does not inherit the last."""
        self._sum = 0.0
        self._count = 0.0

    @torch.no_grad()
    def update(self, value, n=1):
        """Add ``value * n``; tensors are detached so the meter stays off-graph."""
        if torch.is_tensor(value):
            value = value.detach()
        self._sum += value * n
        self._count += n

    @torch.no_grad()
    def compute(self):
        """Mean so far; divide-by-zero if ``update`` was never called."""
        return float(self._sum / self._count)

    def __float__(self):
        """Same as :meth:`compute` for ``float(meter)``."""
        return self.compute()

    def __str__(self):
        """Format the current mean; include ``name`` when set."""
        if self._fmt:
            s = f"{float(self):{self._fmt}}"
        else:
            s = str(float(self))
        if self._name:
            return f"{self._name}: {s}"
        return s
