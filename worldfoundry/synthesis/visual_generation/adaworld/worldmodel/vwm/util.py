import functools
from inspect import isfunction

import torch
from einops import repeat

from worldfoundry.core.configuration.factory import (
    get_obj_from_str as _get_obj_from_str,
)
from worldfoundry.core.configuration.factory import (
    instantiate_from_config as instantiate_from_config,
)
from worldfoundry.core.utils.tensors.torch import (
    count_params as count_params,
)


def repeat_img_seq(x, num_frames):
    return repeat(x, "b ... -> (b t) ...", t=num_frames)


def disabled_train(self):
    """
    Overwrite model.train with this function to make sure train/eval mode does not change anymore.
    """

    return self


def autocast(f, enabled=True):
    def do_autocast(*args, **kwargs):
        with torch.cuda.amp.autocast(
            enabled=enabled, dtype=torch.get_autocast_gpu_dtype(), cache_enabled=torch.is_autocast_cache_enabled()
        ):
            return f(*args, **kwargs)

    return do_autocast


def partialclass(cls, *args, **kwargs):
    class NewCls(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwargs)

    return NewCls


def expand_dims_like(x, y):
    while x.dim() != y.dim():
        x = x.unsqueeze(-1)
    return x


def default(val, d):
    if val is None:
        return d() if isfunction(d) else d
    else:
        return val


def get_obj_from_str(string, reload=False, invalidate_cache=True):
    return _get_obj_from_str(string, reload=reload, invalidate_cache=invalidate_cache)


def append_dims(x, target_dims):
    """
    Appends dimensions to the end of a tensor until it has target_dims dimensions.
    """

    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"Input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]
