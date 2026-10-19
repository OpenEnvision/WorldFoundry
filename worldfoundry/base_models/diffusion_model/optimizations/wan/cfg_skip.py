"""Late-step CFG skip for Wan graphs that batch cond and uncond in one forward.

When ``cfg_skip_ratio`` is set and ``current_steps`` is in the last fraction
of the schedule, the decorator drops the uncond half of a batched input,
runs cond/uncond separately (or cond only), then duplicates the cond result
so the caller still sees a 2× batch. This is opt-in and lossy.

Camera feature caches retain separate projections for each batch slice.
Raises :exc:`ValueError` when no main input tensor is found.
"""

import inspect

import numpy as np
import torch

from ..camera_condition import CameraConditionCache
from ..static_cross_kv import StaticConditionCache


def _slice_batch(value, selection, batch_size):
    if isinstance(value, (torch.Tensor, list, tuple, np.ndarray, CameraConditionCache, StaticConditionCache)) and len(value) == batch_size:
        return value[selection]
    return value


def cfg_skip():
    """Decorator that optionally skips the uncond half of a batched CFG forward."""
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            if 'hidden_states' in kwargs and kwargs['hidden_states'] is not None:
                main_input = kwargs['hidden_states']
            elif 'x' in kwargs and kwargs['x'] is not None:
                main_input = kwargs['x']
            elif len(args) > 0:
                main_input = args[0]
            else:
                raise ValueError("No input tensor found in args or kwargs")

            bs = len(main_input)
            if bs >= 2 and self.cfg_skip_ratio is not None and self.current_steps >= self.num_inference_steps * (1 - self.cfg_skip_ratio):
                bs_half = int(bs // 2)
                new_x = main_input[bs_half:]
                new_args = [_slice_batch(arg, slice(bs_half, None), bs) for arg in args]
                new_kwargs = {k: _slice_batch(v, slice(bs_half, None), bs) for k, v in kwargs.items()}
            else:
                new_x = main_input
                new_args = args
                new_kwargs = kwargs

            sig = inspect.signature(func)
            new_bs = len(new_x)
            new_bs_half = int(new_bs // 2)
            if new_bs >= 2:
                args_i = [_slice_batch(arg, slice(new_bs_half, None), new_bs) for arg in new_args]
                kwargs_i = {k: _slice_batch(v, slice(new_bs_half, None), new_bs) for k, v in new_kwargs.items()}
                if 'cond_flag' in sig.parameters:
                    kwargs_i["cond_flag"] = True
                cond_out = func(self, *args_i, **kwargs_i)

                uncond_args_i = [_slice_batch(arg, slice(None, new_bs_half), new_bs) for arg in new_args]
                uncond_kwargs_i = {k: _slice_batch(v, slice(None, new_bs_half), new_bs) for k, v in new_kwargs.items()}
                if 'cond_flag' in sig.parameters:
                    uncond_kwargs_i["cond_flag"] = False
                uncond_out = func(self, *uncond_args_i, **uncond_kwargs_i)
                result = torch.cat([uncond_out, cond_out], dim=0)
            else:
                result = func(self, *new_args, **new_kwargs)

            if bs >= 2 and self.cfg_skip_ratio is not None and self.current_steps >= self.num_inference_steps * (1 - self.cfg_skip_ratio):
                result = torch.cat([result, result], dim=0)
            return result
        return wrapper
    return decorator
