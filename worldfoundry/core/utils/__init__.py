"""Lazy public utility API with two implementation domains.

- python: object/tree helpers, text parsing and optional-dependency import guards.
- tensors: array operations, batching, shape inference, seeds and parameter counts.

Runtime, graph and concurrency helpers are owned by core.execution; media helpers
are owned by core.media; field validators are owned by core.configuration. Package
import loads no optional stacks.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

# ──────────────────────────────────────────────────────────────────────────
# Lazy name tables — resolved on first attribute access, then cached
# ──────────────────────────────────────────────────────────────────────────

_SUBMODULES = {
    "python": "worldfoundry.core.utils.python",
    "tensors": "worldfoundry.core.utils.tensors",
}

_EXPORT_MODULES = {
    "AverageMeter": "worldfoundry.core.utils.tensors.torch",
    "ClassRegistry": "worldfoundry.core.utils.python.functional_utils",
    "DDPMethodWrapper": "worldfoundry.core.utils.tensors.torch",
    "Every": "worldfoundry.core.utils.python.misc_utils",
    "LazyModule": "worldfoundry.core.utils.python.lazy_module",
    "NoopContext": "worldfoundry.core.utils.python.functional_utils",
    "NoopObject": "worldfoundry.core.utils.python.functional_utils",
    "Once": "worldfoundry.core.utils.python.misc_utils",
    "PeriodicEvent": "worldfoundry.core.utils.python.misc_utils",
    "RunningMeanStd": "worldfoundry.core.utils.tensors.torch",
    "accepts_kwargs": "worldfoundry.core.utils.python.functional_utils",
    "accepts_varargs": "worldfoundry.core.utils.python.functional_utils",
    "add_batch_dim": "worldfoundry.core.utils.tensors.arrays",
    "any_assign": "worldfoundry.core.utils.tensors.arrays",
    "any_chunk": "worldfoundry.core.utils.tensors.arrays",
    "any_concat": "worldfoundry.core.utils.tensors.arrays",
    "any_describe": "worldfoundry.core.utils.tensors.arrays",
    "any_describe_str": "worldfoundry.core.utils.tensors.arrays",
    "any_fill_": "worldfoundry.core.utils.tensors.arrays",
    "any_get_shape": "worldfoundry.core.utils.tensors.arrays",
    "any_mean": "worldfoundry.core.utils.tensors.arrays",
    "any_ones_like": "worldfoundry.core.utils.tensors.arrays",
    "any_slice": "worldfoundry.core.utils.tensors.arrays",
    "any_stack": "worldfoundry.core.utils.tensors.arrays",
    "any_to_primitive": "worldfoundry.core.utils.tensors.arrays",
    "any_transpose_first_two_axes": "worldfoundry.core.utils.tensors.arrays",
    "any_variance": "worldfoundry.core.utils.tensors.arrays",
    "any_zero_": "worldfoundry.core.utils.tensors.arrays",
    "any_zeros_like": "worldfoundry.core.utils.tensors.arrays",
    "argmax": "worldfoundry.core.utils.python.misc_utils",
    "assert_has_keys": "worldfoundry.core.utils.python.functional_utils",
    "assert_implements_method": "worldfoundry.core.utils.python.functional_utils",
    "as_list": "worldfoundry.core.utils.python.functional_utils",
    "broadcast_structures": "worldfoundry.core.utils.python.tree_utils",
    "batch_add": "worldfoundry.core.utils.tensors.batch",
    "batch_div": "worldfoundry.core.utils.tensors.batch",
    "batch_mul": "worldfoundry.core.utils.tensors.batch",
    "batch_sub": "worldfoundry.core.utils.tensors.batch",
    "call_once": "worldfoundry.core.utils.python.functional_utils",
    "check_shape": "worldfoundry.core.utils.tensors.shape",
    "chunk_seq": "worldfoundry.core.utils.tensors.arrays",
    "classify_accuracy": "worldfoundry.core.utils.tensors.torch",
    "clip_grad_norm": "worldfoundry.core.utils.tensors.torch",
    "clip_grad_value": "worldfoundry.core.utils.tensors.torch",
    "clone_model": "worldfoundry.core.utils.tensors.torch",
    "contains_rnn": "worldfoundry.core.utils.tensors.torch",
    "copy_non_leaf": "worldfoundry.core.utils.python.tree_utils",
    "count_parameters": "worldfoundry.core.utils.tensors.torch",
    "count_params": "worldfoundry.core.utils.tensors.torch",
    "deprecated": "worldfoundry.core.utils.python.functional_utils",
    "divide": "worldfoundry.core.utils.python.misc_utils",
    "enable_dict_arg": "worldfoundry.core.utils.python.functional_utils",
    "enable_kwargs": "worldfoundry.core.utils.python.functional_utils",
    "enable_list_arg": "worldfoundry.core.utils.python.functional_utils",
    "enable_varargs": "worldfoundry.core.utils.python.functional_utils",
    "env_is_true": "worldfoundry.core.utils.python.misc_utils",
    "eval_mode": "worldfoundry.core.utils.tensors.torch",
    "extract_yes_no_answer": "worldfoundry.core.utils.python.text_parsing",
    "fast_map_structure": "worldfoundry.core.utils.python.tree_utils",
    "filter_patterns": "worldfoundry.core.utils.python.misc_utils",
    "fix_random_seeds": "worldfoundry.core.utils.tensors.torch",
    "freeze_params": "worldfoundry.core.utils.tensors.torch",
    "has_batchnorms": "worldfoundry.core.utils.tensors.torch",
    "func_has_arg": "worldfoundry.core.utils.python.functional_utils",
    "func_parameters": "worldfoundry.core.utils.python.functional_utils",
    "get_batch_size": "worldfoundry.core.utils.tensors.arrays",
    "get_device": "worldfoundry.core.utils.tensors.torch",
    "get_module_device": "worldfoundry.core.utils.tensors.torch",
    "get_seed": "worldfoundry.core.utils.tensors.torch",
    "getattr_nested": "worldfoundry.core.utils.python.misc_utils",
    "getitem_nested": "worldfoundry.core.utils.python.misc_utils",
    "global_n_times": "worldfoundry.core.utils.python.misc_utils",
    "global_once": "worldfoundry.core.utils.python.misc_utils",
    "has_keys": "worldfoundry.core.utils.python.functional_utils",
    "implements_method": "worldfoundry.core.utils.python.functional_utils",
    "implements_state_dict": "worldfoundry.core.utils.tensors.torch",
    "is_array_tensor": "worldfoundry.core.utils.tensors.arrays",
    "is_mapping": "worldfoundry.core.utils.python.functional_utils",
    "is_numpy": "worldfoundry.core.utils.tensors.arrays",
    "is_sequence": "worldfoundry.core.utils.python.functional_utils",
    "is_signature_compatible": "worldfoundry.core.utils.python.functional_utils",
    "is_tensor": "worldfoundry.core.utils.tensors.arrays",
    "load_state_dict": "worldfoundry.core.utils.tensors.torch",
    "make_list": "worldfoundry.core.utils.python.functional_utils",
    "make_recursive_func": "worldfoundry.core.utils.python.functional_utils",
    "make_tuple": "worldfoundry.core.utils.python.functional_utils",
    "match_patterns": "worldfoundry.core.utils.python.misc_utils",
    "maybe_transfer_module": "worldfoundry.core.utils.tensors.torch",
    "mean_flat": "worldfoundry.core.utils.tensors.torch",
    "merge_kwargs": "worldfoundry.core.utils.python.functional_utils",
    "meta_decorator": "worldfoundry.core.utils.python.functional_utils",
    "method_decorator": "worldfoundry.core.utils.python.functional_utils",
    "multi_one_hot": "worldfoundry.core.utils.tensors.torch",
    "pack_kwargs": "worldfoundry.core.utils.python.functional_utils",
    "pack_varargs": "worldfoundry.core.utils.python.functional_utils",
    "random_derangement": "worldfoundry.core.utils.tensors.torch",
    "remove_batch_dim": "worldfoundry.core.utils.tensors.arrays",
    "safe_hash": "worldfoundry.core.utils.python.misc_utils",
    "sequential_split_dataset": "worldfoundry.core.utils.tensors.torch",
    "set_deterministic": "worldfoundry.core.utils.tensors.torch",
    "set_random_seed": "worldfoundry.core.utils.tensors.torch",
    "set_os_envs": "worldfoundry.core.utils.python.misc_utils",
    "set_requires_grad": "worldfoundry.core.utils.tensors.torch",
    "set_seed_everywhere": "worldfoundry.core.utils.tensors.torch",
    "setattr_nested": "worldfoundry.core.utils.python.misc_utils",
    "setitem_nested": "worldfoundry.core.utils.python.misc_utils",
    "shape_avgpool1d": "worldfoundry.core.utils.tensors.shape",
    "shape_avgpool2d": "worldfoundry.core.utils.tensors.shape",
    "shape_avgpool3d": "worldfoundry.core.utils.tensors.shape",
    "shape_conv1d": "worldfoundry.core.utils.tensors.shape",
    "shape_conv2d": "worldfoundry.core.utils.tensors.shape",
    "shape_conv3d": "worldfoundry.core.utils.tensors.shape",
    "shape_convnd": "worldfoundry.core.utils.tensors.shape",
    "shape_maxpool1d": "worldfoundry.core.utils.tensors.shape",
    "shape_maxpool2d": "worldfoundry.core.utils.tensors.shape",
    "shape_maxpool3d": "worldfoundry.core.utils.tensors.shape",
    "shape_poolnd": "worldfoundry.core.utils.tensors.shape",
    "shape_slice": "worldfoundry.core.utils.tensors.shape",
    "shape_transpose_conv1d": "worldfoundry.core.utils.tensors.shape",
    "shape_transpose_conv2d": "worldfoundry.core.utils.tensors.shape",
    "shape_transpose_conv3d": "worldfoundry.core.utils.tensors.shape",
    "shape_transpose_convnd": "worldfoundry.core.utils.tensors.shape",
    "state_dict_class": "worldfoundry.core.utils.python.functional_utils",
    "stack_or_pad_tensors": "worldfoundry.core.utils.tensors.batch",
    "stack_sequence_fields": "worldfoundry.core.utils.python.tree_utils",
    "tensor_hash": "worldfoundry.core.utils.tensors.torch",
    "temporal_feature_consistency": "worldfoundry.core.utils.tensors.torch",
    "tie_weights": "worldfoundry.core.utils.tensors.torch",
    "to_state_dict": "worldfoundry.core.utils.tensors.torch",
    "torch_compute_stats": "worldfoundry.core.utils.tensors.torch",
    "torch_flatten_indices": "worldfoundry.core.utils.tensors.torch",
    "torch_multi_index_select": "worldfoundry.core.utils.tensors.torch",
    "torch_normalize": "worldfoundry.core.utils.tensors.torch",
    "tree_assign_at_path": "worldfoundry.core.utils.python.tree_utils",
    "tree_value_at_path": "worldfoundry.core.utils.python.tree_utils",
    "unfreeze_params": "worldfoundry.core.utils.tensors.torch",
    "unstack_sequence_fields": "worldfoundry.core.utils.python.tree_utils",
    "unwrap_ddp_model": "worldfoundry.core.utils.tensors.torch",
    "update_soft_params": "worldfoundry.core.utils.tensors.torch",
    "weight_init": "worldfoundry.core.utils.tensors.torch",
}


def __getattr__(name: str) -> Any:
    """Import the owning submodule once and cache the symbol on this module.

    Failure: :exc:`AttributeError` when ``name`` is in neither table. Successful
    lookups write into ``globals()`` so later access skips ``import_module``.
    """
    module_name = _SUBMODULES.get(name) or _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = module if name in _SUBMODULES else getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose already-materialized globals plus every lazy ``__all__`` name."""
    return sorted({*globals(), *__all__})


__all__ = sorted({*_SUBMODULES, *_EXPORT_MODULES})
