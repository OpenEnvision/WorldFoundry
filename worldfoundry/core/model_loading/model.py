"""High-level model construction with optional VRAM management and disk mapping.

:func:`load_model` constructs a class, assigns checkpoint weights, and
optionally installs :func:`~worldfoundry.core.vram.layers.enable_vram_management`.
:func:`load_model_with_disk_offload` keeps inactive weights on a
:class:`~worldfoundry.core.vram.disk_map.DiskMap`. ZeRO-3 uses its own
assignment path.

transformers is imported only for DeepSpeed ZeRO-3; a plain torch load
must not require that package.

Not this module: file deserialization
(:mod:`worldfoundry.core.model_loading.checkpoints.file`), Hydra instantiation
(:mod:`.factory`), or LoRA merge (:mod:`.lora`).
"""

import contextlib

import torch

from worldfoundry.core.model_loading.checkpoints.file import load_state_dict
from worldfoundry.core.vram.disk_map import DiskMap
from worldfoundry.core.vram.initialization import skip_model_initialization
from worldfoundry.core.vram.layers import enable_vram_management

# ──────────────────────────────────────────────────────────────────────────
# ZeRO-3 probe — transformers is optional; missing install means "not enabled"
# ──────────────────────────────────────────────────────────────────────────


def is_deepspeed_zero3_enabled() -> bool:
    """Lazy proxy for ``transformers.integrations.is_deepspeed_zero3_enabled``.

    Returns False when transformers is not installed: the ZeRO-3 loading path
    is only reachable through the transformers integration in the first place.
    """

    try:
        from transformers.integrations import is_deepspeed_zero3_enabled as _impl
    except ImportError:
        return False
    return _impl()


# ──────────────────────────────────────────────────────────────────────────
# Disk-offload buffers — parameters stay lazy; uninitialized empty() must not remain
# ──────────────────────────────────────────────────────────────────────────


def _restore_checkpoint_buffers(
    model: torch.nn.Module,
    disk_map: DiskMap,
    *,
    torch_dtype: torch.dtype,
) -> None:
    """Materialize checkpoint-backed buffers before installing disk wrappers.

    ``skip_model_initialization`` redirects parameters to ``meta`` but leaves
    buffers on CPU.  Buffers created with ``torch.empty`` therefore contain
    uninitialized data unless they are explicitly restored from the
    checkpoint.  Disk-backed wrappers load parameters lazily, so a normal
    ``load_state_dict`` call is intentionally unavailable in this path.

    Only persistent buffers present in the converted checkpoint mapping are
    replaced.  Deterministic, non-checkpoint buffers keep the values produced
    by their constructors.
    """

    for name, current in model.named_buffers():
        if name not in disk_map:
            continue
        value = disk_map[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"checkpoint buffer {name!r} is not a tensor")
        if current.shape != value.shape:
            raise ValueError(
                f"checkpoint buffer {name!r} has shape {tuple(value.shape)}, "
                f"expected {tuple(current.shape)}"
            )
        target_dtype = torch_dtype if value.is_floating_point() or value.is_complex() else value.dtype
        target_device = current.device if current.device.type != "meta" else torch.device("cpu")
        owner_name, separator, local_name = name.rpartition(".")
        owner = model.get_submodule(owner_name) if separator else model
        owner._buffers[local_name] = value.to(device=target_device, dtype=target_dtype)


# ──────────────────────────────────────────────────────────────────────────
# Construct + assign — wrap vs DiskMap vs full state dict; ZeRO-3 is separate
# ──────────────────────────────────────────────────────────────────────────


def load_model(
    model_class,
    path,
    config=None,
    torch_dtype=torch.bfloat16,
    device="cpu",
    state_dict_converter=None,
    use_disk_map=False,
    module_map=None,
    vram_config=None,
    vram_limit=None,
    state_dict=None,
    strict=True,
    post_load_hook=None,
):
    """Construct a model, assign checkpoint weights, and finalize inference placement.

    Args:
        model_class: PyTorch module class to instantiate.
        path: Checkpoint path consumed by ``load_state_dict`` or ``DiskMap``.
        config: Keyword arguments passed to ``model_class``.
        torch_dtype: Final model dtype.
        device: Final model device when fine-grained VRAM management is absent.
        state_dict_converter: Optional key/shape converter applied before assignment.
        use_disk_map: Read parameter tensors lazily instead of loading a full state dict.
        module_map: Source-class to VRAM-wrapper mapping. Enabling it delegates
            placement to ``enable_vram_management``.
        vram_config: Offload/onload/preparing/computation placement dictionary.
        vram_limit: Optional used-memory limit in GiB for wrappers.
        state_dict: Already-loaded weights; takes precedence over ``path``.
        strict: Require an exact parameter match.  When false, construct a
            materialized module, retain its initialization for missing keys,
            and call ``initialize(missing_keys)`` when the model provides it.
        post_load_hook: Optional in-place mutation called after checkpoint
            assignment and before VRAM wrappers or placement are installed.

    Returns:
        An eval-mode model with weights assigned.

    Notes:
        Construction uses meta-device initialization where possible. DeepSpeed
        ZeRO-3 receives its specialized state-dict assignment path.
    """
    config = {} if config is None else config
    if not strict and module_map is not None:
        raise ValueError("non-strict loading is not supported with wrapped VRAM modules")
    init_contexts = get_init_context(torch_dtype=torch_dtype, device=device) if strict else []
    try:
        # Equivalent of transformers.utils.ContextManagers without the import.
        with contextlib.ExitStack() as init_stack:
            for init_context in init_contexts:
                init_stack.enter_context(init_context)
            model = model_class(**config)
    except NotImplementedError as exc:
        # Some third-party-compatible modules move their parameters inside
        # ``__init__``.  That is incompatible with our meta-parameter
        # registration hook because PyTorch cannot copy an unmaterialized
        # tensor to CPU/CUDA.  Retry only this precise incompatibility with a
        # normal allocation; all other construction errors must remain loud.
        if is_deepspeed_zero3_enabled() or "Cannot copy out of meta tensor" not in str(exc):
            raise
        model = model_class(**config)
    # What is `module_map`?
    # This is a module mapping table for VRAM management.
    if module_map is not None:
        devices = [
            vram_config["offload_device"],
            vram_config["onload_device"],
            vram_config["preparing_device"],
            vram_config["computation_device"],
        ]
        device = [d for d in devices if d != "disk"][0]
        dtypes = [
            vram_config["offload_dtype"],
            vram_config["onload_dtype"],
            vram_config["preparing_dtype"],
            vram_config["computation_dtype"],
        ]
        dtype = [d for d in dtypes if d != "disk"][0]
        if vram_config["offload_device"] != "disk":
            # CPU/GPU offload: materialize once, then wrap. Disk offload cannot
            # run post_load_hook — there is no resident state dict to mutate.
            if state_dict is None:
                state_dict = DiskMap(path, device, torch_dtype=dtype)
            if state_dict_converter is not None:
                state_dict = state_dict_converter(state_dict)
            else:
                state_dict = {i: state_dict[i] for i in state_dict}
            model.load_state_dict(state_dict, assign=True)
            if post_load_hook is not None:
                post_load_hook(model)
            model = enable_vram_management(
                model, module_map, vram_config=vram_config, disk_map=None, vram_limit=vram_limit
            )
        else:
            if post_load_hook is not None:
                raise ValueError("post-load mutations are not supported with disk offload")
            disk_map = DiskMap(path, device, state_dict_converter=state_dict_converter)
            model = enable_vram_management(
                model, module_map, vram_config=vram_config, disk_map=disk_map, vram_limit=vram_limit
            )
    else:
        # Why do we use `DiskMap`?
        # Sometimes a model file contains multiple models,
        # and DiskMap can load only the parameters of a single model,
        # avoiding the need to load all parameters in the file.
        if state_dict is not None:
            pass
        elif use_disk_map:
            state_dict = DiskMap(path, device, torch_dtype=torch_dtype)
        else:
            state_dict = load_state_dict(path, torch_dtype, device)
        # Why do we use `state_dict_converter`?
        # Some models are saved in complex formats,
        # and we need to convert the state dict into the appropriate format.
        if state_dict_converter is not None:
            state_dict = state_dict_converter(state_dict)
        else:
            state_dict = {i: state_dict[i] for i in state_dict}
        # Why does DeepSpeed ZeRO Stage 3 need to be handled separately?
        # Because at this stage, model parameters are partitioned across multiple GPUs.
        # Loading them directly could lead to excessive GPU memory consumption.
        if is_deepspeed_zero3_enabled():
            if not strict:
                raise NotImplementedError("non-strict checkpoint loading is not supported with DeepSpeed ZeRO-3")
            from transformers.integrations.deepspeed import _load_state_dict_into_zero3_model

            _load_state_dict_into_zero3_model(model, state_dict)
        else:
            if strict:
                model.load_state_dict(state_dict, assign=True)
            else:
                incompatible = model.load_state_dict(state_dict, strict=False)
                if incompatible.missing_keys and hasattr(model, "initialize"):
                    model.initialize(incompatible.missing_keys)
        if post_load_hook is not None:
            post_load_hook(model)
        # Why do we call `to()`?
        # Because some models override the behavior of `to()`,
        # especially those from libraries like Transformers.
        model = model.to(dtype=torch_dtype, device=device)
    if hasattr(model, "eval"):
        model = model.eval()
    return model


# ──────────────────────────────────────────────────────────────────────────
# Disk-backed construction — skip init, restore buffers, then wrap with disk_map
# ──────────────────────────────────────────────────────────────────────────


def load_model_with_disk_offload(
    model_class, path, config=None, torch_dtype=torch.bfloat16, device="cpu", state_dict_converter=None, module_map=None
):
    """Construct a model whose inactive weights remain disk-backed.

    Args:
        model_class: PyTorch module class to instantiate on the meta device.
        path: Checkpoint path or paths indexed by ``DiskMap``.
        config: Keyword arguments passed to ``model_class``.
        torch_dtype: Computation dtype.
        device: Preparing and computation device.
        state_dict_converter: Optional checkpoint-key converter.
        module_map: Required source-class to wrapper-class mapping.

    Returns:
        Eval-mode model with disk-backed VRAM wrappers installed.
    """
    if isinstance(path, str):
        path = [path]
    config = {} if config is None else config
    with skip_model_initialization():
        model = model_class(**config)
    if hasattr(model, "eval"):
        model = model.eval()
    disk_map = DiskMap(path, device, state_dict_converter=state_dict_converter)
    _restore_checkpoint_buffers(model, disk_map, torch_dtype=torch_dtype)
    vram_config = {
        "offload_dtype": "disk",
        "offload_device": "disk",
        "onload_dtype": "disk",
        "onload_device": "disk",
        "preparing_dtype": getattr(torch, "float8_e4m3fn", torch.bfloat16),
        "preparing_device": device,
        "computation_dtype": torch_dtype,
        "computation_device": device,
    }
    enable_vram_management(
        model, module_map, vram_config=vram_config, disk_map=disk_map, vram_limit=_disk_offload_vram_limit(device)
    )
    return model


def _disk_offload_vram_limit(device) -> float:
    """Used-memory budget (GiB) for keeping disk-backed weights resident.

    Derived from the actual device capacity (90% of total memory) instead of
    the historical hard-coded ``80`` GiB, which assumed A100/H100-80G and let
    smaller cards keep weights resident until they OOMed. Falls back to the
    legacy value when the device capacity cannot be determined.
    """

    try:
        device_obj = torch.device(device)
        if device_obj.type == "cuda" and torch.cuda.is_available():
            total_gib = torch.cuda.get_device_properties(device_obj).total_memory / (1024**3)
            return total_gib * 0.9
    except Exception:
        pass
    return 80.0


def get_init_context(torch_dtype, device):
    """Return construction contexts: ZeRO-3 partition, or skip-init on meta.

    ZeRO-3 must allocate through ``deepspeed.zero.Init`` so parameters are
    partitioned on CPU before they touch a card. The skip-init path avoids
    random fill of a 10B module that will be overwritten by the checkpoint.
    """
    if is_deepspeed_zero3_enabled():
        import deepspeed
        from transformers.modeling_utils import set_zero3_state

        # Why do we use "deepspeed.zero.Init"?
        # Weight segmentation of the model can be performed on the CPU side
        # and loading the segmented weights onto the computing card
        init_contexts = [deepspeed.zero.Init(remote_device=device, dtype=torch_dtype), set_zero3_state()]
    else:
        # Why do we use `skip_model_initialization`?
        # It skips the random initialization of model parameters,
        # thereby speeding up model loading and avoiding excessive memory usage.
        init_contexts = [skip_model_initialization()]

    return init_contexts


__all__ = ["get_init_context", "load_model", "load_model_with_disk_offload"]
