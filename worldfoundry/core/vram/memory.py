"""VRAM accounting and whole-model swap helpers.

:class:`DynamicSwapInstaller` rewrites ``__getattr__`` so parameters/buffers
move onto the computation device on first access and can live on CPU
otherwise. That is cheaper than wrapping every submodule when a *whole*
encoder or VAE is used once per request.

``gpu`` is a lazy module attribute (not an import-time constant) so importing
this file does not create a CUDA context on GPU 0 — the same CC-25 rule as
attention backend probing.

:func:`load_model_as_complete` / :func:`unload_complete_models` keep a
strong-ref list on purpose: dropping the Python name without unload leaves
the module on GPU until the next explicit unload.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────
# Process-wide CPU sentinel and complete-module registry (strong refs on purpose)
# ──────────────────────────────────────────────────────────────────────────

cpu = torch.device("cpu")

#: Modules loaded via :func:`load_model_as_complete`. Holds strong references
#: on purpose so :func:`unload_complete_models` can move them back to CPU;
#: callers that drop a model must call ``unload_complete_models`` themselves
#: or the module stays resident until the next unload.
gpu_complete_modules: list[torch.nn.Module] = []


def _default_gpu_device() -> torch.device:
    """Resolve the current CUDA device lazily.

    CC-25: this used to be a module constant evaluated at import time, which
    created a CUDA context on GPU 0 for every importer (including fork-based
    dataloader workers) and froze the device chosen before any
    ``torch.cuda.set_device`` call.
    """

    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return cpu


def __getattr__(name: str):
    """Resolve ``gpu`` lazily so importing this module does not create a CUDA context.

    Failure: :exc:`AttributeError` for any other name. ``from ... import gpu``
    still works; the device is the *current* CUDA index, not a frozen
    import-time ``cuda:0``.
    """
    # Lazy, never-frozen module attribute so ``from ... import gpu`` keeps
    # working without initializing CUDA at import time of this module.
    if name == "gpu":
        return _default_gpu_device()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class DynamicSwapInstaller:
    """Rewrite ``__getattr__`` so parameters move to the compute device on access.

    Cheaper than wrapping every submodule when a whole encoder/VAE is
    used once per request. Re-install is a no-op so uninstall can still
    restore the original class.
    """

    # ──────────────────────────────────────────────────────────────────────
    # Class rewrite — cheaper than wrapping every submodule for a whole encoder
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _install_module(module: torch.nn.Module, **kwargs) -> None:
        """Rewrite ``module.__class__`` so parameter access copies onto the compute device.

        Re-install is a no-op: backing up the already-hacked class would make
        :meth:`_uninstall_module` unable to restore the real type.
        """
        if "forge_backup_original_class" in module.__dict__:
            # Re-installing would back up the already-hacked class, making
            # uninstall unable to ever restore the real class.
            return
        original_class = module.__class__
        module.__dict__["forge_backup_original_class"] = original_class

        def hacked_get_attr(self, name: str):
            """Return a device-copied Parameter/buffer; fall through to the original class.

            A new ``nn.Parameter`` is constructed so ``requires_grad`` survives
            ``.to()``. Storage in ``_parameters`` stays on the offload device.
            """
            if "_parameters" in self.__dict__:
                parameters = self.__dict__["_parameters"]
                if name in parameters:
                    parameter = parameters[name]
                    if parameter is None:
                        return None
                    if parameter.__class__ == torch.nn.Parameter:
                        return torch.nn.Parameter(parameter.to(**kwargs), requires_grad=parameter.requires_grad)
                    return parameter.to(**kwargs)
            if "_buffers" in self.__dict__:
                buffers = self.__dict__["_buffers"]
                if name in buffers:
                    return buffers[name].to(**kwargs)
            return super(original_class, self).__getattr__(name)

        module.__class__ = type(
            "DynamicSwap_" + original_class.__name__,
            (original_class,),
            {"__getattr__": hacked_get_attr},
        )

    @staticmethod
    def _uninstall_module(module: torch.nn.Module) -> None:
        """Restore the class saved at install time; no-op if never installed."""
        if "forge_backup_original_class" in module.__dict__:
            module.__class__ = module.__dict__.pop("forge_backup_original_class")

    @staticmethod
    def install_model(model: torch.nn.Module, **kwargs) -> None:
        """Install swap on every submodule, including the root."""
        for module in model.modules():
            DynamicSwapInstaller._install_module(module, **kwargs)

    @staticmethod
    def uninstall_model(model: torch.nn.Module) -> None:
        """Undo :meth:`install_model` on every submodule that was rewritten."""
        for module in model.modules():
            DynamicSwapInstaller._uninstall_module(module)


# ──────────────────────────────────────────────────────────────────────────
# Diffusers device probe and free-memory accounting
# ──────────────────────────────────────────────────────────────────────────


def patched_diffusers_current_device(model: torch.nn.Module, target_device: torch.device) -> None:
    """Move one representative tensor so Diffusers' device probe sees ``target_device``.

    Diffusers infers ``.device`` from the first parameter it finds. A
    swap-installed encoder can report CPU while compute is CUDA; moving
    ``scale_shift_table`` or the first weighted child is enough for that
    probe without migrating the whole tree.
    """
    if hasattr(model, "scale_shift_table"):
        model.scale_shift_table.data = model.scale_shift_table.data.to(target_device)
        return

    for _, module in model.named_modules():
        if hasattr(module, "weight"):
            module.to(target_device)
            return


fake_diffusers_current_device = patched_diffusers_current_device


def get_cuda_free_memory_gb(device=None) -> float:
    """Return free + inactive-reserved CUDA memory in GiB, or ``0.0`` without CUDA.

    Inactive reserved bytes are reusable by the allocator, so treating only
    ``mem_get_info`` free as capacity under-counts and rejects valid moves.
    """
    if not torch.cuda.is_available():
        return 0.0
    if device is None:
        device = _default_gpu_device()

    memory_stats = torch.cuda.memory_stats(device)
    bytes_active = memory_stats["active_bytes.all.current"]
    bytes_reserved = memory_stats["reserved_bytes.all.current"]
    bytes_free_cuda, _ = torch.cuda.mem_get_info(device)
    bytes_inactive_reserved = bytes_reserved - bytes_active
    bytes_total_available = bytes_free_cuda + bytes_inactive_reserved
    return bytes_total_available / (1024**3)


def log_gpu_memory(stage: str, device=None, rank: int = 0) -> None:
    """Log used / free / total GiB for *stage*; CUDA-unavailable is still logged."""
    if not torch.cuda.is_available():
        logger.info("[rank %s] [GPU Memory][%s] CUDA unavailable", rank, stage)
        return
    if device is None:
        device = _default_gpu_device()

    free_gb = get_cuda_free_memory_gb(device)
    total_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    used_gb = total_gb - free_gb
    logger.info(
        "[rank %s] [GPU Memory][%s] Used: %.2f GB | Free: %.2f GB | Total: %.2f GB",
        rank,
        stage,
        used_gb,
        free_gb,
        total_gb,
    )


# ──────────────────────────────────────────────────────────────────────────
# Atomic move with rollback — refuse mixed-device or meta modules
# ──────────────────────────────────────────────────────────────────────────


def move_model_to_device_with_memory_preservation(
    model: torch.nn.Module,
    target_device,
    preserved_memory_gb: float = 0,
) -> None:
    """Move *model* atomically if free CUDA memory stays above *preserved_memory_gb*.

    Failure: :exc:`ValueError` for a negative preserve budget; :exc:`RuntimeError`
    when the preflight would dip below the budget, the module spans devices,
    or it is still on ``meta`` (use ``to_empty`` instead).
    """
    logger.info(
        "Moving %s to %s with preserved memory: %s GB", model.__class__.__name__, target_device, preserved_memory_gb
    )
    if preserved_memory_gb < 0:
        raise ValueError("preserved_memory_gb must be non-negative")

    target = _normalize_device(target_device)
    original = _uniform_model_device(model)
    if target.type == "cuda":
        required_gb = _model_transfer_bytes(model, target) / (1024**3)
        available_gb = get_cuda_free_memory_gb(target)
        if available_gb - required_gb < preserved_memory_gb:
            raise RuntimeError(
                f"Insufficient memory to move {model.__class__.__name__} atomically to {target}: "
                f"available={available_gb:.2f} GiB, required~={required_gb:.2f} GiB, "
                f"preserve={preserved_memory_gb:.2f} GiB"
            )

    _move_model_with_rollback(model, target=target, original=original)


def offload_model_from_device_for_memory_preservation(
    model: torch.nn.Module,
    target_device,
    preserved_memory_gb: float = 0,
) -> None:
    """Offload *model* to CPU only when free CUDA memory is below *preserved_memory_gb*.

    No-op when headroom already meets the budget. Failure matches
    :func:`move_model_to_device_with_memory_preservation` for mixed-device
    or meta modules.
    """
    logger.info(
        "Offloading %s from %s to preserve memory: %s GB",
        model.__class__.__name__,
        target_device,
        preserved_memory_gb,
    )
    if preserved_memory_gb < 0:
        raise ValueError("preserved_memory_gb must be non-negative")
    if get_cuda_free_memory_gb(target_device) >= preserved_memory_gb:
        return

    original = _uniform_model_device(model)
    _move_model_with_rollback(model, target=cpu, original=original)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _normalize_device(device) -> torch.device:
    """Bind bare ``cuda`` to the current device index so capacity queries match."""
    normalized = torch.device(device)
    if normalized.type == "cuda" and normalized.index is None and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return normalized


def _model_tensors(model: torch.nn.Module):
    """Yield parameters, their ``.grad`` if present, and buffers.

    ``Module.to`` migrates gradients with parameters; omitting them would
    pass the capacity preflight and then OOM mid-move.
    """
    for parameter in model.parameters():
        yield parameter
        # Module._apply (and therefore Module.to) migrates existing gradients
        # alongside parameters. Include them in both capacity preflight and
        # device-uniformity validation.
        if parameter.grad is not None:
            yield parameter.grad
    yield from model.buffers()


def _uniform_model_device(model: torch.nn.Module) -> torch.device:
    """Return the single device all tensors share, or refuse the move.

    Failure: :exc:`RuntimeError` when tensors span devices (a device map or
    partial offload) or when any tensor is still ``meta``.
    """
    devices = {tensor.device for tensor in _model_tensors(model)}
    if not devices:
        return cpu
    if len(devices) != 1:
        rendered = ", ".join(sorted(str(device) for device in devices))
        raise RuntimeError(
            f"Refusing to migrate mixed-device {model.__class__.__name__}; "
            f"found tensors on: {rendered}"
        )
    device = next(iter(devices))
    if device.type == "meta":
        raise RuntimeError("Meta-initialized modules require to_empty() and cannot use the VRAM preservation mover")
    return device


def _model_transfer_bytes(model: torch.nn.Module, target: torch.device) -> int:
    """Bytes that would be allocated on *target* (already-resident tensors skip)."""
    return sum(
        tensor.numel() * tensor.element_size()
        for tensor in _model_tensors(model)
        if tensor.device != target
    )


def _move_model_with_rollback(
    model: torch.nn.Module,
    *,
    target: torch.device,
    original: torch.device,
) -> None:
    """``model.to(target)`` with rollback to *original* if the move is partial or raises.

    Failure: :exc:`RuntimeError` when rollback also fails (model may be
    split across devices); the original move error is chained as cause.
    """
    if target == original:
        return
    try:
        model.to(device=target)
        moved = _uniform_model_device(model)
        if moved != target:
            raise RuntimeError(f"migration ended on {moved}, expected {target}")
    except Exception as move_error:
        try:
            model.to(device=original)
            restored = _uniform_model_device(model)
            if restored != original:
                raise RuntimeError(f"rollback ended on {restored}, expected {original}")
        except Exception as rollback_error:
            raise RuntimeError(
                f"Failed to move {model.__class__.__name__} to {target}; "
                f"rollback to {original} also failed: {rollback_error}"
            ) from move_error
        raise


# ──────────────────────────────────────────────────────────────────────────
# Complete-module residency — registry holds strong refs until explicit unload
# ──────────────────────────────────────────────────────────────────────────


def unload_complete_models(*models: torch.nn.Module) -> None:
    """Move the registry and any extra *models* to CPU, then clear the registry."""
    for model in gpu_complete_modules + list(models):
        model.to(device=cpu)
        logger.info("Unloaded %s as complete.", model.__class__.__name__)

    gpu_complete_modules.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model_as_complete(model: torch.nn.Module, target_device, unload: bool = True) -> None:
    """Move *model* to *target_device* and register it; optionally unload prior completes first.

    The registry keeps a strong reference so dropping the Python name does
    not free GPU memory until :func:`unload_complete_models` runs.
    """
    if unload:
        unload_complete_models()

    model.to(device=target_device)
    logger.info("Loaded %s to %s as complete.", model.__class__.__name__, target_device)

    gpu_complete_modules.append(model)


__all__ = [
    "DynamicSwapInstaller",
    "cpu",
    "fake_diffusers_current_device",
    "get_cuda_free_memory_gb",
    "gpu",
    "gpu_complete_modules",
    "load_model_as_complete",
    "log_gpu_memory",
    "move_model_to_device_with_memory_preservation",
    "offload_model_from_device_for_memory_preservation",
    "patched_diffusers_current_device",
    "unload_complete_models",
]
