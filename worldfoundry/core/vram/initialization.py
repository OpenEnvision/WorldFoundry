"""Construction-time weight placement: meta init and skip-init hooks.

Building a 10B module on CPU then ``.to(cuda)`` doubles peak RAM.
:func:`init_weights_on_device` patches ``register_parameter`` so new
weights land on ``meta`` (or a chosen device) and never run the real
initializer. :func:`skip_model_initialization` is the coarser switch for
vendored constructors that ignore factory kwargs.

Global PyTorch hooks are restored in ``finally``. Do not nest these
contexts across threads.
"""

from contextlib import contextmanager

import torch

# ──────────────────────────────────────────────────────────────────────────
# Global hook patch — restore in finally; not safe to nest across threads
# ──────────────────────────────────────────────────────────────────────────


@contextmanager
def init_weights_on_device(device=torch.device("meta"), include_buffers: bool = False):
    """Temporarily redirect newly registered module weights to one device.

    Args:
        device: Destination for parameters created inside the context. The
            default ``meta`` device skips real allocation and initialization.
        include_buffers: Also redirect registered buffers and common tensor
            constructors. Leave disabled unless module construction allocates
            large persistent buffers.

    Yields:
        Control while PyTorch registration hooks are patched.

    Notes:
        Global PyTorch hooks are restored in ``finally``. This context is not
        intended to overlap across threads.
    """
    old_register_parameter = torch.nn.Module.register_parameter
    old_register_buffer = torch.nn.Module.register_buffer if include_buffers else None

    def register_empty_parameter(module, name, param):
        """Register then move so the real initializer never allocates on the default device.

        The Parameter subclass and ``requires_grad`` are preserved; only storage
        is redirected to ``device`` (typically ``meta``).
        """
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            module._parameters[name] = param_cls(module._parameters[name].to(device), **kwargs)

    def register_empty_buffer(module, name, buffer, persistent=True):
        """Same redirect as parameters, but only when ``include_buffers`` is on."""
        old_register_buffer(module, name, buffer, persistent=persistent)
        if buffer is not None:
            module._buffers[name] = module._buffers[name].to(device)

    def patch_tensor_constructor(fn):
        """Force ``device=`` on ``empty`` / ``zeros`` / ``ones`` / ``full``.

        Constructors ignore ``register_buffer``; without this patch a vendored
        ``__init__`` still allocates a full CUDA tensor during skip-init.
        """

        def wrapper(*args, **kwargs):
            """Inject the context device; caller-supplied ``device`` is overwritten."""
            kwargs["device"] = device
            return fn(*args, **kwargs)

        return wrapper

    patched_constructors = {}
    if include_buffers:
        patched_constructors = {name: getattr(torch, name) for name in ("empty", "zeros", "ones", "full")}

    try:
        torch.nn.Module.register_parameter = register_empty_parameter
        if include_buffers:
            torch.nn.Module.register_buffer = register_empty_buffer
            for name in patched_constructors:
                setattr(torch, name, patch_tensor_constructor(getattr(torch, name)))
        yield
    finally:
        torch.nn.Module.register_parameter = old_register_parameter
        if include_buffers and old_register_buffer is not None:
            torch.nn.Module.register_buffer = old_register_buffer
            for name, old_fn in patched_constructors.items():
                setattr(torch, name, old_fn)


def skip_model_initialization(device=torch.device("meta")):
    """Return an allocation-skipping context for constructing checkpoint-backed models."""
    return init_weights_on_device(device=device)


__all__ = ["init_weights_on_device", "skip_model_initialization"]
