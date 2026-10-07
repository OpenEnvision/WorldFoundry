"""Local Kandinsky-6 checkpoint assignment and meta-buffer materialization."""
import logging
import torch
from torch import nn
from safetensors.torch import load_file
logger = logging.getLogger(__name__)

def materialize_meta_buffers(module: nn.Module, device: torch.device) -> None:
    """Allocate non-persistent buffers left on ``meta`` after ``load_state_dict(assign=True)``.

    RoPE / TimeEmbeddings register ``persistent=False`` tables that never appear in
    checkpoints; after a meta-device construct they stay meta until recomputed.
    """
    for mod in module.modules():
        touched = False
        for name, buf in list(mod._buffers.items()):
            if buf is not None and buf.device.type == "meta":
                mod._buffers[name] = torch.empty(buf.shape, dtype=buf.dtype, device=device)
                touched = True
        if touched and hasattr(mod, "reset_parameters"):
            mod.reset_parameters()


def load_bf16_checkpoint(dit: nn.Module, path: str) -> None:
    """Read a local safetensors file and assign it into ``dit``."""
    state_dict = load_file(path, device="cpu")
    state_dict = {
        key: value.to(torch.bfloat16) if value.dtype == torch.float32 else value for key, value in state_dict.items()
    }
    missing, unexpected = dit.load_state_dict(state_dict, strict=True, assign=True)
    if missing:
        logger.warning("missing keys (%d): %s", len(missing), missing[:5])
    if unexpected:
        logger.warning("unexpected keys (%d): %s", len(unexpected), unexpected[:5])
