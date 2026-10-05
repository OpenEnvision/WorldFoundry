import os

import torch


def _rank_local_cuda_device():
    if not torch.cuda.is_available():
        return None
    for env_name in ("LOCAL_RANK", "SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK"):
        value = os.environ.get(env_name)
        if value is None:
            continue
        try:
            local_rank = int(value)
        except ValueError:
            continue
        if local_rank >= 0:
            return torch.device(f"cuda:{local_rank % torch.cuda.device_count()}")
    current = torch.cuda.current_device()
    return torch.device(f"cuda:{current}")


def _set_rank_local_cuda_device():
    device = _rank_local_cuda_device()
    if device is not None:
        torch.cuda.set_device(device)
    return device


def _trim_cpu_allocator():
    """Best-effort release of freed glibc heap pages back to the OS."""
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6")
        malloc_trim = getattr(libc, "malloc_trim", None)
        if malloc_trim is not None:
            malloc_trim(0)
    except Exception:
        pass


def _release_cached_memory():
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass
    _trim_cpu_allocator()
