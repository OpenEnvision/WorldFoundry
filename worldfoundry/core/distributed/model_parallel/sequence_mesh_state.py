"""DeviceMesh state for sequence-parallel world sizes.

:class:`ParallelDims` records ``sp`` vs ``world_size`` so a 1-rank job
does not build a useless mesh. Used by newer SP paths that prefer
DeviceMesh over raw process groups.

Public surface: :class:`ParallelDims`, :func:`initialize_parallel_state`,
:func:`get_parallel_state`.
"""

import os
from dataclasses import dataclass

import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

# ──────────────────────────────────────────────────────────────────────────
# ParallelDims — DeviceMesh snapshot; skip the mesh when world_size == 1
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class ParallelDims:
    """Record sequence-parallel width versus world size and own the DeviceMesh.

    ``world_size == -1`` means "read from the live process group or
    ``WORLD_SIZE``". A 1-rank job leaves ``world_mesh`` as ``None`` so callers
    do not pay for a useless communicator.
    """

    sp: int = 1
    world_size: int = -1

    def __post_init__(self):
        """Validate ``sp`` and optionally build the ``dp × sp`` CUDA mesh."""

        self.sp = int(self.sp)
        if self.sp < 1:
            raise ValueError("sp must be at least 1")
        if self.world_size == -1:
            if dist.is_initialized():
                self.world_size = dist.get_world_size()
            else:
                self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.world_size = int(self.world_size)
        self.world_mesh = None
        if dist.is_initialized() or self.world_size > 1:
            self.build_mesh("cuda")

    def build_mesh(self, device_type):
        """Create a 2-D ``[dp, sp]`` mesh; ``world_size`` must divide by ``sp``."""

        assert self.world_size % self.sp == 0, "world_size must be divisible by sp"
        mesh = init_device_mesh(
            device_type,
            [self.world_size // self.sp, self.sp],
            mesh_dim_names=["dp", "sp"],
        )
        self.world_mesh = mesh
        return mesh

    @property
    def sp_enabled(self):
        """True when sequence parallelism actually splits the sequence."""

        return self.sp > 1

    @property
    def sp_group(self):
        """SP process group, or ``None`` when no mesh was built."""

        if self.world_mesh is None:
            return None
        return self.world_mesh["sp"].get_group()

    @property
    def sp_mesh(self):
        """SP DeviceMesh slice, or ``None`` when no mesh was built."""

        if self.world_mesh is None:
            return None
        return self.world_mesh["sp"]

    @property
    def sp_rank(self):
        """Local SP rank, or global rank / 0 when SP is off."""

        if self.sp_enabled:
            return self.world_mesh["sp"].get_local_rank()
        if dist.is_initialized():
            return dist.get_rank()
        return 0

    @property
    def dp_enabled(self):
        """True when more than one data-parallel replica exists beside SP."""

        return self.world_size // self.sp > 1


# ──────────────────────────────────────────────────────────────────────────
# Process-wide singleton — default is SP=1 so imports stay cheap
# ──────────────────────────────────────────────────────────────────────────

__parallel_dims = None


def initialize_parallel_state(
    sp: int = 1,
):
    """Replace the process-wide :class:`ParallelDims` and return it."""

    global __parallel_dims
    __parallel_dims = ParallelDims(sp=sp)
    return __parallel_dims


def get_parallel_state():
    """Return the singleton, creating an SP-disabled default on first use."""

    if __parallel_dims is None:
        # create default parallel states (without enabling any parallelism)
        initialize_parallel_state()
    return __parallel_dims
