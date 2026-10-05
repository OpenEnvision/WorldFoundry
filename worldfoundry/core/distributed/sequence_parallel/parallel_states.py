# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.

"""Compatibility shim for the Hunyuan / CleanCode ``parallel_states`` API.

Attention kernels in sub/CleanCode import :func:`get_parallel_state` and
read ``sp_enabled`` / ``sp_group`` / ``sp_rank``. This module maps those
names onto :mod:`.parallel_state` :class:`GroupCoordinator` objects.

Not a second source of truth — it never creates process groups.
:func:`initialize_parallel_state` is a no-op; real init is
:func:`~.parallel_state.maybe_init_distributed_environment_and_model_parallel`.
Do not confuse with :mod:`worldfoundry.core.distributed.model_parallel.sequence_mesh_state`,
which builds a DeviceMesh.

Public surface: :class:`ParallelDims`, :func:`get_parallel_state`,
:func:`initialize_parallel_state`.
"""

from .parallel_state import (
    get_sp_group,
    get_sp_parallel_rank,
    get_sp_world_size,
    model_parallel_is_initialized,
)


# ──────────────────────────────────────────────────────────────────────────
# Read-only view — properties fail closed (sp=1, rank=0) before init
# ──────────────────────────────────────────────────────────────────────────


class ParallelDims:
    """
    Lightweight view over the canonical Wan sequence-parallel state used by
    sub/CleanCode's attention.py.
    """

    @property
    def sp_enabled(self):
        """Returns True if sequence parallelism is enabled (sp_size > 1)."""
        if not model_parallel_is_initialized():
            return False
        return get_sp_world_size() > 1

    @property
    def sp_group(self):
        """Returns the ProcessGroup for sequence parallelism."""
        return get_sp_group().device_group

    @property
    def sp_rank(self):
        """Returns the rank within the sequence parallel group."""
        if not model_parallel_is_initialized():
            return 0
        return get_sp_parallel_rank()

    @property
    def sp(self):
        """Returns the sequence parallel world size."""
        if not model_parallel_is_initialized():
            return 1
        return get_sp_world_size()

    @property
    def dp_enabled(self):
        """Returns True if data parallelism is enabled."""
        return self.sp_enabled  # In FSDP setup, DP is enabled when SP is


_parallel_dims = ParallelDims()


def get_parallel_state():
    """
    Returns the global ParallelDims instance.

    This function is called by attention.py to query SP configuration.
    It delegates to the training framework's parallel state, which is initialized
    by maybe_init_distributed_environment_and_model_parallel() during pipeline setup.
    """
    return _parallel_dims


# Legacy compatibility: initialize_parallel_state is a no-op in this shim
def initialize_parallel_state(sp: int = 1):
    """
    No-op for compatibility. The training framework initializes parallel state
    via maybe_init_distributed_environment_and_model_parallel().
    """
    return _parallel_dims
