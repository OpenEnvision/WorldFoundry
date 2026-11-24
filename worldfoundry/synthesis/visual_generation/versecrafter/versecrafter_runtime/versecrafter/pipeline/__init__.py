import importlib.util

from .pipeline_wan_versecrafter import WanVerseCrafterPipeline

if importlib.util.find_spec("paifuser") is not None:
    # --------------------------------------------------------------- #
    #   Sparse Attention
    # --------------------------------------------------------------- #
    from paifuser.ops import sparse_reset

    WanVerseCrafterPipeline.__call__ = sparse_reset(WanVerseCrafterPipeline.__call__)
