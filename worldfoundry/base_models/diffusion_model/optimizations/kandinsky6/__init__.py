"""Runtime policies applied by the pipeline factory.

Import a stage from its package (``kandinsky.runtime.cache``, ``.offload``,
``.kernels``). Each strategy is its own module inside that package.
This package keeps the shared protocol and the application order.
"""

from worldfoundry.base_models.diffusion_model.optimizations.kandinsky6.base import ORDER, ConfigModel, Runtime

__all__ = ("ORDER", "ConfigModel", "Runtime")
