"""Load-time quantization. One module per method.

A model factory chooses which component receives a method. Removing a method
means reloading that component.
"""

from worldfoundry.base_models.diffusion_model.optimizations.kandinsky6.quantize.nf4 import qwen_load_kwargs

__all__ = ("qwen_load_kwargs",)
